from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from gtex_explorer.analysis.gene_panels import load_cv_gene_rankings
from gtex_explorer.analysis.selector_diagnostics import load_selector_diagnostics


from gtex_explorer.paths import APP_ROOT as APP_DIR
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "model_bundle"


@dataclass
class InputAudit:
    source_orientation: str
    sample_count: int
    supplied_gene_count: int
    matched_gene_count: int
    missing_gene_count: int
    duplicate_gene_count: int
    negative_value_count: int
    missing_value_count: int
    infinite_value_count: int
    nonfinite_value_count: int
    matched_fraction: float
    appears_log_transformed: bool
    ready: bool
    message: str
    invalid_numeric_value_count: int
    below_threshold_sample_count: int
    completeness_threshold: float
    low_positive_sample_count: int
    positive_expression_threshold: float
    sample_quality: pd.DataFrame


def strip_ensembl_version(gene_id: str) -> str:
    """Return the stable Ensembl identifier before its optional version suffix."""

    return str(gene_id).strip().split(".", 1)[0].upper()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_optional_csv(filename: str) -> pd.DataFrame:
    """Load a notebook export when available without breaking older app snapshots."""

    path = DATA_DIR / filename
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_reference_tables() -> dict[str, pd.DataFrame | dict]:
    """Load the compact Part A result tables used by the application pages."""

    tables = {
        "run_summary": load_json(DATA_DIR / "run_summary.json"),
        "cv_summary": pd.read_csv(DATA_DIR / "experiment_cv_summary.csv"),
        "cv_folds": pd.read_csv(DATA_DIR / "experiment_cv_folds.csv"),
        "selector_cv_summary": pd.read_csv(
            DATA_DIR / "feature_selection_cv_summary.csv"
        ),
        "selector_cv_folds": pd.read_csv(
            DATA_DIR / "feature_selection_cv_folds.csv"
        ),
        "baseline_cv_folds": pd.read_csv(
            DATA_DIR / "xgboost_all_5000_cv_folds.csv"
        ),
        "experiment_validation": load_json(
            DATA_DIR / "experiment_validation_report.json"
        ),
        "selected_genes": pd.read_csv(DATA_DIR / "selected_genes.csv"),
        "final_test": pd.read_csv(DATA_DIR / "final_test_summary.csv"),
        "final_holdout": load_optional_csv("final_holdout_comparison.csv"),
        "final_holdout_predictions": load_optional_csv(
            "final_holdout_predictions.csv"
        ),
        "final_holdout_per_tissue": load_optional_csv(
            "final_holdout_per_tissue_comparison.csv"
        ),
        "per_tissue": pd.read_csv(DATA_DIR / "test_per_tissue_xgboost.csv"),
        "labels": pd.read_csv(DATA_DIR / "label_mapping.csv"),
        "config_status": pd.read_csv(
            DATA_DIR / "feature_selection_configuration_status.csv"
        ),
        "stability": pd.read_csv(
            DATA_DIR / "feature_selection_stability_summary.csv"
        ),
        "selector_fit": pd.read_csv(DATA_DIR / "final_selector_fit_diagnostics.csv"),
    }
    # Part A has one deployed selected model. Reject stale/mixed exports rather
    # than silently displaying a removed DummyClassifier alongside it.
    final_test = tables["final_test"]
    if len(final_test) != 1 or not final_test["classifier"].eq("xgboost").all():
        raise ValueError("final_test_summary.csv bukan ekspor Part A terbaru (satu model XGBoost terpilih).")
    summary = tables["run_summary"]
    row = final_test.iloc[0]
    if row["selector_id"] != summary["best_selector_id"] or int(row["genes"]) != int(summary["best_feature_count"]):
        raise ValueError("Konfigurasi final tidak cocok dengan run_summary.json.")
    tables["cv_gene_rankings"] = load_cv_gene_rankings(DATA_DIR, summary, tables["cv_folds"])
    tables["selector_diagnostics"] = load_selector_diagnostics(DATA_DIR, summary)
    return tables


def model_bundle_available() -> bool:
    return (
        (MODEL_DIR / "preprocessor.joblib").is_file()
        and (MODEL_DIR / "xgboost_model.json").is_file()
    )


def load_model_bundle() -> tuple[dict, XGBClassifier]:
    """Load preprocessing constants and the trained final XGBoost model."""

    if not model_bundle_available():
        raise FileNotFoundError(
            "Model bundle belum tersedia. Jalankan export_model_bundle.py terlebih dahulu."
        )

    preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")
    classifier = XGBClassifier()
    classifier.load_model(MODEL_DIR / "xgboost_model.json")
    return preprocessor, classifier


def _read_delimited_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV, TSV, or a small GCT file without relying on its extension alone."""

    lower_name = filename.lower()
    text_start = raw_bytes[:200].decode("utf-8", errors="ignore")
    is_gct = lower_name.endswith(".gct") or text_start.lstrip().startswith("#1.2")
    separator = "\t" if lower_name.endswith((".tsv", ".txt", ".gct")) else ","
    skiprows = 2 if is_gct else 0
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=separator,
        skiprows=skiprows,
        dtype_backend="numpy_nullable",
    )


def parse_expression_file(file_object: BinaryIO, filename: str, *, model_genes=None) -> pd.DataFrame:
    """Convert sample-by-gene or gene-by-sample input into sample-by-gene form."""

    raw_bytes = file_object.read()
    try:
        table = _read_delimited_bytes(raw_bytes, filename)
    except pd.errors.EmptyDataError as error:
        raise ValueError('Berkas kosong atau tidak berisi tabel. Unggah berkas dengan kolom ID dan nilai ekspresi TPM.') from error
    if table.empty or table.shape[1] < 2:
        raise ValueError("Berkas ekspresi kosong atau tidak memiliki kolom gen/sampel.")

    first_column = str(table.columns[0])
    if table.iloc[:, 0].isna().any() or table.iloc[:, 0].astype(str).str.strip().eq("").any():
        raise ValueError("Identifier gen/sampel pada kolom pertama tidak boleh kosong.")
    first_values = table.iloc[:, 0].astype(str).str.strip()
    gene_row_fraction = first_values.str.upper().str.startswith("ENSG").mean()
    model_ids = {strip_ensembl_version(gene) for gene in (model_genes or [])}
    row_matches = any(strip_ensembl_version(value) in model_ids for value in first_values)
    column_matches = any(strip_ensembl_version(value) in model_ids for value in table.columns[1:])
    if row_matches and column_matches:
        raise ValueError('Orientasi matriks ambigu. ID gen model ditemukan pada kedua sumbu.')

    if row_matches or (not column_matches and gene_row_fraction >= 0.5):
        # GCT commonly includes a Description column that is not an expression sample.
        non_expression_columns = {
            first_column,
            "Description",
            "description",
            "gene_name",
        }
        expression_columns = [
            column for column in table.columns if column not in non_expression_columns
        ]
        matrix = table.set_index(first_column)[expression_columns].T
        matrix.index.name = "SAMPID"
        source_orientation = "gen × sampel"
    else:
        matrix = table.set_index(first_column)
        matrix.index.name = "SAMPID"
        source_orientation = "sampel × gen"

    matrix.index = matrix.index.astype(str).str.strip()
    matrix.columns = pd.Index([str(column).strip() for column in matrix.columns])
    if matrix.index.has_duplicates:
        duplicate_samples = matrix.index[matrix.index.duplicated()].unique().tolist()
        raise ValueError(f"SAMPID duplikat ditemukan: {', '.join(duplicate_samples[:5])}")

    matrix, invalid_count = _numeric_expression(matrix)
    if invalid_count:
        raise ValueError(f"Ditemukan {invalid_count:,} nilai ekspresi non-numerik. Gunakan angka TPM atau nilai kosong, bukan teks.")
    matrix.attrs["source_orientation"] = source_orientation
    return matrix


def _numeric_expression(matrix):
    # One vectorized conversion avoids a Python conversion for every gene column.
    original = pd.Series(matrix.to_numpy(dtype=object).reshape(-1))
    original = original.mask(original.astype('string').str.fullmatch(r'\s*', na=False))
    numeric = pd.to_numeric(original, errors='coerce')
    invalid = int((original.notna() & numeric.isna()).sum())
    values = numeric.to_numpy(dtype=float, na_value=np.nan).reshape(matrix.shape)
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns), invalid


def audit_and_align_input(
    input_matrix: pd.DataFrame,
    preprocessor: dict,
    minimum_observed_fraction: float = 0.90,
    minimum_positive_fraction: float = 0.90,
) -> tuple[pd.DataFrame, InputAudit]:
    """Align model genes and impute missing values. Coverage is advisory only.

    minimum_observed_fraction controls the per-sample completeness reminder.
    Observed zero TPM is available data, independently of positive expression.
    """
    for threshold in (minimum_observed_fraction, minimum_positive_fraction):
        if (not isinstance(threshold, (int, float, np.integer, np.floating))
                or isinstance(threshold, (bool, np.bool_))
                or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError("Nilai ambang pengingat harus berada antara 0 dan 1.")

    required_genes = [str(gene) for gene in preprocessor["selected_genes"]]
    required_base_ids = [strip_ensembl_version(gene) for gene in required_genes]

    supplied_by_base: dict[str, str] = {}
    duplicate_base_ids: set[str] = set()
    for supplied_gene in input_matrix.columns:
        base_id = strip_ensembl_version(supplied_gene)
        if base_id in supplied_by_base:
            duplicate_base_ids.add(base_id)
        else:
            supplied_by_base[base_id] = str(supplied_gene)

    numeric_matrix, invalid_count = _numeric_expression(input_matrix)
    raw_values = numeric_matrix.to_numpy(dtype=float, na_value=np.nan)
    negative_count = int(np.nansum(raw_values < 0))
    missing_value_count = int(np.isnan(raw_values).sum())
    infinite_value_count = int(np.isinf(raw_values).sum())
    nonfinite_count = int((~np.isfinite(raw_values)).sum())

    aligned = pd.DataFrame(
        np.tile(
            np.asarray(preprocessor["raw_tpm_medians"], dtype=float),
            (len(input_matrix), 1),
        ),
        index=input_matrix.index,
        columns=required_genes,
    )

    matched_count = 0
    observed_count = np.zeros(len(input_matrix), dtype=int)
    expressed_count = np.zeros(len(input_matrix), dtype=int)
    for gene_position, (required_gene, base_id) in enumerate(
        zip(required_genes, required_base_ids)
    ):
        supplied_gene = supplied_by_base.get(base_id)
        if supplied_gene is None or base_id in duplicate_base_ids:
            continue
        # Nullable integer columns must become float before a fractional median
        # is inserted (e.g. an integer-valued column containing missing values).
        values = numeric_matrix[supplied_gene].astype(float).replace(
            [np.inf, -np.inf], np.nan
        )
        median_value = float(preprocessor["raw_tpm_medians"][gene_position])
        aligned[required_gene] = values.fillna(median_value).astype(float)
        observed = values.notna().to_numpy()
        observed_count += observed
        expressed_count += values.gt(0).fillna(False).to_numpy(dtype=bool)
        matched_count += 1

    missing_count = len(required_genes) - matched_count
    matched_fraction = matched_count / max(len(required_genes), 1)
    finite_values = raw_values[np.isfinite(raw_values)]
    appears_log = bool(
        finite_values.size > 0
        and np.nanpercentile(finite_values, 99) < 30
        and np.nanmax(finite_values) < 100
    )
    # Count availability before imputation. Neither coverage nor zero TPM blocks inference.
    observed_fraction = observed_count / max(len(required_genes), 1)
    meets_threshold = observed_fraction >= minimum_observed_fraction
    # This denominator is the uploaded gene count, independent of model IDs.
    uploaded_positive_count = ((raw_values > 0) & np.isfinite(raw_values)).sum(axis=1)
    uploaded_positive_fraction = uploaded_positive_count / max(input_matrix.shape[1], 1)
    meets_positive_threshold = uploaded_positive_fraction >= minimum_positive_fraction
    sample_quality = pd.DataFrame({
        "SAMPID": input_matrix.index.astype(str),
        "observed_model_genes": observed_count,
        "imputed_model_genes": len(required_genes) - observed_count,
        "observed_fraction": observed_fraction,
        "positive_expression_genes": expressed_count,
        "meets_reminder_threshold": meets_threshold,
        "uploaded_positive_genes": uploaded_positive_count,
        "uploaded_positive_fraction": uploaded_positive_fraction,
        "meets_positive_reminder_threshold": meets_positive_threshold,
    })
    below_threshold_sample_count = int((~meets_threshold).sum())
    ready = bool(
        negative_count == 0
        and infinite_value_count == 0
        and len(duplicate_base_ids) == 0
        and invalid_count == 0
        and len(input_matrix) > 0
    )

    if invalid_count:
        message = f"Ditemukan {invalid_count:,} nilai ekspresi non-numerik."
    elif len(input_matrix) == 0:
        message = "Matriks tidak memuat sampel."
    elif negative_count:
        message = "Nilai TPM negatif ditemukan."
    elif infinite_value_count:
        message = "Nilai tak hingga ditemukan dan input tidak dapat diprediksi."
    elif duplicate_base_ids:
        message = "Identifier Ensembl duplikat ditemukan setelah versi dihapus."
    elif not observed_count.any():
        message = "Tidak ada nilai gen model yang tersedia. Seluruh input model diisi dengan median data training tersimpan. Prediksi tetap dapat dijalankan."
    elif below_threshold_sample_count:
        sample_ids = sample_quality.loc[~sample_quality["meets_reminder_threshold"], "SAMPID"].head(5).tolist()
        message = (
            f"{below_threshold_sample_count} sampel di bawah ambang pengingat "
            f"kelengkapan {minimum_observed_fraction:.0%}: {', '.join(sample_ids)}. "
            "Nilai yang tidak tersedia diisi dengan median data training tersimpan. Prediksi tetap dapat dijalankan."
        )
    else:
        message = "Input dapat diprediksi."

    aligned.attrs["input_audit_ready"] = ready
    return aligned, InputAudit(
        source_orientation=str(
            input_matrix.attrs.get("source_orientation", "sampel × gen")
        ),
        sample_count=len(input_matrix),
        supplied_gene_count=input_matrix.shape[1],
        matched_gene_count=matched_count,
        missing_gene_count=missing_count,
        duplicate_gene_count=len(duplicate_base_ids),
        negative_value_count=negative_count,
        missing_value_count=missing_value_count,
        infinite_value_count=infinite_value_count,
        nonfinite_value_count=nonfinite_count,
        matched_fraction=matched_fraction,
        appears_log_transformed=appears_log,
        ready=ready,
        message=message,
        invalid_numeric_value_count=invalid_count,
        below_threshold_sample_count=below_threshold_sample_count,
        completeness_threshold=float(minimum_observed_fraction),
        low_positive_sample_count=int((~meets_positive_threshold).sum()),
        positive_expression_threshold=float(minimum_positive_fraction),
        sample_quality=sample_quality,
    )


def _prediction_iteration_range(
    classifier: XGBClassifier, boosting_rounds: int | None,
) -> tuple[int, int]:
    """Validate a prefix of the saved model without changing its parameters."""
    total_rounds = int(classifier.get_booster().num_boosted_rounds())
    rounds = total_rounds if boosting_rounds is None else boosting_rounds
    if (
        isinstance(rounds, (bool, np.bool_))
        or not isinstance(rounds, (int, np.integer))
        or not 1 <= rounds <= total_rounds
    ):
        raise ValueError(f"Jumlah putaran harus berupa bilangan bulat dari 1 sampai {total_rounds}.")
    return 0, int(rounds)


def predict_tissues(
    aligned_tpm: pd.DataFrame,
    preprocessor: dict,
    classifier: XGBClassifier,
    *,
    boosting_rounds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the frozen preprocessing and the chosen prefix of saved boosting rounds."""

    if aligned_tpm.attrs.get("input_audit_ready") is False:
        raise ValueError("Input belum lolos audit. Prediksi diblokir.")
    iteration_range = _prediction_iteration_range(classifier, boosting_rounds)
    scaled_values = transform_aligned_tpm(aligned_tpm, preprocessor)

    probabilities = classifier.predict_proba(scaled_values, iteration_range=iteration_range)
    classes = np.asarray(preprocessor["classes"], dtype=object)
    predicted_indices = probabilities.argmax(axis=1)
    predictions = pd.DataFrame(
        {
            "SAMPID": aligned_tpm.index,
            "predicted_tissue": classes[predicted_indices],
            "top_probability": probabilities.max(axis=1),
            "boosting_rounds": iteration_range[1],
        }
    )
    probability_table = pd.DataFrame(
        probabilities,
        index=aligned_tpm.index,
        columns=classes,
    )
    probability_table.index.name = "SAMPID"
    return predictions, probability_table


def transform_aligned_tpm(
    aligned_tpm: pd.DataFrame,
    preprocessor: dict,
) -> np.ndarray:
    """Apply the frozen log and scaling steps used during model inference."""

    values = aligned_tpm.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Matriks TPM terselaraskan masih memuat nilai non-finite.")
    if (values < 0).any():
        raise ValueError("Transformasi log2(TPM+1) memerlukan nilai TPM non-negatif.")
    if len(values) == 0:
        raise ValueError("Matriks tidak memuat sampel.")
    log_values = np.log2(values + 1.0)
    scaled_values = (
        log_values - np.asarray(preprocessor["scaler_mean"], dtype=float)
    ) / np.asarray(preprocessor["scaler_scale"], dtype=float)
    return scaled_values


def explain_prediction(
    aligned_tpm: pd.DataFrame,
    sample_id: str,
    preprocessor: dict,
    classifier: XGBClassifier,
    limit: int = 8,
    *,
    boosting_rounds: int | None = None,
) -> pd.DataFrame:
    """Return the strongest XGBoost feature contributions for one prediction.

    Contributions are model-local evidence on the margin scale. They explain
    the model output for the selected sample and must not be read as causal
    biological effects.
    """

    if sample_id not in aligned_tpm.index:
        raise KeyError(f"SAMPID tidak ditemukan: {sample_id}")

    iteration_range = _prediction_iteration_range(classifier, boosting_rounds)
    sample_tpm = aligned_tpm.loc[[sample_id]]
    scaled_values = transform_aligned_tpm(sample_tpm, preprocessor)
    probabilities = classifier.predict_proba(scaled_values, iteration_range=iteration_range)
    predicted_class_index = int(probabilities[0].argmax())
    contributions = classifier.get_booster().predict(
        xgb.DMatrix(scaled_values),
        pred_contribs=True,
        iteration_range=iteration_range,
    )
    class_contributions = contributions[0, predicted_class_index, :-1]
    genes = np.asarray(preprocessor["selected_genes"], dtype=object)
    raw_tpm = sample_tpm.iloc[0].to_numpy(dtype=float)

    explanation = pd.DataFrame(
        {
            "gene": genes,
            "raw_tpm": raw_tpm,
            "contribution": class_contributions,
        }
    )
    explanation["absolute_contribution"] = explanation["contribution"].abs()
    explanation["direction"] = np.where(
        explanation["contribution"] >= 0,
        "Mendukung kelas",
        "Menurunkan skor",
    )
    explanation = explanation.sort_values(
        ["absolute_contribution", "gene"],
        ascending=[False, True],
    ).head(limit)
    explanation.insert(0, "rank", np.arange(1, len(explanation) + 1))
    return explanation.reset_index(drop=True)
