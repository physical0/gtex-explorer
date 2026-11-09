"""Check an expression matrix and align it to a model's genes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gtex_explorer.inference.parsing import _numeric_expression, strip_ensembl_version


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

    # Column positions, not labels: a frame passed in directly may repeat a label.
    supplied_by_base: dict[str, int] = {}
    duplicate_base_ids: set[str] = set()
    for column_position, supplied_gene in enumerate(input_matrix.columns):
        base_id = strip_ensembl_version(supplied_gene)
        if base_id in supplied_by_base:
            duplicate_base_ids.add(base_id)
        else:
            supplied_by_base[base_id] = column_position

    numeric_matrix, invalid_count = _numeric_expression(input_matrix)
    raw_values = numeric_matrix.to_numpy(dtype=float, na_value=np.nan)
    negative_count = int(np.nansum(raw_values < 0))
    missing_value_count = int(np.isnan(raw_values).sum())
    infinite_value_count = int(np.isinf(raw_values).sum())
    nonfinite_count = int((~np.isfinite(raw_values)).sum())

    medians = np.asarray(preprocessor["raw_tpm_medians"], dtype=float)
    aligned_values = np.tile(medians, (len(input_matrix), 1))
    matches = [
        (gene_position, supplied_by_base[base_id])
        for gene_position, base_id in enumerate(required_base_ids)
        if base_id in supplied_by_base and base_id not in duplicate_base_ids
    ]
    matched_count = len(matches)
    observed_count = np.zeros(len(input_matrix), dtype=int)
    expressed_count = np.zeros(len(input_matrix), dtype=int)
    if matches:
        model_positions = np.array([gene_position for gene_position, _ in matches])
        block = raw_values[:, [column_position for _, column_position in matches]]
        block[np.isinf(block)] = np.nan  # Infinite values count as unavailable, like missing ones.
        observed = ~np.isnan(block)
        observed_count = observed.sum(axis=1)
        expressed_count = (block > 0).sum(axis=1)
        aligned_values[:, model_positions] = np.where(observed, block, medians[model_positions])
    aligned = pd.DataFrame(aligned_values, index=input_matrix.index, columns=required_genes)

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
