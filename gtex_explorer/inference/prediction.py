"""Run a saved model on an aligned expression matrix and explain the result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier


def _prediction_iteration_range(
    classifier: XGBClassifier,
    boosting_rounds: int | None,
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
    scaled_values = (log_values - np.asarray(preprocessor["scaler_mean"], dtype=float)) / np.asarray(
        preprocessor["scaler_scale"], dtype=float
    )
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
