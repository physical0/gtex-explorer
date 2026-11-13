from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from gtex_explorer.experiments.reference_tables import DATA_DIR
from gtex_explorer.inference.model_bundle import MODEL_DIR, load_model_bundle
from gtex_explorer.inference.audit import audit_and_align_input
from gtex_explorer.inference.prediction import explain_prediction, predict_tissues
from gtex_explorer.inference.parsing import parse_expression_file


def test_model_bundle_integrity_and_demo_prediction() -> None:
    manifest = json.loads(
        (MODEL_DIR / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    model_hash = hashlib.sha256(
        (MODEL_DIR / "xgboost_model.json").read_bytes()
    ).hexdigest()
    assert manifest["status"] == "validated"
    assert model_hash == manifest["xgboost_model_sha256"]

    preprocessor, classifier = load_model_bundle()
    feature_count = int(preprocessor["feature_count"])
    class_count = len(preprocessor["classes"])
    assert feature_count == 1000
    assert class_count == 13
    assert len(preprocessor["selected_genes"]) == feature_count
    assert len(preprocessor["raw_tpm_medians"]) == feature_count
    assert len(preprocessor["scaler_mean"]) == feature_count
    assert len(preprocessor["scaler_scale"]) == feature_count

    demo_path = DATA_DIR / "demo_tpm.csv"
    with demo_path.open("rb") as handle:
        matrix = parse_expression_file(handle, demo_path.name)
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    assert audit.matched_gene_count == feature_count

    predictions, probabilities = predict_tissues(
        aligned,
        preprocessor,
        classifier,
    )
    probability_values = probabilities.to_numpy(dtype=float)
    assert len(predictions) == len(matrix)
    assert probabilities.shape == (len(matrix), class_count)
    assert np.isfinite(probability_values).all()
    assert ((probability_values >= 0) & (probability_values <= 1)).all()
    assert np.allclose(probability_values.sum(axis=1), 1.0, atol=1e-6)
    assert predictions["predicted_tissue"].tolist() == (
        probabilities.idxmax(axis=1).tolist()
    )
    assert np.allclose(
        predictions["top_probability"].to_numpy(dtype=float),
        probability_values.max(axis=1),
    )

    explanation = explain_prediction(
        aligned,
        str(aligned.index[0]),
        preprocessor,
        classifier,
        limit=8,
    )
    assert len(explanation) == 8
    assert explanation["absolute_contribution"].is_monotonic_decreasing
    assert np.isfinite(explanation["contribution"].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("value", [-1., np.inf, "not_a_number"])
def test_rejected_sample_never_reaches_classifier(value) -> None:
    preprocessor, _ = load_model_bundle()
    matrix = pd.DataFrame(value, index=["invalid"], columns=preprocessor["selected_genes"])
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert not audit.ready
    classifier = MagicMock()
    with pytest.raises(ValueError, match="audit"):
        predict_tissues(aligned, preprocessor, classifier)
    classifier.predict_proba.assert_not_called()
