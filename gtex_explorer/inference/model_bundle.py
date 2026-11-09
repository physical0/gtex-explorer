"""Load the final model from the thesis experiment."""
from __future__ import annotations

import joblib
from xgboost import XGBClassifier

from gtex_explorer.platform.paths import APP_ROOT as APP_DIR

MODEL_DIR = APP_DIR / "model_bundle"


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
