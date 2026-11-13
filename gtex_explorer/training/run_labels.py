"""Names, dates, and settings labels that describe a saved training run.

The labels match the fields of the Latih Model form, so a run reads the same on every page.
"""

import json
import time
from pathlib import Path

from gtex_explorer.display import METHODS, RUN_STATUS, count, decimal, method_name

NOTEBOOK_DATA = "Data eksperimen notebook"
XGBOOST_LABELS = {
    "n_estimators": "Putaran boosting",
    "learning_rate": "Learning rate",
    "max_depth": "Max depth",
    "min_child_weight": "Min child weight",
    "subsample": "Subsample",
    "colsample_bytree": "Colsample (bytree)",
    "reg_lambda": "L2 XGBoost (lambda)",
    "reg_alpha": "L1 XGBoost (alpha)",
    "gamma": "Gamma",
    "n_jobs": "Thread CPU",
}
L1_OPTION_LABELS = {"l1_max_iter": "Maksimum iterasi", "l1_tol": "Toleransi", "l1_class_weight": "Bobot kelas"}


# The name a user gives a run. It lives beside config.json, which the gallery verifies by hash, so a run can
# be renamed after it is saved without changing what was verified.
RUN_NAME_FILE = "name.json"
RUN_NAME_LIMIT = 80


def run_name(path):
    """The name the user gave a run, or None."""
    try:
        value = json.loads((Path(path) / RUN_NAME_FILE).read_text(encoding="utf-8")).get("name")
    except (OSError, ValueError, AttributeError):
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def run_created(path):
    """Start time saved with the run; older runs fall back to the file time."""
    config = Path(path) / "config.json"
    try:
        created = json.loads(config.read_text(encoding="utf-8")).get("created")
        if isinstance(created, (int, float)):
            return float(created)
    except (OSError, ValueError):
        pass
    try:
        return config.stat().st_mtime
    except OSError:
        return 0.0


def default_preprocessing(config):
    """Whether a run or model used log2(TPM+1) and standardization, as every one did before this was a choice."""
    return config.get("log2_transform", True) and config.get("standardize", True)


def preprocessing_label(config):
    """Describe the feature transforms of a run's config or a model's preprocessor."""
    return {
        (True, True): "log2(TPM+1) + standardisasi",
        (True, False): "log2(TPM+1) tanpa standardisasi",
        (False, True): "TPM + standardisasi",
        (False, False): "TPM tanpa transformasi",
    }[(config.get("log2_transform", True), config.get("standardize", True))]


def method_description(config, *, with_preprocessing=True):
    """Describe the selection method of a run, naming its transforms only when they are not the default."""
    method = method_name(config["selector"], l1_c=config.get("l1_c"), n_neighbors=config.get("mi_neighbors"))
    text = f"{method} - {count(config['top_k'])} gen"
    if not with_preprocessing or default_preprocessing(config):
        return text
    return f"{text} - {preprocessing_label(config)}"


def evaluation_label(config):
    """Describe how a model was scored, in the words of the training form."""
    parts = [f"Holdout {config['holdout_percent']}%"] if config.get("holdout_percent") else []
    if config["evaluate_cv"]:
        parts.append(f"CV {config['cv_folds']} fold")
    return " dan ".join(parts) or "Tanpa holdout dan CV"


def _value(value):
    """Format a setting as the result pages write numbers; the form may still have left it empty."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return decimal(value)
    return count(value) if isinstance(value, int) and not isinstance(value, bool) else str(value)


def data_settings(config, candidate_count=None):
    """Pembagian data utama and Pra-pemrosesan as (label, value) rows."""
    holdout = config.get("holdout_percent")
    rows = [
        ("Data development", f"{100 - (holdout or 0)}%"),
        ("Data holdout", f"{holdout}%" if holdout else "Tidak ada"),
    ]
    if config["evaluate_cv"] or holdout:
        rows.append(("Pembagian", "Donor terpisah"))
    if candidate_count is not None:
        rows.append(("Gen kandidat", count(candidate_count)))
    rows.append(("Pra-pemrosesan", preprocessing_label(config)))
    return rows


def model_settings(config):
    """Seleksi fitur, Evaluasi tambahan, and the boosting rounds as (label, value) rows."""
    method = config["selector"]
    rows = [
        ("Metode seleksi fitur", METHODS.get(method, method)),
        ("Top-k gen", _value(config.get("top_k"))),
    ]
    if method == "l1":
        rows.append(("C (regularisasi)", _value(config.get("l1_c"))))
    elif method == "mutual_information":
        rows.append(("n_neighbors", _value(config.get("mi_neighbors"))))
    rows += [
        ("Evaluasi tambahan", f"Cross-validation {config['cv_folds']} fold" if config["evaluate_cv"] else "Tidak ada"),
        ("Classifier", "XGBoost"),
        (XGBOOST_LABELS["n_estimators"], _value(config["xgboost"].get("n_estimators"))),
    ]
    return rows


def parameter_settings(config):
    """Every XGBoost parameter, and the L1 solver options, as (label, value) rows."""
    rows = [(XGBOOST_LABELS[name], _value(value)) for name, value in config["xgboost"].items() if name in XGBOOST_LABELS]
    if config["selector"] == "l1":
        rows += [
            (L1_OPTION_LABELS["l1_max_iter"], _value(config.get("l1_max_iter"))),
            (L1_OPTION_LABELS["l1_tol"], _value(config.get("l1_tol"))),
            (L1_OPTION_LABELS["l1_class_weight"], config.get("l1_class_weight") or "None"),
        ]
    return rows


def run_label(path, status=None):
    """Build the label of a run in selection lists, led by its status when one is given."""
    try:
        saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
        dataset = saved.get("dataset", {})
        filename = dataset.get("source", {}).get("expression_file", NOTEBOOK_DATA)
        stamp = time.strftime("%d/%m %H:%M", time.localtime(run_created(path)))
        label = f"{stamp} - {method_description(saved['config'])} - {filename} - {path.name[-6:]}"
    except (OSError, ValueError, KeyError):
        label = f"Riwayat tidak dapat dibaca - {path.name[-6:]}"
    name = run_name(path)
    if name:
        label = f"{name} · {label}"
    if status in RUN_STATUS:
        symbol, name = RUN_STATUS[status]
        label = f"{symbol} {name} · {label}"
    return label
