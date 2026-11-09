"""Load the notebook result tables shown on the experiment pages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gtex_explorer.experiments.gene_panels import load_cv_gene_rankings
from gtex_explorer.experiments.selector_diagnostics import load_selector_diagnostics
from gtex_explorer.platform.paths import APP_ROOT as APP_DIR

DATA_DIR = APP_DIR / "data"


def load_json(path: Path) -> dict:
    """Read one JSON file."""
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
        "selector_cv_summary": pd.read_csv(DATA_DIR / "feature_selection_cv_summary.csv"),
        "selector_cv_folds": pd.read_csv(DATA_DIR / "feature_selection_cv_folds.csv"),
        "baseline_cv_folds": pd.read_csv(DATA_DIR / "xgboost_all_5000_cv_folds.csv"),
        "experiment_validation": load_json(DATA_DIR / "experiment_validation_report.json"),
        "selected_genes": pd.read_csv(DATA_DIR / "selected_genes.csv"),
        "final_test": pd.read_csv(DATA_DIR / "final_test_summary.csv"),
        "final_holdout": load_optional_csv("final_holdout_comparison.csv"),
        "final_holdout_predictions": load_optional_csv("final_holdout_predictions.csv"),
        "final_holdout_per_tissue": load_optional_csv("final_holdout_per_tissue_comparison.csv"),
        "per_tissue": pd.read_csv(DATA_DIR / "test_per_tissue_xgboost.csv"),
        "labels": pd.read_csv(DATA_DIR / "label_mapping.csv"),
        "config_status": pd.read_csv(DATA_DIR / "feature_selection_configuration_status.csv"),
        "stability": pd.read_csv(DATA_DIR / "feature_selection_stability_summary.csv"),
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
