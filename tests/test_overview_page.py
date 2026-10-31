"""Check that the redesign renders exports, not numbers copied from the mockup."""
from pathlib import Path
import json
import sys

import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]


def overview_app(changes: str = "") -> AppTest:
    source = (
        f"import sys\nsys.path.insert(0, {str(APP_DIR)!r})\n"
        "from gtex_explorer.analysis.core import load_reference_tables\n"
        "from gtex_explorer.ui.overview_page import render_overview_page\n"
        "tables = load_reference_tables()\n"
        f"{changes}\n"
        "render_overview_page(tables)\n"
    )
    app = AppTest.from_string(source, default_timeout=60).run()
    assert not app.exception
    return app


def visible_html(app: AppTest) -> str:
    return "\n".join(item.value for item in [*app.markdown, *app.caption] if "<style>" not in item.value)


def test_overview_shows_saved_results_and_readable_labels():
    app = overview_app()
    text = visible_html(app)
    assert "L1 - C = 1" in text
    assert "1.000" in text and "494" in text
    assert "15 konfigurasi seleksi fitur + 1 baseline" in text
    assert "16 konfigurasi × 5 fold = 80 hasil evaluasi" in text
    assert "807 sampel - 494 donor" in text
    assert "193 sampel - 124 donor" in text
    assert "Donor unik" in text and ">618</div>" in text
    for value in ["96,25%", "95,49%", "96,37%", "97,55%"]:
        assert value in text
    assert "Part A" not in text and "l1_c_1" not in text
    assert not app.dataframe


def test_overview_keeps_scope_and_source_details_in_their_respective_sections():
    app = overview_app()
    text = visible_html(app)
    assert "Holdout dipisahkan sebelum validasi silang" in text
    assert "tidak digunakan untuk memilih konfigurasi" in text
    assert "dilatih ulang pada seluruh data development" in text
    assert "**Macro-F1:** 99,27% ± 0,49 poin persentase" in text
    assert "bukan interval kepercayaan atau skor holdout" in text
    sources = next(item for item in app.expander if item.label == "Asal angka dan rincian eksperimen")
    assert not sources.proto.expanded
    for filename in ["sample_split.csv", "run_summary.json", "label_mapping.csv",
                     "experiment_cv_summary.csv", "experiment_cv_folds.csv",
                     "final_test_summary.csv", "test_per_tissue_xgboost.csv"]:
        assert filename in text
    assert "bukan satu sampel atau satu jaringan" in text
    assert "500, 1.000, 2.000 gen" in text


def test_cohort_numbers_reconcile_to_original_split_and_saved_cv_results():
    source = APP_DIR.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_only"
    if not (source / "sample_split.csv").is_file():
        pytest.skip("Original notebook export is not distributed with the portable app.")
    split = pd.read_csv(source / "sample_split.csv")
    summary = json.loads((APP_DIR / "data/run_summary.json").read_text())
    original = json.loads((source / "run_summary.json").read_text())
    assert original["experiment_fingerprint"] == summary["experiment_fingerprint"]
    assert len(split) == split.SAMPID.nunique() == summary["samples_loaded"]
    groups = {}
    for split_name, prefix in [("development", "development"), ("test", "test")]:
        rows = split.loc[split.split.eq(split_name)]
        assert len(rows) == summary[f"{prefix}_samples"]
        assert rows.SUBJID.nunique() == summary[f"{prefix}_donors"]
        groups[split_name] = set(rows.SUBJID)
    assert not groups["development"] & groups["test"]
    assert split.SUBJID.nunique() == summary["development_donors"] + summary["test_donors"]
    labels = pd.read_csv(APP_DIR / "data/label_mapping.csv")
    assert set(split.SMTSD) == set(labels.tissue)
    folds = pd.read_csv(APP_DIR / "data/experiment_cv_folds.csv")
    configurations = pd.read_csv(APP_DIR / "data/experiment_cv_summary.csv")
    assert not folds.duplicated(["configuration_id", "fold"]).any()
    assert folds.groupby("configuration_id").size().eq(summary["cv_folds"]).all()
    assert set(folds.configuration_id) == set(configurations.configuration_id)
    assert len(folds) == len(configurations) * summary["cv_folds"]
    holdout = pd.read_csv(APP_DIR / "data/final_holdout_predictions.csv")
    assert set(holdout.SAMPID) == set(split.loc[split.split.eq("test"), "SAMPID"])


def test_split_and_cv_labels_use_current_result_values():
    text = visible_html(overview_app(
        "tables['run_summary']['development_samples'] = 777\n"
        "tables['run_summary']['test_samples'] = 223\n"
        "tables['run_summary']['test_donors'] = 90\n"
        "tables['run_summary']['cv_folds'] = 4\n"
        "tables['cv_folds'] = tables['cv_folds'].iloc[:64]\n"
    ))
    assert "777 sampel - 494 donor" in text
    assert "223 sampel - 90 donor" in text
    assert ">584</div>" in text
    assert "16 konfigurasi × 4 fold = 64 hasil evaluasi" in text


@pytest.mark.parametrize("changes, expected", [
    ("tables['run_summary']['samples_loaded'] = 2345", "2.345"),
    ("tables['run_summary']['development_donors'] = 87", "807 sampel - 87 donor"),
    ("tables['final_test'].loc[0, 'macro_f1'] = 0.8123", "81,23%"),
    ("tables['labels'] = tables['labels'].iloc[:3]", ">3</div>"),
    ("tables['run_summary']['xgboost_parameters']['n_estimators'] = 42", "42 putaran boosting"),
])
def test_overview_changes_when_export_values_change(changes, expected):
    assert expected in visible_html(overview_app(changes))


def test_selector_label_is_not_fixed_to_l1():
    app = overview_app("tables['final_test'].loc[0, 'selector_id'] = 'anova'")
    text = visible_html(app)
    assert "ANOVA" in text
    assert "L1 - C = 1" not in text


def test_chart_keeps_full_names_zero_based_axis_and_saved_f1():
    app = overview_app()
    chart = app.get("vega_lite_chart")[0]
    spec = json.loads(chart.proto.spec)
    encoding = {**spec.get("encoding", {}), **spec["layer"][0].get("encoding", {})}
    assert encoding["x"]["field"] == "f1-score"
    assert encoding["x"]["scale"]["domain"] == [0, 1]
    assert encoding["y"]["axis"]["labelLimit"] == 0
    assert len(encoding["y"]["sort"]) == 13
    assert "Adipose - Visceral (Omentum)" in encoding["y"]["sort"]
    assert "Heart - Left Ventricle" in encoding["y"]["sort"]
    assert spec["height"] == 13 * 30
    assert encoding["y"]["sort"][0] == "Artery - Coronary"
    assert any(layer.get("encoding", {}).get("text", {}).get("field") == "display_support" for layer in spec["layer"])
    assert any(layer["mark"]["type"] == "text" for layer in spec["layer"])

    changed = overview_app(
        "tables['per_tissue'].loc[tables['per_tissue']['tissue'].eq('Thyroid'), 'f1-score'] = 0.5"
    )
    changed_spec = json.loads(changed.get("vega_lite_chart")[0].proto.spec)
    changed_encoding = {**changed_spec.get("encoding", {}), **changed_spec["layer"][0].get("encoding", {})}
    assert changed_encoding["y"]["sort"][0] == "Thyroid"
    changed.selectbox(key="overview_tissue_order").set_value("F1-score tertinggi").run()
    reordered = json.loads(changed.get("vega_lite_chart")[0].proto.spec)
    reordered_encoding = {**reordered.get("encoding", {}), **reordered["layer"][0].get("encoding", {})}
    assert reordered_encoding["y"]["sort"][-1] == "Thyroid"
    changed.selectbox(key="overview_tissue_order").set_value("Nama jaringan").run()
    alphabetical = json.loads(changed.get("vega_lite_chart")[0].proto.spec)
    alphabetical_encoding = {**alphabetical.get("encoding", {}), **alphabetical["layer"][0].get("encoding", {})}
    assert alphabetical_encoding["y"]["sort"] == sorted(encoding["y"]["sort"])
