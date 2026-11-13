"""Page content, provenance, and navigation checks. Not visual acceptance tests."""
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

APP_DIR = Path(__file__).resolve().parents[1]


def comparison_app(changes=""):
    """Ringkasan, whose Cross-validation tab holds the method comparison."""
    source = (
        "from gtex_explorer.experiments.reference_tables import load_reference_tables\n"
        "from gtex_explorer.ui.pages.overview import render_overview_page\n"
        "tables = load_reference_tables()\n" + changes + "\nrender_overview_page(tables)\n"
    )
    app = AppTest.from_string(source, default_timeout=60).run()
    assert not app.exception
    return app


def tissue_chart(app):
    """The per-tissue F1 chart of the holdout tab."""
    return next(chart for chart in app.get("vega_lite_chart") if "display_support" in chart.proto.spec)


def test_comparison_keeps_notebook_choice_even_when_other_scores_change():
    app = comparison_app(
        "tables['cv_summary'].loc[tables['cv_summary']['selector_id'].eq('anova'), 'mean_macro_f1'] = 1.0\n"
    )
    text = "\n".join(item.value for item in app.markdown)
    assert "<dd>Regresi logistik L1 - C = 1</dd>" in text
    summary = app.dataframe[0].value
    chosen = summary.loc[summary.selection_status.eq("Terpilih"), "configuration_label"].tolist()
    assert chosen == ["Regresi logistik L1 - C = 1"]
    assert not app.success


def test_comparison_metric_changes_table_and_chart_and_zoom_is_explicit():
    app = comparison_app()
    assert "mean_macro_f1" in app.dataframe[0].value
    app.selectbox(key="comparison_metric").set_value("Macro-precision").run()
    assert not app.exception
    assert "mean_macro_precision" in app.dataframe[0].value
    assert "std_macro_precision" in app.dataframe[0].value
    assert "mean_macro_f1" not in app.dataframe[0].value
    spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    line = spec["layer"][1]
    assert line["encoding"]["y"]["field"] == "mean_macro_precision"
    assert line["encoding"]["y"]["scale"]["domain"] == [0, 1]
    app.checkbox(key="comparison_zoom").check().run()
    zoomed = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    assert "rentang diperbesar" in zoomed["layer"][1]["encoding"]["y"]["title"]
    assert zoomed["layer"][1]["encoding"]["y"]["scale"]["zero"] is False


def test_overview_chart_scores_and_denominators_equal_saved_tissue_results():
    app = AppTest.from_string(
        'from gtex_explorer.experiments.reference_tables import load_reference_tables\nfrom gtex_explorer.ui.pages.overview import render_overview_page\nrender_overview_page(load_reference_tables())\n', default_timeout=60,
    ).run()
    assert not app.exception
    dataset = tissue_chart(app).proto.datasets[0]
    shown = pa.ipc.open_stream(dataset.data.data).read_all().to_pandas().sort_values("tissue").reset_index(drop=True)
    saved = pd.read_csv(APP_DIR / "data/test_per_tissue_xgboost.csv").sort_values("tissue").reset_index(drop=True)
    pd.testing.assert_series_equal(shown["f1-score"], saved["f1-score"], check_names=False)
    assert shown["support"].sum() == 193
    assert shown["display_support"].tolist() == [f"n = {int(value)}" for value in saved["support"]]


@pytest.mark.parametrize("key,title", [
    ("overview_open_genes", "Panel gen eksperimen"),
    ("overview_open_prediction", "Prediksi jaringan"),
])
def test_overview_actions_open_existing_pages(key, title):
    app = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60).run()
    app = enter_application(app)
    app.button(key="nav_Ringkasan").click().run()
    app.button(key=key).click().run()
    assert not app.exception
    assert title in [element.value for element in app.title]


def test_prediction_header_follows_bundle_and_has_no_duplicate_footer():
    app = AppTest.from_string(
        "from gtex_explorer.experiments.reference_tables import load_reference_tables\nfrom gtex_explorer.inference.model_bundle import load_model_bundle\nfrom gtex_explorer.ui.pages.prediction import render_prediction_page\ntables=load_reference_tables()\np,c=load_model_bundle()\np.update(selector_id='anova', feature_selection='anova', feature_count=500)\nfrom unittest.mock import patch\nwith patch('gtex_explorer.ui.pages.prediction._choose_model', return_value=(p,c)):\n    render_prediction_page()\n", default_timeout=60,
    ).run()
    assert not app.exception
    text = "\n".join(element.value for element in app.markdown)
    assert "XGBoost - ANOVA - 500 gen" in text
    assert "L1 C=1" not in text
    assert "offline-strip" not in text
    assert app.title[0].value == "Prediksi jaringan"


def test_method_chart_draws_sd_as_bars_within_the_score_range_and_whole_percent_ticks():
    app = comparison_app()
    app.checkbox(key="comparison_zoom").check().run()
    chart = app.get("vega_lite_chart")[0]
    spec = json.loads(chart.proto.spec)
    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert marks == ["rule", "line", "point", "rule", "text"]
    assert spec["layer"][1]["encoding"]["y"]["axis"]["tickMinStep"] == 0.01
    assert spec["config"]["background"] == "transparent"
    data = pa.ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()
    assert data["upper"].le(1).all() and data["lower"].ge(0).all()
    # Selectors at one gene count sit side by side, not on top of each other.
    assert data.groupby("genes")["x"].nunique().eq(5).all()
