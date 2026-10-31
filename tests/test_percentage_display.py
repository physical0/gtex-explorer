"""Display units change, while model inputs and saved fractions stay untouched."""
import json

import pandas as pd
import pyarrow as pa

from test_overview_page import overview_app, visible_html
from test_page_design import comparison_app
from test_gene_panels import gene_app


def styled_values(element):
    return pa.ipc.open_stream(element.proto.arrow_data.styler.display_values).read_all().to_pandas()


def test_overview_formats_percent_without_changing_chart_fractions():
    app = overview_app()
    text = visible_html(app)
    assert "96,25%" in text and "95,49%" in text and "96,37%" in text
    assert "99,27% ± 0,49 poin persentase" in text
    spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    bar = spec["layer"][0]["encoding"]["x"]
    assert bar["scale"]["domain"] == [0, 1]
    assert bar["axis"]["format"] == ".0%"
    dataset = app.get("vega_lite_chart")[0].proto.datasets[0]
    rows = pa.ipc.open_stream(dataset.data.data).read_all().to_pandas()
    coronary = rows.loc[rows.tissue.eq("Artery - Coronary")].iloc[0]
    assert coronary["display_f1"] == "82,35%"
    assert coronary["f1-score"] < 1
    assert rows["support"].sum() == 193


def test_comparison_formats_all_axes_tooltips_tables_and_point_differences():
    app = comparison_app()
    text = "\n".join(item.value for item in [*app.markdown, *app.caption])
    assert "99,27%" in text
    for chart in app.get("vega_lite_chart"):
        spec = json.loads(chart.proto.spec)
        if "layer" in spec and spec["layer"][1].get("encoding", {}).get("y", {}).get("field", "").startswith("mean_"):
            axis = spec["layer"][1]["encoding"]["y"]
            assert axis["axis"]["format"] == ".0%"
            assert spec["layer"][1]["encoding"]["tooltip"][2]["format"] == ".2%"
        else:
            encoding = {**spec.get("encoding", {}), **spec.get("layer", [{}])[0].get("encoding", {})}
            axis = encoding["y"] if encoding["y"]["field"] == "score" else encoding["x"]
            assert axis["axis"]["format"] == ".0%"
            metric_tooltip = [tip for tip in encoding["tooltip"] if tip["field"] in {"score", "f1-score"}][0]
            assert metric_tooltip["format"] == ".2%"
    raw_cv = app.dataframe[0].value
    display_cv = styled_values(app.dataframe[0])
    selected = raw_cv.loc[raw_cv.configuration_label.str.startswith("L1 (C=1)") & raw_cv.genes.eq(1000)].index[0]
    assert display_cv.loc[selected, "mean_macro_f1"] == "99,27%"
    assert display_cv.loc[selected, "std_macro_f1"] == "0,49"
    assert raw_cv.loc[selected, "mean_macro_f1"] < 1
    assert raw_cv.loc[selected, "std_macro_f1"] < 0.01
    holdout_table = next(item for item in app.dataframe if "macro_f1" in item.value.columns)
    assert all("%" in value for value in styled_values(holdout_table)["macro_f1"])
    assert "Selisih Macro-F1: +0,97 poin persentase" in text
    assert "193 sampel - 124 donor" in text


def test_gene_panel_metrics_are_percent_but_selection_scores_are_not_rescaled():
    app = gene_app()
    final_genes = app.dataframe[0].value.copy()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    assert not app.exception
    classification_metrics = [metric for metric in app.metric if metric.label.endswith("validasi")]
    assert len(classification_metrics) == 3
    assert all(metric.value.endswith("%") for metric in classification_metrics)
    assert all(float(metric.value[:-1].replace(",", ".")) > 90 for metric in classification_metrics)
    app.radio(key="gene_panel_source").set_value("Model final").run()
    pd.testing.assert_frame_equal(app.dataframe[0].value, final_genes)
