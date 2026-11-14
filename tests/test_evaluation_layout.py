"""Catch mixed CV/holdout results and summaries detached from the saved exports."""
import json

import pyarrow as pa

from gtex_explorer.ui.pages.glossary import find_terms
from test_overview_page import overview_app, visible_html
from test_page_design import comparison_app, tissue_chart


def tab_text(tab):
    return "\n".join(item.value for item in [*tab.markdown, *tab.caption])


def test_overview_keeps_cv_and_holdout_scores_in_different_tabs():
    app = overview_app()
    assert [tab.label for tab in app.tabs] == ["Cross-validation", "Hasil akhir - holdout"]
    assert app.get("tab_container")[0].proto.tab_container.default_tab_index == 1
    cv, holdout = app.tabs
    assert "99,27%" in tab_text(cv)
    assert "96,25%" not in tab_text(cv)
    assert "96,25%" in tab_text(holdout)
    assert "99,27%" not in tab_text(holdout)
    assert "807 sampel - 494 donor" in tab_text(cv)
    assert "193 sampel - 124 donor" in tab_text(holdout)
    # The CV tab charts the methods on CV scores; the holdout tab charts tissues on holdout scores.
    [method_chart] = cv.get("vega_lite_chart")
    assert all(
        layer["encoding"]["y"]["field"].startswith(("mean_", "lower"))
        for layer in json.loads(method_chart.proto.spec)["layer"]
    )
    # The holdout tab charts the final model against the XGBoost baseline, then each tissue.
    baseline, tissue = holdout.get("vega_lite_chart")
    assert "Baseline XGBoost" in baseline.proto.spec
    assert "display_support" in tissue.proto.spec


def test_overview_holdout_changes_do_not_change_cv_scores():
    app = overview_app("tables['final_test'].loc[0, 'macro_f1'] = 0.8123")
    assert "99,27%" in tab_text(app.tabs[0])
    assert "81,23%" not in tab_text(app.tabs[0])
    assert "81,23%" in tab_text(app.tabs[1])


def test_overview_removes_marked_copy_and_uses_development_term():
    text = visible_html(overview_app())
    assert "Data development" in text
    for removed in [
        "bukan seluruh dataset", "Satu donor dapat menyumbangkan lebih dari satu sampel",
        "Tidak ada donor yang muncul pada kedua kelompok", "Kinerja model final pada donor holdout",
        "Skor ditampilkan dalam persen, semakin tinggi semakin baik", "data pengembangan",
    ]:
        assert removed not in text


def test_ringkasan_holds_the_method_comparison_and_keeps_all_cv_configurations():
    app = comparison_app()
    assert [tab.label for tab in app.tabs] == ["Cross-validation", "Hasil akhir - holdout"]
    cv_tab = app.tabs[0]
    cv_details = next(item for item in cv_tab.expander if item.label == "Lihat 16 konfigurasi")
    assert not cv_details.proto.expanded
    assert len(cv_details.dataframe[0].value) == 16
    assert cv_details.dataframe[0].value.eligibility_reason.str.contains("nonzero").sum() == 7
    assert len(cv_tab.dataframe[0].value) == 4
    assert cv_tab.dataframe[0].value.genes.tolist() == [1000, 1000, 1000, 5000]
    assert not any(item.key == "overview_cv_comparison" for item in app.button)
    text = "\n".join(item.value for item in [*app.markdown, *app.caption])
    assert "807 sampel - 494 donor" in text
    assert "193 sampel - 124 donor" in text
    assert "Perbandingan metode" in text


def test_holdout_tab_compares_the_final_model_with_the_baseline():
    app = comparison_app()
    holdout_tab = app.tabs[1]
    text = "\n".join(item.value for item in holdout_tab.markdown)
    # Final model minus baseline, from the exported holdout scores.
    for delta in ["+0,97 poin vs baseline", "+0,15 poin vs baseline", "+0,52 poin vs baseline"]:
        assert delta in text
    details = next(item for item in holdout_tab.expander if item.label == "Metrik lengkap holdout")
    metrics = details.dataframe[0].value
    assert list(metrics.columns) == ["Metrik", "Model final", "Baseline", "Selisih (poin persentase)"]
    assert metrics.set_index("Metrik").loc["Macro-F1"].tolist() == ["96,25%", "95,28%", "+0,97"]
    assert "99,27%" not in metrics.to_string()
    chart = tissue_chart(app)
    spec = json.loads(chart.proto.spec)
    assert any(layer.get("mark", {}).get("type") == "tick" for layer in spec["layer"])
    data = pa.ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()
    assert data["baseline_f1"].between(0, 1).all() and len(data) == 13


def test_biology_dictionary_excludes_the_three_data_partition_entries():
    for term in ["Data development", "Data training", "Data validation"]:
        assert find_terms(term).empty


def test_cv_tab_shows_the_method_comparison_with_its_summary_before_the_chosen_configuration():
    app = overview_app()
    cv = app.tabs[0]
    blocks = [item.value for item in cv.markdown]
    position = lambda text: next(i for i, value in enumerate(blocks) if text in value)
    comparison = position("Perbandingan metode")
    summary = position("Ringkasan konfigurasi")
    chosen = position("Pemilihan konfigurasi dengan cross-validation")
    assert comparison < summary < chosen
