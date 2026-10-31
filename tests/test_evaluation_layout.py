"""Catch mixed CV/holdout results and summaries detached from the saved exports."""
import json

import pyarrow as pa

from gtex_explorer.ui.glossary import find_terms
from test_overview_page import overview_app, visible_html
from test_page_design import comparison_app


def tab_text(tab):
    return "\n".join(item.value for item in [*tab.markdown, *tab.caption])


def test_overview_keeps_cv_and_holdout_scores_in_different_tabs():
    app = overview_app()
    assert [tab.label for tab in app.tabs] == ["CV Folding", "Hasil akhir - holdout"]
    assert app.get("tab_container")[0].proto.tab_container.default_tab_index == 1
    cv, holdout = app.tabs
    assert "99,27%" in tab_text(cv)
    assert "96,25%" not in tab_text(cv)
    assert "96,25%" in tab_text(holdout)
    assert "99,27%" not in tab_text(holdout)
    assert "807 sampel - 494 donor" in tab_text(cv)
    assert "193 sampel - 124 donor" in tab_text(holdout)
    assert not cv.get("vega_lite_chart")
    assert len(holdout.get("vega_lite_chart")) == 1


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


def test_comparison_separates_two_stages_and_keeps_all_cv_configurations():
    app = comparison_app()
    headings = [item.value for item in app.subheader]
    assert "1. Hasil CV Folding" in headings
    assert "2. Hasil akhir - holdout" in headings
    assert headings.index("1. Hasil CV Folding") < headings.index("2. Hasil akhir - holdout")
    cv_details = next(item for item in app.expander if item.label == "Lihat 16 konfigurasi")
    assert not cv_details.proto.expanded
    assert len(cv_details.dataframe[0].value) == 16
    assert cv_details.dataframe[0].value.eligibility_reason.str.contains("nonzero").sum() == 7
    assert len(app.dataframe[0].value) == 4
    assert app.dataframe[0].value.genes.tolist() == [1000, 1000, 1000, 5000]
    text = "\n".join(item.value for item in [*app.markdown, *app.caption])
    assert "807 sampel - 494 donor" in text
    assert "193 sampel - 124 donor" in text
    assert "Data development" in text


def test_comparison_final_chart_has_direct_labels_and_only_holdout_values():
    app = comparison_app()
    chart = app.get("vega_lite_chart")[1]
    spec = json.loads(chart.proto.spec)
    assert any(layer.get("mark", {}).get("type") == "text" for layer in spec["layer"])
    data = pa.ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()
    assert len(data) == 10
    assert set(data["display_score"]) >= {"96,25%", "95,28%", "97,55%", "95,39%"}
    assert "99,27%" not in set(data["display_score"])
    assert data.score.between(0, 1).all()


def test_biology_dictionary_excludes_the_three_data_partition_entries():
    for term in ["Data development", "Data training", "Data validation"]:
        assert find_terms(term).empty
