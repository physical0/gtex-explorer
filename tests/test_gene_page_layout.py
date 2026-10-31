"""Keep table, training diagnostics, validation scores, and cross-fold stability distinct."""
import pytest

from test_gene_panels import gene_app


def metrics(block):
    return {item.label: item.value for item in block.metric}


def text(block):
    return "\n".join(item.value for item in [*block.markdown, *block.caption])


def test_cv_tabs_keep_gene_list_separate_from_diagnostics_and_jaccard():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    assert [tab.label for tab in app.tabs] == ["Daftar gen", "Diagnostik metode", "Stabilitas antar-fold"]
    genes, diagnostics, stability = app.tabs
    assert not genes.metric
    assert len(genes.dataframe[0].value) == 100
    assert metrics(diagnostics)["Gen nonzero"] == "1.582 / 5.000"
    assert metrics(diagnostics)["Macro-F1 validasi"] == "100,00%"
    assert "Rerata Jaccard" not in metrics(diagnostics)
    assert metrics(stability) == {"Rerata Jaccard": "78,88%", "Minimum Jaccard": "77,62%"}
    assert len(stability.dataframe[0].value) == 10
    assert any(button.label == "Unduh CSV" for button in genes.get("download_button"))


def test_final_source_has_no_cv_stability_tab_or_borrowed_validation_scores():
    app = gene_app()
    assert [tab.label for tab in app.tabs] == ["Daftar gen", "Diagnostik metode"]
    assert not app.tabs[0].metric
    assert metrics(app.tabs[1])["Gen nonzero"] == "1.623 / 5.000"
    assert metrics(app.tabs[1])["Iterasi solver"] == "609"
    assert "Macro-F1 validasi" not in metrics(app.tabs[1])
    assert "Rerata Jaccard" not in metrics(app)
    assert "Data development: 807 sampel - 494 donor" in text(app)


def test_fold_sample_scope_tracks_real_fold_not_the_mockup():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    assert "Data training: 640 sampel - Data validation: 167 sampel" in text(app)
    app.selectbox(key="gene_cv_fold").set_value(5).run()
    assert not app.exception
    assert "Data training: 665 sampel - Data validation: 142 sampel" in text(app)
    assert "Data training: 640 sampel" not in text(app)


def test_baseline_has_validation_scores_without_selector_or_stability_tabs():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("xgboost_all_5000").run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["Daftar gen", "Kinerja XGBoost"]
    assert app.tabs[0].dataframe[0].value.columns.tolist() == ["gene"]
    assert "Macro-F1 validasi" in metrics(app.tabs[1])
    assert "Gen nonzero" not in metrics(app)
    assert "Rerata Jaccard" not in metrics(app)


def test_gene_filter_does_not_recompute_full_panel_diagnostics_or_jaccard():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    assert len(app.tabs) == 3
    original_diagnostics = metrics(app.tabs[1])
    original_stability = metrics(app.tabs[2])
    app.text_input(key="gene_panel_search").set_value("ENSG00000170549.4").run()
    assert not app.exception
    assert app.tabs[0].dataframe[0].value.gene.tolist() == ["ENSG00000170549.4"]
    assert metrics(app.tabs[1]) == original_diagnostics
    assert metrics(app.tabs[2]) == original_stability


@pytest.mark.parametrize("count,panel_summary", [
    (500, "Panel top-500: 500 gen nonzero dan 0 gen berskor nol."),
    (2000, "Panel top-2.000: 1.582 gen nonzero dan 418 gen berskor nol."),
])
def test_l1_diagnostics_distinguish_candidate_fit_from_selected_top_k(count, panel_summary):
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_count").set_value(count).run()
    assert not app.exception
    diagnostics = app.tabs[1]
    assert "Data training fold 1 - 5.000 gen kandidat sebelum pemilihan top-k." in text(diagnostics)
    assert panel_summary in text(diagnostics)
    assert "seluruh panel" not in text(diagnostics)
    assert metrics(diagnostics)["Gen nonzero"] == "1.582 / 5.000"
    assert metrics(diagnostics)["Sparsitas gen"] == "68,36%"
    if count == 2000:
        assert "418 dari 2.000 gen" in app.warning[0].value
    app.text_input(key="gene_panel_search").set_value("does-not-match-any-gene").run()
    assert app.tabs[0].dataframe[0].value.empty
    assert panel_summary in text(app.tabs[1])


def test_final_l1_diagnostics_separate_full_fit_from_deployed_panel():
    app = gene_app()
    assert "5.000 gen kandidat sebelum pemilihan top-k." in text(app.tabs[1])
    assert "Panel top-1.000: 1.000 gen nonzero dan 0 gen berskor nol." in text(app.tabs[1])
    assert metrics(app.tabs[1])["Gen nonzero"] == "1.623 / 5.000"


@pytest.mark.parametrize("method,nonzero,zero,sparsity", [
    ("l1_c_0p1", "280", "1.720", "94,40%"),
    ("l1_c_0p01", "47", "1.953", "99,06%"),
])
def test_other_l1_settings_keep_candidate_sparsity_distinct_from_panel_padding(method, nonzero, zero, sparsity):
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value(method).run()
    app.selectbox(key="gene_cv_count").set_value(2000).run()
    assert not app.exception
    diagnostics = app.tabs[1]
    assert "5.000 gen kandidat sebelum pemilihan top-k." in text(diagnostics)
    assert f"Panel top-2.000: {nonzero} gen nonzero dan {zero} gen berskor nol." in text(diagnostics)
    assert metrics(diagnostics)["Gen nonzero"] == f"{nonzero} / 5.000"
    assert metrics(diagnostics)["Sparsitas gen"] == sparsity


@pytest.mark.parametrize("method,scope", [
    ("anova", "η² dan flag variansi: gen top-k. Spearman: seluruh 5.000 gen kandidat."),
    ("mutual_information", "Gen berskor positif dan Spearman: seluruh 5.000 gen kandidat. Cutoff: gen top-k."),
])
def test_filter_methods_explain_which_metrics_use_top_k_and_all_candidates(method, scope):
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value(method).run()
    app.selectbox(key="gene_cv_count").set_value(500).run()
    assert not app.exception
    assert "Data training fold 1 - panel top-500 dari 5.000 gen kandidat." in text(app.tabs[1])
    assert scope in text(app.tabs[1])
    if method == "mutual_information":
        assert metrics(app.tabs[1])["Gen berskor MI positif"] == "5.000 / 5.000"
        assert metrics(app.tabs[1])["Cutoff MI top-k (nat)"] == "1,018184"
    else:
        assert metrics(app.tabs[1])["Median η²"] == "91,63%"
    app.selectbox(key="gene_cv_count").set_value(2000).run()
    assert "Data training fold 1 - panel top-2.000 dari 5.000 gen kandidat." in text(app.tabs[1])
    assert scope in text(app.tabs[1])
