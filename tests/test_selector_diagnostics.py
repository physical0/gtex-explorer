"""Selector diagnostics must describe the chosen training fold, not holdout."""
import json
import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from gtex_explorer.analysis.core import APP_DIR, load_reference_tables
from test_gene_panels import gene_app


def metric_values(app):
    return {item.label: item.value for item in app.metric}


@pytest.mark.parametrize("method,expected", [
    ("anova", {"Median η²": "89,73%", "Variansi berbeda (BF)": "97,20%"}),
    ("mutual_information", {"Spearman 3 vs 5": "0,9945"}),
    ("l1_c_1", {"Gen nonzero": "1.582 / 5.000", "Sparsitas gen": "68,36%", "Iterasi solver": "618"}),
    ("l1_c_0p1", {"Gen nonzero": "280 / 5.000", "Sparsitas gen": "94,40%"}),
    ("l1_c_0p01", {"Gen nonzero": "47 / 5.000", "Sparsitas gen": "99,06%"}),
])
def test_method_specific_metrics_use_saved_fold_one(method, expected):
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value(method).run()
    assert not app.exception
    assert "Diagnostik seleksi fitur" in [item.value for item in app.subheader]
    for label, value in expected.items():
        assert metric_values(app)[label] == value
    if method != "anova":
        assert "Median η²" not in metric_values(app)


def test_anova_gene_details_follow_search_fold_and_top_k():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("anova").run()
    app.selectbox(key="gene_cv_fold").set_value(3).run()
    app.selectbox(key="gene_cv_count").set_value(500).run()
    assert metric_values(app)["Median η²"] == "91,39%"
    gene = app.dataframe[0].value.iloc[0]["gene"]
    app.text_input(key="gene_panel_search").set_value(gene).run()
    detail = next(item.value for item in app.dataframe if "eta_squared" in item.value)
    assert detail["gene"].tolist() == [gene]
    source = pd.read_csv(APP_DIR / "data/anova_gene_diagnostics_by_fold.csv.gz")
    expected = source.loc[source.fold.eq(3) & source.gene.eq(gene)].iloc[0]
    assert detail.iloc[0]["f_statistic"] == pytest.approx(expected.f_statistic)
    assert detail.iloc[0]["eta_squared"] == pytest.approx(expected.eta_squared)


def test_mi_neighbor_checks_are_distinct_from_interfold_jaccard():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("mutual_information").run()
    app.selectbox(key="gene_cv_fold").set_value(5).run()
    detail = next(item.value for item in app.dataframe if "mi_n3" in item.value)
    assert {"mi_n3", "mi_n5", "mi_n10", "abs_rank_change_3_vs_5"}.issubset(detail)
    neighbors = next(item.value for item in app.dataframe if "neighbors_a" in item.value)
    assert len(neighbors) == 3 and neighbors.fold.eq(5).all()
    assert set(zip(neighbors.neighbors_a, neighbors.neighbors_b)) == {(3, 5), (3, 10), (5, 10)}
    assert "Stabilitas antar-fold" in [item.value for item in app.subheader]


def test_jaccard_summary_uses_ten_pairs_not_diagonal_or_one_fold():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    values = metric_values(app)
    assert values["Rerata Jaccard"] == "78,88%"
    assert values["Minimum Jaccard"] == "77,62%"
    pairs = next(item.value for item in app.dataframe if "fold_a" in item.value)
    assert len(pairs) == 10
    assert pairs.fold_a.lt(pairs.fold_b).all()
    before = values["Rerata Jaccard"]
    app.selectbox(key="gene_cv_fold").set_value(5).run()
    assert metric_values(app)["Rerata Jaccard"] == before
    assert metric_values(app)["Iterasi solver"] != values["Iterasi solver"]


def test_l1_details_compare_c_values_within_the_same_training_fold():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    comparisons = [item.value for item in app.dataframe if "gene_sparsity" in item.value]
    assert len(comparisons) == 1
    frame = comparisons[0].sort_values("l1_c", ascending=False)
    assert frame.nonzero_gene_count.tolist() == [1582, 280, 47]
    assert frame.fold.eq(1).all()
    assert frame.top_k_supported.tolist() == [True, False, False]


def test_final_diagnostics_do_not_borrow_cv_metrics_and_baseline_has_no_selector():
    app = gene_app()
    values = metric_values(app)
    assert values["Gen nonzero"] == "1.623 / 5.000"
    assert values["Iterasi solver"] == "609"
    assert "Rerata Jaccard" not in values
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("xgboost_all_5000").run()
    assert not app.exception
    assert "Rerata Jaccard" not in metric_values(app)
    assert "Gen nonzero" not in metric_values(app)


def test_missing_diagnostics_preserves_panel_without_inventing_zeros():
    app = AppTest.from_string(
        "from gtex_explorer.analysis.core import load_reference_tables\nfrom gtex_explorer.ui.gene_page import render_gene_page\ntables=load_reference_tables()\ntables['selector_diagnostics']={}\nrender_gene_page(tables)\n", default_timeout=60,
    ).run()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("anova").run()
    assert not app.exception and len(app.dataframe[0].value) == 100
    assert "Median η²" not in metric_values(app)
    assert any("Diagnostik tambahan belum tersedia" in item.value for item in app.info)


def test_exported_diagnostics_reconcile_with_rankings_and_all_fit_rows():
    from gtex_explorer.analysis.selector_diagnostics import validate_diagnostic_tables
    tables = load_reference_tables()
    diagnostics = tables["selector_diagnostics"]
    validate_diagnostic_tables(diagnostics, tables["cv_folds"], tables["cv_gene_rankings"])
    assert len(diagnostics["anova_genes"]) == len(diagnostics["mi_genes"]) == 25000
    assert len(diagnostics["l1_fits"]) == 15
    assert len(diagnostics["stability_pairs"]) == 150
    broken = {key: value.copy() for key, value in diagnostics.items()}
    broken["l1_fits"].loc[0, "nonzero_gene_count"] = 5000
    with pytest.raises(ValueError, match="L1"):
        validate_diagnostic_tables(broken, tables["cv_folds"], tables["cv_gene_rankings"])


def test_all_saved_mi_and_l1_counts_cutoffs_and_support_flags_match_the_rankings():
    tables = load_reference_tables()
    rankings = tables["cv_gene_rankings"]
    fits = tables["cv_folds"]
    checked = 0
    for row in fits.loc[fits.feature_selection.isin(["l1", "mutual_information"])].itertuples():
        candidates = rankings.loc[
            rankings.selector_id.eq(row.selector_id) & rankings.fold.eq(row.fold)
        ].sort_values("selection_rank")
        panel = candidates.head(row.genes)
        tolerance = 1e-12 if row.feature_selection == "l1" else 0.0
        nonzero = int(candidates.selection_score.gt(tolerance).sum())
        assert len(candidates) == 5000
        assert len(panel) == row.genes
        assert row.zero_scores_in_top_k == int(panel.selection_score.le(tolerance).sum())
        assert row.top_k_fully_supported == bool(row.selector_converged and nonzero >= row.genes)
        if row.feature_selection == "l1":
            assert row.nonzero_gene_count == nonzero
            assert row.coefficient_cutoff == pytest.approx(panel.selection_score.iloc[-1], abs=1e-12)
            assert panel.loc[panel.selection_score.le(tolerance), "dominant_tissue"].isna().all()
        else:
            assert row.positive_score_gene_count == nonzero
            assert row.score_cutoff == pytest.approx(panel.selection_score.iloc[-1], abs=1e-12)
        checked += 1
    assert checked == 60


@pytest.mark.parametrize("table", ["anova_summary", "mi_neighbors", "l1_fits"])
def test_diagnostic_validation_rejects_missing_method_fold_checks(table):
    from gtex_explorer.analysis.selector_diagnostics import validate_diagnostic_tables
    tables = load_reference_tables()
    broken = {key: value.copy() for key, value in tables["selector_diagnostics"].items()}
    broken[table] = broken[table].iloc[1:]
    with pytest.raises(ValueError):
        validate_diagnostic_tables(broken, tables["cv_folds"], tables["cv_gene_rankings"])


@pytest.mark.parametrize("defect", ["fingerprint", "hash", "missing", "fits"])
def test_diagnostic_loader_rejects_mixed_or_incomplete_exports(tmp_path, defect):
    from gtex_explorer.analysis.selector_diagnostics import load_selector_diagnostics
    manifest_path = APP_DIR / "data/selector_diagnostics_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename in [*manifest["files"], *manifest["bindings"]]:
        shutil.copyfile(APP_DIR / "data" / filename, tmp_path / filename)
    if defect == "fingerprint":
        manifest["experiment_fingerprint"] = "other_experiment"
    elif defect == "hash":
        manifest["files"][next(iter(manifest["files"]))]["sha256"] = "bad"
    elif defect == "missing":
        (tmp_path / next(iter(manifest["files"]))).unlink()
    else:
        (tmp_path / "experiment_cv_folds.csv").write_text("changed")
    (tmp_path / manifest_path.name).write_text(json.dumps(manifest))
    summary = json.loads((APP_DIR / "data/run_summary.json").read_text())
    with pytest.raises(ValueError):
        load_selector_diagnostics(tmp_path, summary)
