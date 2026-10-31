from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from gtex_explorer.analysis.core import APP_DIR, load_reference_tables
from gtex_explorer.analysis.gene_panels import cv_gene_panel, load_cv_gene_rankings


@pytest.fixture(scope="module")
def tables():
    return load_reference_tables()


def test_all_75_selector_panels_match_the_actual_fit_checkpoints(tables):
    checkpoint_dir = APP_DIR.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_progress_v4"
    saved = {}
    for filename in ["step_30_anova.joblib", "step_31_mutual_information.joblib", "step_32_l1.joblib"]:
        for row in joblib.load(checkpoint_dir / filename)["selected_gene_sets"]:
            saved[(row["selector_id"], row["genes"], row["fold"])] = set(row["selected_genes"])
    fits = tables["cv_folds"].loc[tables["cv_folds"]["uses_feature_selection"]]
    assert len(fits) == len(saved) == 75
    for _, fit in fits.iterrows():
        panel = cv_gene_panel(tables["cv_gene_rankings"], fit)
        assert set(panel["gene"]) == saved[(fit["selector_id"], fit["genes"], fit["fold"])]
        assert panel["selection_rank"].tolist() == list(range(1, int(fit["genes"]) + 1))
        assert panel["configuration_id"].eq(fit["configuration_id"]).all()
        assert panel["fold"].eq(fit["fold"]).all()


def test_five_baselines_have_all_candidates_and_no_selector_scores(tables):
    fits = tables["cv_folds"].loc[~tables["cv_folds"]["uses_feature_selection"]]
    assert len(fits) == 5
    for _, fit in fits.iterrows():
        panel = cv_gene_panel(tables["cv_gene_rankings"], fit)
        assert len(panel) == panel["gene"].nunique() == 5000
        assert panel[["selection_rank", "selection_score", "dominant_tissue"]].isna().all().all()
        assert not panel["uses_feature_selection"].any()


@pytest.mark.parametrize("defect", ["fingerprint", "hash", "fits", "missing_pair", "ranks", "duplicate", "candidates"])
def test_panel_loader_rejects_inconsistent_exports(tmp_path, tables, defect):
    for filename in ["cv_gene_rankings.csv.gz", "cv_gene_panel_manifest.json", "experiment_cv_folds.csv"]:
        shutil.copyfile(APP_DIR / "data" / filename, tmp_path / filename)
    path = tmp_path / "cv_gene_panel_manifest.json"
    manifest = json.loads(path.read_text())
    if defect == "fingerprint":
        manifest["experiment_fingerprint"] = "other_run"
    elif defect == "hash":
        manifest["ranking_sha256"] = "bad"
    elif defect == "fits":
        (tmp_path / "experiment_cv_folds.csv").write_text("changed")
    else:
        rankings = tables["cv_gene_rankings"].copy()
        if defect == "missing_pair":
            rankings = rankings.loc[~(rankings["selector_id"].eq("anova") & rankings["fold"].eq(1))]
        elif defect == "ranks":
            rankings.loc[0, "selection_rank"] = 2
        elif defect == "duplicate":
            rankings.loc[0, "gene"] = rankings.loc[1, "gene"]
        else:
            rankings.loc[0, "gene"] = "ENSG_DIFFERENT"
        rankings.to_csv(tmp_path / "cv_gene_rankings.csv.gz", index=False)
        manifest["ranking_sha256"] = hashlib.sha256((tmp_path / "cv_gene_rankings.csv.gz").read_bytes()).hexdigest()
        manifest["ranking_rows"] = len(rankings)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_cv_gene_rankings(tmp_path, tables["run_summary"], tables["cv_folds"])


def gene_app():
    app = AppTest.from_string(
        'from gtex_explorer.analysis.core import load_reference_tables\nfrom gtex_explorer.ui.gene_page import render_gene_page\nrender_gene_page(load_reference_tables())\n', default_timeout=30,
    ).run()
    assert not app.exception
    return app


def test_final_panel_provenance_and_readable_table():
    app = gene_app()
    text = "\n".join(x.value for x in [*app.markdown, *app.caption])
    assert "Regresi logistik L1" in text and "C = 1" in text and "1.000 gen" in text
    assert "Menampilkan 100 dari 1.000" in text
    assert len(app.dataframe[0].value) == 100
    assert app.dataframe[0].value.columns.tolist() == ["selection_rank", "gene", "selection_score", "dominant_tissue"]


@pytest.mark.parametrize("method", ["anova", "mutual_information", "l1_c_1", "l1_c_0p1", "l1_c_0p01", "xgboost_all_5000"])
def test_each_cv_method_and_fold_renders_correct_saved_panel(tables, method):
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value(method).run()
    app.selectbox(key="gene_cv_fold").set_value(5).run()
    assert not app.exception
    count = app.selectbox(key="gene_cv_count").value
    fits = tables["cv_folds"]
    method_mask = fits["configuration_id"].eq(method) if method == "xgboost_all_5000" else fits["selector_id"].eq(method)
    fit = fits.loc[method_mask & fits["genes"].eq(count) & fits["fold"].eq(5)].iloc[0]
    expected = cv_gene_panel(tables["cv_gene_rankings"], fit).head(100)
    assert app.dataframe[0].value["gene"].tolist() == expected["gene"].tolist()
    validation_f1 = next(item for item in app.metric if item.label == "Macro-F1 validasi")
    assert validation_f1.value == f"{fit['macro_f1']:.2%}".replace(".", ",")
    if method.startswith("l1_"):
        if expected["selection_score"].eq(0).any():
            assert app.warning
    else:
        assert "dominant_tissue" not in app.dataframe[0].value


def test_gene_counts_really_change_the_panel_and_search_is_literal():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("anova").run()
    for count in [500, 1000, 2000]:
        app.selectbox(key="gene_cv_count").set_value(count).run()
        app.number_input[0].set_value(count).run()
        assert not app.exception
        assert len(app.dataframe[0].value) == count
    app.text_input(key="gene_panel_search").set_value("[").run()
    assert not app.exception
    assert app.dataframe[0].value.empty
    app.text_input(key="gene_panel_search").set_value("").run()
    assert len(app.dataframe[0].value) == 2000


def test_l1_zero_score_genes_have_no_invented_dominant_class():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("l1_c_0p01").run()
    app.selectbox(key="gene_panel_dominant").set_value("Tanpa koefisien nonzero").run()
    assert not app.exception
    displayed = app.dataframe[0].value
    assert not displayed.empty
    assert displayed["dominant_tissue"].isna().all()
    assert displayed["selection_score"].eq(0).all()


def test_missing_cv_export_does_not_invent_results():
    app = AppTest.from_string(
        "import pandas as pd\nfrom gtex_explorer.analysis.core import load_reference_tables\nfrom gtex_explorer.ui.gene_page import render_gene_page\ntables = load_reference_tables()\ntables['cv_gene_rankings'] = pd.DataFrame()\nrender_gene_page(tables)\n", default_timeout=30,
    ).run()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    assert not app.exception and app.warning
    assert len(app.dataframe) == 0


def test_download_matches_the_filtered_cv_table_and_records_provenance():
    app = gene_app()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    app.selectbox(key="gene_cv_method").set_value("mutual_information").run()
    app.selectbox(key="gene_cv_count").set_value(500).run()
    app.selectbox(key="gene_cv_fold").set_value(3).run()
    query = app.dataframe[0].value.iloc[0]["gene"]
    with patch("streamlit.download_button") as download:
        app.text_input(key="gene_panel_search").set_value(query).run()
        assert not app.exception
        arguments = next(call.kwargs for call in download.call_args_list if call.args[0] == "Unduh CSV")
    exported = pd.read_csv(io.BytesIO(arguments["data"]))
    assert exported["gene"].tolist() == app.dataframe[0].value["gene"].tolist() == [query]
    assert exported["panel_source"].eq("cv_training").all()
    assert exported["configuration_id"].eq("mutual_information__genes_500").all()
    assert exported["fold"].eq(3).all() and exported["genes"].eq(500).all()
    assert arguments["file_name"] == "gene_panel_mutual_information__genes_500_fold_3.csv"
