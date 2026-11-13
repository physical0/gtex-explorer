from __future__ import annotations

import io
from unittest.mock import patch

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.platform.paths import APP_ROOT as APP_DIR
from gtex_explorer.experiments.reference_tables import load_reference_tables
from gtex_explorer.experiments.gene_panels import L1_INSUFFICIENT_GENES, clarify_eligibility_reasons
from gtex_explorer.ui.pages.glossary import SOURCES, find_terms, glossary_table


def test_biology_dictionary_covers_all_target_tissues_with_indonesian_names():
    terms = glossary_table()
    assert terms["term"].is_unique
    assert set(terms["category"]) == {
        "Biologi molekuler", "Jaringan dan anatomi", "Pengukuran ekspresi",
    }
    assert not terms[["term", "definition", "english", "context"]].isna().any().any()
    assert set(terms["source"]) <= set(SOURCES)
    for tissue in load_reference_tables()["labels"]["tissue"]:
        matches = find_terms(tissue)
        assert not matches.empty, tissue
        assert matches.iloc[0]["english"] == tissue
        assert matches.iloc[0]["category"] == "Jaringan dan anatomi"


@pytest.mark.parametrize(("query", "expected"), [
    ("  GENE EXPRESSION  ", "Ekspresi gen"),
    ("tiroid", "Tiroid"),
    ("Thyroid", "Tiroid"),
    ("Heart - Atrial Appendage", "Apendiks atrium"),
    ("messenger RNA", "mRNA"),
    ("Transcripts Per Million", "TPM"),
])
def test_search_resolves_biological_names_and_aliases(query, expected):
    assert find_terms(query).iloc[0]["term"] == expected


@pytest.mark.parametrize("query", [
    "ANOVA", "XGBoost", "L1", "Macro-F1", "Jaccard", "SHA256", "coefficient",
    "Data development", "Data training", "Data validation", "holdout",
])
def test_dictionary_excludes_statistics_and_software_entries(query):
    assert find_terms(query).empty


@pytest.mark.parametrize("query", ["[", ".*", "not_a_term_987"])
def test_search_treats_punctuation_as_literal_text(query):
    assert find_terms(query).empty


def test_related_terms_have_real_dictionary_targets():
    terms = glossary_table()
    for related in terms["related"]:
        assert set(related) <= set(terms["term"])


def test_l1_message_changes_only_where_diagnostics_establish_insufficient_genes():
    tables = load_reference_tables()
    original = tables["cv_summary"].copy(deep=True)
    clarified = clarify_eligibility_reasons(original, tables["cv_folds"])
    changed = clarified["eligibility_reason"].ne(original["eligibility_reason"])
    assert changed.sum() == 7
    assert clarified.loc[changed, "feature_selection"].eq("l1").all()
    assert clarified.loc[changed, "eligibility_reason"].eq(L1_INSUFFICIENT_GENES).all()
    pd.testing.assert_frame_equal(clarified.drop(columns="eligibility_reason"), original.drop(columns="eligibility_reason"))
    pd.testing.assert_frame_equal(original, tables["cv_summary"])
    no_evidence = tables["cv_folds"].copy()
    no_evidence["nonzero_gene_count"] = 5000
    pd.testing.assert_frame_equal(clarify_eligibility_reasons(original, no_evidence), original)


def glossary_app():
    app = AppTest.from_string('from gtex_explorer.ui.pages.glossary import render_glossary_page\nrender_glossary_page()', default_timeout=30).run()
    assert not app.exception
    return app


def test_dictionary_opens_one_definition_and_changes_it_from_the_index():
    app = glossary_app()
    assert app.title[0].value == "Kamus biologi"
    assert app.session_state["glossary_selected_term"] == "Ekspresi gen"
    assert "Ekspresi gen" in [item.value for item in app.header]
    app.button(key="glossary_term_Protein").click().run()
    assert not app.exception
    assert app.session_state["glossary_selected_term"] == "Protein"
    assert "Protein" in [item.value for item in app.header]
    assert "Ekspresi gen" not in [item.value for item in app.header]


def test_dictionary_filters_reconcile_selection_and_recover_from_no_results():
    app = glossary_app()
    app.selectbox(key="glossary_category").set_value("Jaringan dan anatomi").run()
    assert not app.exception
    assert app.session_state["glossary_selected_term"] == "Adiposa subkutan"
    app.text_input(key="glossary_search").set_value("Thyroid").run()
    assert not app.exception
    assert app.session_state["glossary_selected_term"] == "Tiroid"
    assert "Tiroid" in [item.value for item in app.header]
    app.text_input(key="glossary_search").set_value("not_a_term_987").run()
    assert app.info and not app.header
    app.text_input(key="glossary_search").set_value("").run()
    assert not app.exception and app.header
    assert "Tiroid" in [item.value for item in app.header]


def test_related_term_navigation_opens_target_even_outside_active_filters():
    app = glossary_app()
    app.text_input(key="glossary_search").set_value("gene expression").run()
    app.button(key="glossary_related_TPM").click().run()
    assert not app.exception
    assert app.session_state["glossary_selected_term"] == "TPM"
    assert app.text_input(key="glossary_search").value == ""
    assert app.selectbox(key="glossary_category").value == "Semua kategori"
    assert "TPM" in [item.value for item in app.header]


def test_dictionary_download_contains_entire_filtered_index_and_source_addresses():
    app = glossary_app()
    with patch("streamlit.download_button") as download:
        app.selectbox(key="glossary_category").set_value("Jaringan dan anatomi").run()
        csv = pd.read_csv(io.BytesIO(download.call_args.kwargs["data"]))
    assert len(csv) == 14  # Thirteen target tissues and the general tissue concept.
    assert {"Istilah", "Nama Inggris atau GTEx", "Definisi", "Rujukan", "Alamat rujukan"} <= set(csv)
    assert csv["Kategori"].eq("Jaringan dan anatomi").all()
    assert csv["Istilah"].is_unique
    assert csv["Alamat rujukan"].str.startswith("https://").all()


def test_comparison_retains_l1_reasons_and_navigates_to_biology_dictionary():
    app = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60).run()
    app = enter_application(app)
    app.button(key="nav_Ringkasan").click().run()
    assert not app.exception
    table = next(item.value for item in app.dataframe if "eligibility_reason" in item.value.columns)
    assert table["eligibility_reason"].eq(L1_INSUFFICIENT_GENES).sum() == 7
    app.button(key="nav_Kamus Biologi").click().run()
    assert not app.exception and app.title[0].value == "Kamus biologi"
    app.text_input(key="glossary_search").set_value("coefficient").run()
    assert app.info and not app.header
