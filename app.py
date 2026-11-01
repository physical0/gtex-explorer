from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from gtex_explorer.cloud.media import image, download_button

from gtex_explorer.analysis.core import (
    APP_DIR,
    DATA_DIR,
    MODEL_DIR,
    audit_and_align_input,
    load_model_bundle,
    load_reference_tables,
    model_bundle_available,
    parse_expression_file,
    predict_tissues,
)
from gtex_explorer.ui.prediction_page import render_prediction_page as render_offline_prediction_page
from gtex_explorer.ui.overview_page import render_overview_page
from gtex_explorer.ui.gene_page import render_gene_page
from gtex_explorer.analysis.gene_panels import clarify_eligibility_reasons
from gtex_explorer.ui.glossary import render_glossary_page
from gtex_explorer.ui.training_page import render_training_page, render_training_results_page, TRAINING_FORM_KEYS
from gtex_explorer.ui.training_gene_page import render_training_gene_page
from gtex_explorer.desktop_runtime import state_directory
from gtex_explorer.ui.workspace_ui import (render_workspace_controls, current_workspace_root, render_workspace_launcher,
                                           restore_location, remember_location)
from gtex_explorer.workspaces import bundled_models_enabled


st.set_page_config(
    page_title="GTEx Tissue Classifier",
    page_icon=str(APP_DIR / "assets" / "brand_dna.png"),
    layout="wide",
    initial_sidebar_state="auto",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #657184;
        --line: #e2e8ec;
        --soft: #f5f8f8;
        --teal: #0f8b8d;
        --teal-dark: #087173;
        --indigo: #4844c9;
        --success: #1f9d55;
        --warning: #c97a12;
    }

    html, body, [class*="css"] {
        color: var(--ink);
    }

    #MainMenu, footer {
        display: none;
    }

    [data-testid="stHeader"] {
        height: .25rem;
        background: transparent;
    }

    [data-testid="stAppViewContainer"] {
        background: #ffffff;
    }

    [data-testid="stMain"] .block-container {
        max-width: none;
        padding: 1.15rem 1.45rem 1.4rem;
    }

    [data-testid="stSidebar"] {
        width: 270px !important;
        background: #fbfcfc;
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 270px !important;
        padding: 1.2rem 1rem;
    }

    [data-testid="stSidebar"] button {
        justify-content: flex-start;
        border: 0;
        background: transparent;
        padding: .72rem .78rem;
        color: #253044;
    }

    [data-testid="stSidebar"] button:hover {
        background: #edf5f5;
        color: var(--teal-dark);
    }

    [data-testid="stSidebar"] button[kind="primary"] {
        background: #e9f5f5 !important;
        border-left: 4px solid var(--teal) !important;
        color: var(--teal-dark) !important;
    }

    [data-testid="stSidebar"] button[kind="primary"] * {
        color: var(--teal-dark) !important;
    }

    h1, h2, h3 {
        color: var(--ink);
        letter-spacing: -0.025em;
    }

    h1 { font-size: 2rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.05rem !important; }
    h4 {
        font-size: 1rem !important;
        margin: .55rem 0 .25rem !important;
    }

    .brand-title {
        font-weight: 760;
        font-size: 1.22rem;
        line-height: 1.1;
        letter-spacing: -0.025em;
    }

    .brand-subtitle, .eyebrow, .muted {
        color: var(--muted);
    }

    .brand-subtitle {
        font-size: .78rem;
        margin-top: .2rem;
    }

    .eyebrow {
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }

    .pipeline-badge {
        display: inline-block;
        padding: .32rem .62rem;
        border-radius: .35rem;
        color: white;
        background: var(--indigo);
        font-weight: 700;
        font-size: .82rem;
    }

    .result-title {
        font-size: 1.82rem;
        line-height: 1.15;
        font-weight: 500;
        margin: .15rem 0 .35rem;
    }

    .result-title strong {
        color: var(--teal);
        font-weight: 720;
    }

    .confidence {
        color: var(--teal);
        font-size: 2.35rem;
        line-height: 1;
        font-weight: 760;
        margin: .15rem 0 .4rem;
    }

    .section-rule {
        border-top: 1px solid var(--line);
        margin: .8rem 0 1rem;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        padding: .18rem 0;
        font-size: .88rem;
    }

    .detail-row span:first-child { color: var(--muted); }

    .sample-box {
        margin-top: 1rem;
        padding-top: .9rem;
        border-top: 1px solid var(--line);
        font-size: .82rem;
    }

    .status-good {
        color: var(--success);
        font-weight: 700;
    }

    .status-warn {
        color: var(--warning);
        font-weight: 700;
    }

    .footer-note {
        border-top: 1px solid var(--line);
        color: #6c7889;
        font-size: .8rem;
        text-align: center;
        padding-top: .9rem;
        margin-top: 1rem;
    }

    [data-testid="stMetric"] {
        background: var(--soft);
        border: 1px solid #e7edef;
        padding: .8rem .95rem;
        border-radius: .55rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: .45rem;
        overflow: hidden;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: .42rem;
        min-height: 2.55rem;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--indigo);
        border-color: var(--indigo);
    }

    @media (max-width: 980px) {
        [data-testid="stSidebar"] { width: 235px !important; }
        [data-testid="stSidebar"] > div:first-child { width: 235px !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Visual language for the offline audit-first workspace.
st.markdown(
    """
    <style>
    :root {
        --ink: #14213d;
        --muted: #667085;
        --line: #dfe3e8;
        --soft: #f7f8f8;
        --teal: #078c92;
        --teal-dark: #087278;
        --success: #138a4b;
        --warning: #a8660d;
    }

    html, body, [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background: #fbfbfa;
    }

    [data-testid="stMain"] .block-container {
        max-width: none;
        padding: 1rem 1.5rem 1.1rem;
    }

    [data-testid="stSidebar"] {
        width: 290px !important;
        background: #fcfcfb;
    }

    [data-testid="stSidebar"] > div:first-child {
        width: 290px !important;
        padding: 1rem .75rem;
    }

    [data-testid="stSidebar"] button {
        border-radius: .25rem;
        min-height: 2.65rem;
    }

    [data-testid="stSidebar"] .st-key-main_workflow button {
        min-height: 2.75rem;
        border: 0;
    }

    [data-testid="stSidebar"] .st-key-main_workflow button p {
        font-weight: 600;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap: .45rem;}
    [data-testid="stSidebar"] button {padding: .55rem .75rem;}
    [data-testid="stSidebar"] button:hover {background: #f0f2f3; color: var(--ink);}
    [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
        min-height: calc(100dvh - 4rem);
    }
    [data-testid="stSidebar"] .st-key-sidebar_support {
        padding-top: 2rem;
        flex-shrink: 0;
    }
    [data-testid="stSidebar"] [data-testid="stLayoutWrapper"]:has(> .st-key-sidebar_support) {
        margin-top: auto;
    }
    [data-testid="stSidebar"] .st-key-workspace_context {
        background: #f4f7f7;
        border-radius: .4rem;
        padding: .65rem .7rem;
    }
    [data-testid="stSidebar"] .st-key-workspace_context [data-testid="stVerticalBlock"] {gap:.15rem;}
    .st-key-workspace_context [data-testid="stCaptionContainer"],
    .st-key-workspace_context [data-testid="stMarkdownContainer"] {margin:0; padding:0; line-height:1.3;}
    .st-key-workspace_context [data-testid="stCaptionContainer"] p,
    .st-key-workspace_context [data-testid="stMarkdownContainer"] p {margin:0;}
    [data-testid="stSidebar"] .st-key-workspace_context button {
        padding: .35rem .2rem;
        min-height: 2rem;
        color: var(--teal-dark);
    }
    .workspace-context-name {
        font-weight:600;
        line-height:1.3;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
        min-width:0;
    }
    [data-testid="stSidebar"] .st-key-reference_section hr {
        margin: 1rem 0;
        border-color: #cfd7df;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] button {padding-left: 1.8rem;}

    [data-testid="stSidebar"] button[kind="primary"] {
        background: #eef7f7 !important;
        border-left: 3px solid var(--teal) !important;
    }

    .stButton > button[kind="primary"],
    .stDownloadButton > button[kind="primary"] {
        background: var(--teal) !important;
        border-color: var(--teal) !important;
        color: white !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: #eef7f7 !important;
        border: 0 !important;
        border-left: 3px solid var(--teal) !important;
        color: var(--teal-dark) !important;
    }

    [data-testid="stSidebar"] .stButton > button[kind="primary"] * {
        color: var(--teal-dark) !important;
    }

    .stButton > button, .stDownloadButton > button {
        border-radius: .25rem;
        min-height: 2.35rem;
        box-shadow: none;
    }

    .brand-title {
        font-size: 1.14rem;
        font-weight: 720;
    }

    .brand-subtitle {
        font-size: .72rem;
    }

    .workspace-header {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1.5rem;
        padding: .15rem 0 1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: .95rem;
    }

    .workspace-title {
        color: var(--ink);
        font-size: 1.55rem;
        line-height: 1.15;
        font-weight: 720;
        letter-spacing: -.025em;
    }

    .workspace-subtitle {
        color: var(--muted);
        font-size: .84rem;
        margin-top: .25rem;
    }

    .model-provenance {
        color: #344054;
        font-size: .84rem;
        text-align: right;
        padding-top: .25rem;
        white-space: normal;
        overflow-wrap: anywhere;
        max-width: 100%;
    }

    .panel-heading {
        font-size: 1.18rem;
        font-weight: 700;
        color: var(--ink);
        margin: .1rem 0 .25rem;
    }

    .panel-copy {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: .85rem;
    }

    .file-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: .75rem;
        padding: .65rem .7rem;
        border: 1px solid var(--line);
        background: #fff;
        font-size: .82rem;
        margin: .5rem 0 .8rem;
    }

    .file-row strong {
        overflow-wrap: anywhere;
    }

    .audit-banner {
        border: 1px solid #b9dec8;
        border-left: 3px solid var(--success);
        background: #f3fbf6;
        padding: .75rem .85rem;
        margin-top: .65rem;
    }

    .audit-banner strong {
        color: #126c3e;
        display: block;
        margin-bottom: .1rem;
    }

    .audit-banner span {
        color: #526255;
        font-size: .78rem;
    }

    .audit-banner.invalid {
        border-color: #ead1a9;
        border-left-color: var(--warning);
        background: #fffaf1;
    }

    .audit-banner.invalid strong {
        color: #87510a;
    }

    .result-name {
        color: var(--teal-dark);
        font-size: 1.85rem;
        line-height: 1.05;
        font-weight: 720;
        margin: .1rem 0 .35rem;
    }

    .result-probability {
        color: var(--teal-dark);
        font-size: 2.15rem;
        line-height: 1;
        font-weight: 760;
        margin: .12rem 0 .8rem;
    }

    .probability-summary {
        display: grid;
        grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr) minmax(0, 1fr);
        gap: 1rem;
        border-bottom: 1px solid var(--line);
        padding: .5rem 0 .8rem;
        margin-bottom: .4rem;
    }
    .probability-summary-item { min-width: 0; }
    .probability-summary-item + .probability-summary-item {
        border-left: 1px solid var(--line);
        padding-left: 1rem;
    }
    .probability-summary-label { color: var(--muted); font-size: .8rem; margin-bottom: .4rem; }
    .probability-summary-name, .probability-summary .result-name {
        font-size: 1.05rem; line-height: 1.3; font-weight: 650;
        overflow-wrap: anywhere; min-height: 2.75rem;
    }
    .probability-summary .result-probability { font-size: 1.8rem; margin: .45rem 0 0; }
    .probability-summary-value { font-size: 1.45rem; margin-top: .5rem; font-weight: 650; font-variant-numeric: tabular-nums; }
    @media (max-width: 1050px) {
        .probability-summary { grid-template-columns: 1fr; gap: .6rem; }
        .probability-summary-item + .probability-summary-item { border-left: 0; padding-left: 0; border-top: 1px solid var(--line); padding-top: .6rem; }
        .probability-summary-name, .probability-summary .result-name { min-height: 0; }
    }

    .evidence-row {
        display: flex;
        justify-content: space-between;
        gap: .75rem;
        padding: .48rem 0;
        border-top: 1px solid var(--line);
        font-size: .82rem;
    }

    .evidence-row span { color: var(--muted); }

    .empty-result {
        min-height: 17rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: var(--muted);
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        margin-top: 1rem;
    }

    .empty-result strong {
        color: #344054;
        margin-bottom: .25rem;
    }

    .st-key-prediction_input {
        border-right: 1px solid var(--line);
        padding-right: 1.35rem;
        gap: .6rem;
    }

    .st-key-prediction_output .empty-result {
        min-height: 25rem;
        border-bottom: 0;
        margin-top: 1.4rem;
    }

    .prediction-audit-summary {
        width: 100%;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: .88rem;
    }
    .prediction-audit-summary th, .prediction-audit-summary td {
        border: 1px solid var(--line);
        padding: .55rem .65rem;
        text-align: left;
        vertical-align: middle;
        overflow-wrap: anywhere;
    }
    .prediction-audit-summary th:first-child { width: 59%; }
    .prediction-audit-summary td:first-child { width: 59%; }
    .prediction-audit-summary thead { display: none; }
    .prediction-audit-summary th { font-weight: 500; color: var(--muted); }

    [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] > style:only-child) {
        display: none;
    }

    [data-testid="stSidebarHeader"] {
        height: 1rem;
        min-height: 1rem;
    }

    .st-key-prediction_ready {
        min-height: 25rem;
        justify-content: center;
        align-items: center;
        gap: .35rem;
        text-align: center;
    }
    .st-key-prediction_ready [data-testid="stMarkdownContainer"] span {
        font-size: 3rem;
        color: #98a2b3;
    }
    .st-key-prediction_ready p { text-align: center; }

    .st-key-prediction_reminders [data-testid="stVerticalBlock"] {
        gap: .45rem;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] details {
        border: 0;
        border-radius: 0;
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {background: #f0f2f3;}

    [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
        font-weight: 650;
        color: var(--ink);
    }

    [data-testid="stSidebar"] [data-testid="stExpander"] summary > span {
        flex-direction: row-reverse;
        justify-content: space-between;
    }

    .st-key-prediction_controls .stButton > button:disabled {
        background: #e7eded !important;
        border-color: #d8e1e1 !important;
        color: #657184 !important;
        opacity: 1;
    }

    .st-key-prediction_controls .stButton > button:disabled * {
        color: #657184 !important;
    }

    .st-key-prediction_workspace [data-testid="stFileUploaderDropzone"] {
        background: #fff;
        border: 1px dashed #98a2b3;
        border-radius: .25rem;
    }

    .st-key-prediction_workspace [data-testid="stDataFrame"] {
        border-radius: 0;
        border-color: var(--line);
    }

    .st-key-prediction_workspace [data-testid="stExpander"] {
        border-radius: 0;
        border-color: var(--line);
        box-shadow: none;
    }

    .st-key-prediction_workspace [data-testid="stDivider"] {
        margin: .45rem 0 .55rem;
    }

    .st-key-prediction_workspace h3 {
        font-size: 1rem !important;
        margin: .55rem 0 .25rem !important;
    }

    .offline-strip {
        display: flex;
        flex-wrap: wrap;
        gap: .35rem 1rem;
        align-items: center;
        color: #475467;
        border-top: 1px solid var(--line);
        margin-top: 1rem;
        padding: .8rem .1rem .1rem;
        font-size: .76rem;
    }

    .offline-strip strong { color: var(--ink); }

    [data-testid="stMetric"] {
        border-radius: .25rem;
        box-shadow: none;
    }

    @media (max-width: 1050px) {
        .workspace-header { display: block; }
        .model-provenance { text-align: left; margin-top: .45rem; }
        .st-key-prediction_input {
            border-right: 0;
            padding-right: 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Memuat referensi eksperimen...")
def reference_tables(snapshot_version: tuple) -> dict:
    del snapshot_version
    return load_reference_tables()


@st.cache_resource(show_spinner="Memuat model bawaan...")
def prediction_bundle(bundle_version: tuple):
    del bundle_version
    return load_model_bundle()


def file_version(directory) -> tuple:
    """Refresh caches after replacing exported tables or model files."""
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.iterdir()) if path.is_file()
    )


def navigation() -> str:
    if "page" not in st.session_state:
        st.session_state.page = "Prediksi Jaringan"

    with st.sidebar:
        brand_columns = st.columns([1, 3.5], vertical_alignment="center")
        with brand_columns[0]:
            image(APP_DIR / "assets" / "brand_dna.png", width=44)
        with brand_columns[1]:
            st.markdown(
                '<div class="brand-title">GTEx Tissue Classifier</div>'
                '<div class="brand-subtitle">Analisis ekspresi gen lokal</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        def destination(route, label, icon):
            if st.button(
                label,
                icon=icon,
                key=f"nav_{route}",
                width="stretch",
                type="primary" if st.session_state.page == route else "secondary",
            ):
                st.session_state.page = route
                st.rerun()

        with st.container(key="main_workflow"):
            destination("Pelatihan Model", "Latih Model", ":material/settings:")
            destination("Prediksi Jaringan", "Prediksi Jaringan", ":material/biotech:")
        notebook_routes = {"Ringkasan", "Perbandingan Metode", "Panel Gen"}
        notebook_active = st.session_state.page in notebook_routes
        training_active = st.session_state.page in {"Hasil Pelatihan", "Panel Gen Pelatihan"}
        with st.expander("Gallery Hasil Pelatihan", expanded=training_active,
                         key=f"workspace_training_{training_active}"):
            destination("Hasil Pelatihan", "Hasil Pelatihan", ":material/monitoring:")
            destination("Panel Gen Pelatihan", "Panel Gen", ":material/table_view:")
        destination("Kamus Biologi", "Kamus Biologi", ":material/menu_book:")

        with st.container(key="sidebar_support"):
            render_workspace_controls(state_directory())
            with st.container(key="reference_section"):
                st.divider()
                with st.expander("Referensi Eksperimen", expanded=notebook_active,
                                 key=f"workspace_notebook_{notebook_active}"):
                    destination("Ringkasan", "Ringkasan", ":material/home:")
                    destination("Perbandingan Metode", "Perbandingan Metode", ":material/monitoring:")
                    destination("Panel Gen", "Panel Gen Eksperimen", ":material/table_view:")

    return st.session_state.page


def header_controls(preprocessor: dict | None, classifier) -> None:
    columns = st.columns([1.25, 2.25, 1.25, 1.05], vertical_alignment="bottom")
    with columns[0]:
        st.markdown('<div class="eyebrow">Pipeline aktif</div>', unsafe_allow_html=True)
        st.markdown('<span class="pipeline-badge">Part A - XGBoost</span>', unsafe_allow_html=True)
    with columns[1]:
        st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
        if preprocessor:
            st.markdown(
                f"**L1 C={preprocessor['l1_c']:g} - "
                f"{preprocessor['feature_count']:,} gen**"
            )
        else:
            st.markdown("**Bundle model belum tersedia**")
    with columns[2]:
        with st.popover(
            "Unggah Sampel TPM",
            icon=":material/upload:",
            width="stretch",
        ):
            uploaded_file = st.file_uploader(
                "CSV, TSV, atau GCT",
                type=["csv", "tsv", "txt", "gct"],
                help="Baris dapat berupa sampel atau gen; aplikasi akan mendeteksi orientasinya.",
                key="tpm_upload",
            )
            if uploaded_file is not None and st.button(
                "Validasi dan prediksi",
                icon=":material/science:",
                type="primary",
                width="stretch",
            ):
                if not preprocessor or classifier is None:
                    st.error("Bundle model belum tersedia.")
                else:
                    try:
                        matrix = parse_expression_file(
                            uploaded_file,
                            uploaded_file.name,
                        )
                        aligned, audit = audit_and_align_input(matrix, preprocessor)
                        st.session_state.input_audit = audit
                        st.session_state.input_filename = uploaded_file.name
                        if audit.ready:
                            predictions, probabilities = predict_tissues(
                                aligned,
                                preprocessor,
                                classifier,
                            )
                            st.session_state.predictions = predictions
                            st.session_state.probabilities = probabilities
                            st.session_state.active_sample = str(
                                predictions.iloc[0]["SAMPID"]
                            )
                            st.session_state.pop("active_sample_selector", None)
                            st.success("Prediksi selesai.")
                            st.rerun()
                        else:
                            st.error(audit.message)
                    except Exception as error:
                        st.error(f"Berkas tidak dapat diproses: {error}")
    with columns[3]:
        if "predictions" in st.session_state:
            export_table = st.session_state.predictions.copy()
            probabilities = st.session_state.probabilities.reset_index()
            export_table = export_table.merge(probabilities, on="SAMPID")
            download_button(
                "Unduh Hasil",
                data=export_table.to_csv(index=False).encode("utf-8"),
                file_name="gtex_tissue_predictions.csv",
                mime="text/csv",
                icon=":material/download:",
                type="primary",
                width="stretch",
            )
        else:
            st.button(
                "Unduh Hasil",
                icon=":material/download:",
                disabled=True,
                width="stretch",
            )
    st.divider()


def initialize_demo(preprocessor: dict, classifier) -> None:
    if "predictions" in st.session_state:
        return
    demo_path = DATA_DIR / "demo_tpm.csv"
    if not demo_path.is_file():
        return
    with demo_path.open("rb") as handle:
        matrix = parse_expression_file(handle, demo_path.name)
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    if not audit.ready:
        raise ValueError(f"Data demo gagal diaudit: {audit.message}")
    predictions, probabilities = predict_tissues(aligned, preprocessor, classifier)
    st.session_state.predictions = predictions
    st.session_state.probabilities = probabilities
    st.session_state.input_audit = audit
    st.session_state.input_filename = "demo_tpm.csv"
    st.session_state.active_sample = str(predictions.iloc[0]["SAMPID"])


def render_prediction_page(preprocessor: dict | None, classifier, tables: dict) -> None:
    if preprocessor and classifier is not None:
        initialize_demo(preprocessor, classifier)
    header_controls(preprocessor, classifier)

    if "predictions" not in st.session_state:
        st.info(
            "Unggah matriks TPM untuk memulai. Jika bundle model baru saja dibuat, "
            "muat ulang aplikasi agar sampel demo tersedia."
        )
        return

    predictions = st.session_state.predictions
    probabilities = st.session_state.probabilities
    sample_ids = predictions["SAMPID"].astype(str).tolist()
    if st.session_state.active_sample not in sample_ids:
        st.session_state.active_sample = sample_ids[0]
    if st.session_state.get("active_sample_selector") not in sample_ids:
        st.session_state.active_sample_selector = st.session_state.active_sample

    active_sample = st.selectbox(
        "Sampel aktif",
        sample_ids,
        key="active_sample_selector",
        label_visibility="collapsed",
    )
    st.session_state.active_sample = active_sample
    prediction_row = predictions.loc[predictions["SAMPID"].eq(active_sample)].iloc[0]
    probability_row = probabilities.loc[active_sample].sort_values(ascending=False)
    predicted_tissue = str(prediction_row["predicted_tissue"])
    confidence = float(prediction_row["confidence"])
    audit = st.session_state.input_audit
    safe_filename = html.escape(str(st.session_state.input_filename))
    safe_sample_id = html.escape(str(active_sample))

    left, center, right = st.columns([1.0, 2.8, 2.1], gap="medium")
    with left:
        st.markdown('<div class="eyebrow">Kontrol atlas</div>', unsafe_allow_html=True)
        if "anatomy_view" not in st.session_state:
            st.session_state.anatomy_view = "left"
        if st.button(
            "Kiri",
            icon=":material/rotate_left:",
            key="atlas_left",
            type="primary" if st.session_state.anatomy_view == "left" else "secondary",
            width="stretch",
        ):
            st.session_state.anatomy_view = "left"
            st.rerun()
        if st.button(
            "Depan",
            icon=":material/person:",
            key="atlas_front",
            type="primary" if st.session_state.anatomy_view == "front" else "secondary",
            width="stretch",
        ):
            st.session_state.anatomy_view = "front"
            st.rerun()
        if st.button(
            "Kanan",
            icon=":material/rotate_right:",
            key="atlas_right",
            type="primary" if st.session_state.anatomy_view == "right" else "secondary",
            width="stretch",
        ):
            st.session_state.anatomy_view = "right"
            st.rerun()
        zoom = st.slider("Zoom", 70, 125, 92, 5)
        st.caption(
            "Tampilan multiview adalah implementasi 3D-lite. Highlight anatomi "
            "dinamis saat ini tersedia untuk Thyroid."
        )

        st.markdown('<div class="sample-box">', unsafe_allow_html=True)
        st.markdown("**Sampel aktif**")
        st.markdown(
            f'<div class="detail-row"><span>File</span><strong>{safe_filename}</strong></div>'
            f'<div class="detail-row"><span>SAMPID</span><strong>{safe_sample_id}</strong></div>'
            f'<div class="detail-row"><span>Gen cocok</span><strong>{audit.matched_gene_count:,}</strong></div>'
            f'<div class="detail-row"><span>Status</span><strong class="status-good">Selesai</strong></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with center:
        if predicted_tissue == "Thyroid":
            anatomy_file = APP_DIR / "assets" / f"anatomy_{st.session_state.anatomy_view}.png"
            anatomy_caption = "Thyroid disorot sebagai jaringan hasil prediksi."
        else:
            anatomy_file = APP_DIR / "assets" / "anatomy_neutral.png"
            anatomy_caption = (
                "Atlas netral ditampilkan karena highlight spesifik jaringan ini "
                "belum tersedia pada MVP."
            )
        image(anatomy_file, width=int(540 * zoom / 100))
        st.caption(anatomy_caption)

        distribution = probability_row.rename("probability").reset_index()
        distribution.columns = ["tissue", "probability"]
        st.markdown("#### Distribusi probabilitas seluruh jaringan")
        full_chart = (
            alt.Chart(distribution)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X(
                    "tissue:N",
                    sort="-y",
                    title=None,
                    axis=alt.Axis(labelAngle=-35, labelLimit=95),
                ),
                y=alt.Y(
                    "probability:Q",
                    title="Probabilitas",
                    axis=alt.Axis(format="%"),
                    stack=None,
                ),
                color=alt.condition(
                    alt.datum.tissue == predicted_tissue,
                    alt.value("#0f8b8d"),
                    alt.value("#6761db"),
                ),
                tooltip=[
                    alt.Tooltip("tissue:N", title="Jaringan"),
                    alt.Tooltip("probability:Q", title="Probabilitas", format=".2%"),
                ],
            )
            .properties(height=170)
        )
        st.altair_chart(full_chart, width="stretch")

    with right:
        st.markdown('<div class="eyebrow">Hasil prediksi</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="result-title">Prediksi: <strong>{predicted_tissue}</strong></div>',
            unsafe_allow_html=True,
        )
        st.caption("Kelas jaringan GTEx")
        st.markdown('<div class="eyebrow" style="margin-top:1.2rem">Keyakinan model</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="confidence">{confidence:.1%}</div>',
            unsafe_allow_html=True,
        )
        st.progress(confidence)

        st.markdown("#### Probabilitas jaringan teratas")
        top_three = probability_row.head(3).rename("probability").reset_index()
        top_three.columns = ["tissue", "probability"]
        top_chart_base = alt.Chart(top_three).encode(
            y=alt.Y(
                "tissue:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=135),
            ),
            x=alt.X(
                "probability:Q",
                title=None,
                scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1.0]),
                stack=None,
            ),
        )
        top_bars = (
            top_chart_base
            .mark_bar(cornerRadiusEnd=3, size=18)
            .encode(
                color=alt.condition(
                    alt.datum.tissue == predicted_tissue,
                    alt.value("#0f8b8d"),
                    alt.value("#6861db"),
                ),
                tooltip=[
                    alt.Tooltip("tissue:N", title="Jaringan"),
                    alt.Tooltip("probability:Q", title="Probabilitas", format=".2%"),
                ],
            )
        )
        top_labels = top_chart_base.mark_text(
            align="left",
            baseline="middle",
            dx=5,
            color="#344054",
            fontSize=10,
        ).encode(text=alt.Text("probability:Q", format=".1%"))
        top_chart = (top_bars + top_labels).properties(height=112)
        st.altair_chart(top_chart, width="stretch")
        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown("#### Kualitas input")
        quality_rows = [
            ("Validasi ID gen", "Passed" if audit.ready else "Review"),
            (
                "Kecocokan gen",
                f"{audit.matched_gene_count:,} / {preprocessor['feature_count']:,}",
            ),
        ]
        for label, value in quality_rows:
            st.markdown(
                f'<div class="detail-row"><span>{label}</span><strong>{value}</strong></div>',
                unsafe_allow_html=True,
            )
        st.caption(
            f"Negatif {audit.negative_value_count:,} - "
            f"kosong {audit.missing_value_count:,} - "
            f"tak hingga {audit.infinite_value_count:,} - "
            f"duplikat {audit.duplicate_gene_count:,}"
        )
        if audit.appears_log_transformed:
            st.warning(
                "Distribusi nilai tampak kecil dan dapat menyerupai data yang sudah "
                "ditransformasi. Pastikan input masih berupa TPM mentah."
            )

        st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
        st.markdown("#### Detail model")
        details = [
            ("Algoritma", "XGBoost"),
            ("Seleksi fitur", "L1-Regularization"),
            ("C", f"{preprocessor['l1_c']:g}"),
            ("Jumlah gen", f"{preprocessor['feature_count']:,}"),
            ("Jumlah kelas", f"{len(preprocessor['classes'])}"),
        ]
        for label, value in details:
            st.markdown(
                f'<div class="detail-row"><span>{label}</span><strong>{value}</strong></div>',
                unsafe_allow_html=True,
            )

def render_comparison_page(tables: dict) -> None:
    st.markdown('''<style>
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) { background: #fff; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stMain"] .block-container {
        padding: 1.4rem 2rem 1.5rem;
    }
    [data-testid="stElementContainer"]:has(.comparison-page-marker),
    [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style) { display: none; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stDivider"] { margin: 1.4rem 0 .7rem; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stSubheader"] h3 { font-size: 1.4rem; }
    @media (max-width: 640px) {
        [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stMain"] .block-container { padding: 1rem; }
    }
    </style><div class="comparison-page-marker"></div>''', unsafe_allow_html=True)
    st.title("Perbandingan metode")
    st.caption("CV Folding untuk memilih konfigurasi, holdout untuk evaluasi akhir.")
    summary = tables["run_summary"]
    validation = tables["experiment_validation"]
    cv_summary = clarify_eligibility_reasons(tables["cv_summary"], tables["cv_folds"])
    cv_summary["eligibility_reason"] = cv_summary["eligibility_reason"].str.replace(" | ", ", ", regex=False)
    heading, controls = st.columns([3, 1])
    with heading:
        st.subheader("1. Hasil CV Folding")
        st.markdown(f"**Data development: {summary['development_samples']:,} sampel - {summary['development_donors']:,} donor**".replace(",", "."))
        st.caption(
            f"{summary['selector_configurations']} konfigurasi seleksi fitur + {summary['baseline_configurations']} baseline - "
            f"{summary['cv_folds']} fold - {len(tables['cv_folds'])} hasil evaluasi"
        )
    with controls:
        metric_label = st.selectbox(
            "Metrik",
            ["Macro-F1", "Macro precision", "Macro recall", "Balanced accuracy", "Accuracy"],
        )
        zoom_scores = st.checkbox("Perbesar rentang skor pada grafik", key="comparison_zoom", value=False)
    metric_lookup = {
        "Macro-F1": ("mean_macro_f1", "std_macro_f1"),
        "Macro precision": ("mean_macro_precision", "std_macro_precision"),
        "Macro recall": ("mean_macro_recall", "std_macro_recall"),
        "Balanced accuracy": ("mean_balanced_accuracy", "std_balanced_accuracy"),
        "Accuracy": ("mean_accuracy", "std_accuracy"),
    }
    mean_column, std_column = metric_lookup[metric_label]
    selector_data = cv_summary.loc[
        cv_summary["experiment_type"].eq("feature_selection")
    ].copy()
    baseline_data = cv_summary.loc[
        cv_summary["experiment_type"].eq("baseline")
    ].copy()
    selector_data["lower"] = selector_data[mean_column] - selector_data[std_column]
    selector_data["upper"] = selector_data[mean_column] + selector_data[std_column]
    selector_legend = alt.Legend(
        orient="bottom", columns=3, symbolOpacity=1,
        symbolType="stroke", symbolStrokeWidth=3, labelColor="#344054", labelFontSize=13,
    )
    selector_colors = alt.Scale(range=["#4477aa", "#0f8b8d", "#9c6b21", "#5b5f97", "#587d71"])

    line = (
        alt.Chart(selector_data)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("genes:O", title="Jumlah gen terpilih"),
            y=alt.Y(
                f"{mean_column}:Q",
                title=f"Rata-rata {metric_label}" + (" - rentang diperbesar" if zoom_scores else ""),
                scale=alt.Scale(zero=False) if zoom_scores else alt.Scale(domain=[0, 1], nice=False),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color("selector_label:N", title="Selektor", scale=selector_colors, legend=selector_legend),
            tooltip=[
                alt.Tooltip("selector_label:N", title="Selektor"),
                alt.Tooltip("genes:O", title="Gen"),
                alt.Tooltip(f"{mean_column}:Q", title=metric_label, format=".2%"),
                alt.Tooltip(
                    "eligible_for_selector_ranking:N",
                    title="Eligible sebagai selector",
                ),
            ],
        )
    )
    points = line.mark_point(size=70, filled=True, strokeWidth=2).encode(
        stroke=alt.Color("selector_label:N", scale=selector_colors, legend=None),
        fill=alt.condition(
            alt.datum.eligible_for_selector_ranking,
            alt.Color("selector_label:N", scale=selector_colors, legend=None),
            alt.value("#ffffff"),
        ),
    )
    band = (
        alt.Chart(selector_data)
        .mark_errorband(opacity=0.10, clip=True)
        .encode(
            x=alt.X("genes:O"),
            y=alt.Y("lower:Q", title=None, axis=alt.Axis(format=".0%")),
            y2="upper:Q",
            color=alt.Color("selector_label:N", title="Selektor", scale=selector_colors, legend=selector_legend),
        )
    )
    baseline_rule = (
        alt.Chart(baseline_data)
        .mark_rule(color="#172033", strokeDash=[7, 5], strokeWidth=2)
        .encode(
            y=alt.Y(f"{mean_column}:Q", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("configuration_label:N", title="Baseline"),
                alt.Tooltip(f"{mean_column}:Q", title=metric_label, format=".2%"),
                alt.Tooltip("genes:Q", title="Gen"),
            ],
        )
    )
    st.altair_chart(
        (band + line + baseline_rule + points).resolve_legend(color="shared").properties(height=280),
        width="stretch",
    )
    st.caption(
        "Titik berongga: tidak memenuhi syarat seleksi. Garis hitam: baseline. Pita: ±1 SD antar-fold."
    )

    # Display the notebook's locked choice instead of choosing again in the UI.
    saved_choice = tables["run_summary"]
    best_row = selector_data.loc[
        selector_data["selector_id"].eq(saved_choice["best_selector_id"])
        & selector_data["genes"].eq(saved_choice["best_feature_count"])
    ].iloc[0]
    baseline_row = baseline_data.iloc[0]
    st.markdown(
        f"**Konfigurasi terpilih pada notebook:** {best_row['selector_label']} - "
        f"{int(best_row['genes']):,}".replace(",", ".")
        + f" gen - Macro-F1 CV {best_row['mean_macro_f1']:.2%}".replace(".", ",")
    )
    # Show the locked configuration and ANOVA/MI at the same k; keep the full grid below.
    preview = cv_summary.loc[
        cv_summary["configuration_id"].eq(best_row["configuration_id"])
        | (cv_summary["feature_selection"].isin(["anova", "mutual_information"])
           & cv_summary["genes"].eq(best_row["genes"]))
        | cv_summary["experiment_type"].eq("baseline")
    ].copy()
    preview["preview_order"] = preview["feature_selection"].map({"l1": 0, "anova": 1, "mutual_information": 2}).fillna(3)
    preview = preview.sort_values("preview_order")
    preview["selection_status"] = preview["configuration_id"].eq(best_row["configuration_id"]).map({True: "Terpilih", False: "Pembanding"})
    preview["configuration_label"] = preview["selector_label"].where(preview["experiment_type"].ne("baseline"), "Baseline")
    st.markdown("#### Ringkasan konfigurasi")
    st.dataframe(
        preview[["configuration_label", "genes", mean_column, std_column, "selection_status"]].style.apply(
            lambda row: ["background-color: #edf7f7; color: #087278; font-weight: 600" if row["selection_status"] == "Terpilih" else ""] * len(row), axis=1,
        ).format(
            {
                "genes": lambda value: f"{int(value):,}".replace(",", "."),
                mean_column: lambda value: f"{value:.2%}".replace(".", ","),
                std_column: lambda value: f"{value * 100:.2f}".replace(".", ","),
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "configuration_label": "Metode",
            "genes": "Gen",
            mean_column: f"Rerata {metric_label} CV (%)",
            std_column: "SD (poin persentase)",
            "selection_status": "Status",
        },
    )
    with st.expander(f"Lihat {len(cv_summary)} konfigurasi"):
        st.dataframe(
            cv_summary[["performance_rank", "configuration_label", "experiment_type", "genes", mean_column, std_column,
                        "eligible_for_selector_ranking", "eligibility_reason"]].style.format({
                "genes": lambda value: f"{int(value):,}".replace(",", "."),
                mean_column: lambda value: f"{value:.2%}".replace(".", ","),
                std_column: lambda value: f"{value * 100:.2f}".replace(".", ","),
            }), width="stretch", hide_index=True,
            column_config={
                "performance_rank": "Peringkat Macro-F1",
                "configuration_label": st.column_config.TextColumn("Konfigurasi", width="large"),
                "experiment_type": "Jenis eksperimen", "genes": "Gen",
                mean_column: f"Rerata {metric_label} CV (%)", std_column: "SD (poin persentase)",
                "eligible_for_selector_ranking": st.column_config.CheckboxColumn("Memenuhi syarat seleksi"),
                "eligibility_reason": st.column_config.TextColumn("Keterangan", width="large"),
            },
        )
        st.caption("Peringkat memakai Macro-F1. Konfigurasi terpilih tetap mengikuti notebook dan harus memenuhi syarat seleksi. Baseline bukan kandidat selector.")
        st.caption(
            f"Validasi pipeline: {validation['configuration_count']} konfigurasi, "
            f"{validation['folds_per_configuration']} fold per konfigurasi, {validation['fold_level_results']} hasil fold-level."
        )
        st.caption(f"Baseline memakai seluruh {int(baseline_row['genes']):,} gen kandidat tanpa seleksi fitur.".replace(",", "."))

    st.divider()
    st.subheader("2. Hasil akhir - holdout")
    st.markdown(f"**{summary['test_samples']:,} sampel - {summary['test_donors']:,} donor**".replace(",", "."))
    st.caption(
        "Konfigurasi terpilih CV dibandingkan dengan baseline. Keduanya dilatih pada data development yang sama; holdout tidak digunakan untuk memilih ulang model."
    )

    holdout = tables["final_holdout"].copy()
    if holdout.empty:
        st.info(
            "Hasil perbandingan holdout belum tersedia. Jalankan LANGKAH 37A sampai "
            "39C pada notebook Part A, lalu salin ekspor terbaru ke folder data aplikasi."
        )
    else:
        selected_label = f"{best_row['selector_label']}, " + f"{int(best_row['genes']):,} gen".replace(",", ".")
        baseline_label = "Baseline, " + f"{int(baseline_row['genes']):,} gen".replace(",", ".")
        model_order = [selected_label, baseline_label]
        holdout["model"] = holdout["experiment_type"].map(
            {
                "feature_selection": selected_label,
                "baseline": baseline_label,
            }
        ).fillna(holdout["configuration_label"])

        selected_holdout = holdout.loc[
            holdout["experiment_type"].eq("feature_selection")
        ].iloc[0]
        baseline_holdout = holdout.loc[
            holdout["experiment_type"].eq("baseline")
        ].iloc[0]

        st.markdown(
            "**Selisih Macro-F1: "
            + f"{(selected_holdout['macro_f1'] - baseline_holdout['macro_f1']) * 100:+.2f}".replace(".", ",")
            + " poin persentase**"
        )

        holdout_metric_labels = {
            "macro_f1": "Macro-F1",
            "macro_precision": "Macro precision",
            "macro_recall": "Macro recall",
            "balanced_accuracy": "Balanced accuracy",
            "accuracy": "Accuracy",
        }
        holdout_long = holdout.melt(
            id_vars=["model"],
            value_vars=list(holdout_metric_labels),
            var_name="metric",
            value_name="score",
        )
        holdout_long["metric"] = holdout_long["metric"].map(
            holdout_metric_labels
        )
        holdout_long["display_score"] = holdout_long["score"].map(lambda value: f"{value:.2%}".replace(".", ","))
        holdout_base = (
            alt.Chart(holdout_long)
            .encode(
                x=alt.X("metric:N", title=None, sort=list(holdout_metric_labels.values())),
                xOffset=alt.XOffset("model:N", sort=model_order),
                y=alt.Y("score:Q", title="Skor holdout", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "model:N",
                    title=None, legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(domain=model_order, range=["#078c92", "#667085"]),
                ),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("metric:N", title="Metrik"),
                    alt.Tooltip("score:Q", title="Skor", format=".2%"),
                ],
            )
        )
        holdout_chart = (
            holdout_base.mark_bar()
            + holdout_base.mark_text(dy=-8, fontSize=12, color="#344054").encode(text="display_score:N")
        ).properties(height=240, padding={"top": 18})
        st.altair_chart(holdout_chart, width="stretch")

        holdout_table = holdout[
            [
                "configuration_label",
                "genes",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "balanced_accuracy",
                "accuracy",
            ]
        ].copy()
        holdout_table["configuration_label"] = holdout["model"]
        with st.expander("Metrik lengkap holdout"):
            st.dataframe(
                holdout_table.style.format(
                    {
                        "genes": lambda value: f"{int(value):,}".replace(",", "."),
                        "macro_f1": lambda value: f"{value:.2%}".replace(".", ","),
                        "macro_precision": lambda value: f"{value:.2%}".replace(".", ","),
                        "macro_recall": lambda value: f"{value:.2%}".replace(".", ","),
                        "balanced_accuracy": lambda value: f"{value:.2%}".replace(".", ","),
                        "accuracy": lambda value: f"{value:.2%}".replace(".", ","),
                    }
                ),
                width="stretch", hide_index=True,
                column_config={
                    "configuration_label": "Konfigurasi final", "genes": "Jumlah gen",
                    "macro_f1": "Macro-F1", "macro_precision": "Macro-precision",
                    "macro_recall": "Macro-recall", "balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy",
                },
            )

        per_tissue_holdout = tables["final_holdout_per_tissue"].copy()
        if not per_tissue_holdout.empty:
            per_tissue_holdout["model"] = per_tissue_holdout[
                "configuration_id"
            ].map(
                {
                    str(selected_holdout["configuration_id"]): selected_label,
                    str(baseline_holdout["configuration_id"]): baseline_label,
                }
            )
            tissue_chart = (
                alt.Chart(per_tissue_holdout)
                .mark_bar()
                .encode(
                    y=alt.Y("tissue:N", title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False, labelFontSize=13)),
                    x=alt.X(
                        "f1-score:Q",
                        title="F1-score per jaringan",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    yOffset=alt.YOffset("model:N", sort=model_order),
                    color=alt.Color(
                        "model:N",
                        title="Model final",
                        scale=alt.Scale(domain=model_order, range=["#078c92", "#667085"]),
                    ),
                    tooltip=[
                        alt.Tooltip("tissue:N", title="Jaringan"),
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("f1-score:Q", title="F1-score", format=".2%"),
                    ],
                )
                .properties(height=max(430, per_tissue_holdout["tissue"].nunique() * 40))
            )
            with st.expander("F1-score per jaringan"):
                st.altair_chart(tissue_chart, width="stretch")


restore_location(state_directory())
if not st.session_state.get('workspace_entered', False):
    render_workspace_launcher(state_directory(), APP_DIR / 'assets' / 'brand_dna.png')
    st.stop()

try:
    tables = reference_tables(file_version(DATA_DIR))
except (ValueError, FileNotFoundError) as data_error:
    st.error(f"Ekspor Part A tidak konsisten atau belum tersedia: {data_error}")
    st.info("Sinkronkan tabel dari results/notebook_run_500_1000_2000_l1_grid/part_a_only ke folder data aplikasi.")
    st.stop()
page = navigation()
remember_location()
# Streamlit drops widget values for widgets not drawn in a run; re-assigning keeps the training form
# intact while the user visits other pages. Workspace switches clear these keys separately.
for key in TRAINING_FORM_KEYS:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]
active_state_root = current_workspace_root(state_directory())

preprocessor = None
classifier = None
if bundled_models_enabled(active_state_root) and model_bundle_available():
    try:
        preprocessor, classifier = prediction_bundle(file_version(MODEL_DIR))
    except Exception as bundle_error:
        st.sidebar.error(f"Bundle model gagal dimuat: {bundle_error}")

if page == "Pelatihan Model":
    render_training_page(APP_DIR / 'training_data', active_state_root)
elif page == "Hasil Pelatihan":
    render_training_results_page(APP_DIR / 'training_data', active_state_root)
elif page == "Panel Gen Pelatihan":
    render_training_gene_page(active_state_root)
elif page == "Ringkasan":
    render_overview_page(tables)
elif page == "Perbandingan Metode":
    render_comparison_page(tables)
elif page == "Panel Gen":
    render_gene_page(tables)
elif page in {"Kamus Biologi", "Kamus Istilah"}:
    render_glossary_page()
else:
    render_offline_prediction_page(preprocessor, classifier, tables)

