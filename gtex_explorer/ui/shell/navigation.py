"""Sidebar navigation."""

from __future__ import annotations

import streamlit as st

from gtex_explorer.platform.cloud.media import image
from gtex_explorer.platform.desktop_runtime import state_directory
from gtex_explorer.platform.paths import APP_ROOT as APP_DIR
from gtex_explorer.ui.workspace import render_workspace_controls


def navigation() -> str:
    """Draw the sidebar and return the current page."""
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
        with st.expander(
            "Gallery Hasil Pelatihan", expanded=training_active, key=f"workspace_training_{training_active}"
        ):
            destination("Hasil Pelatihan", "Hasil Pelatihan", ":material/monitoring:")
            destination("Panel Gen Pelatihan", "Panel Gen", ":material/table_view:")
        destination("Kamus Biologi", "Kamus Biologi", ":material/menu_book:")

        with st.container(key="sidebar_support"):
            render_workspace_controls(state_directory())
            with st.container(key="reference_section"):
                st.divider()
                with st.expander(
                    "Referensi Eksperimen", expanded=notebook_active, key=f"workspace_notebook_{notebook_active}"
                ):
                    destination("Ringkasan", "Ringkasan", ":material/home:")
                    destination("Perbandingan Metode", "Perbandingan Metode", ":material/monitoring:")
                    destination("Panel Gen", "Panel Gen Eksperimen", ":material/table_view:")

    return st.session_state.page
