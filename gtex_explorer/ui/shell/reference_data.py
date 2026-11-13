"""Cached notebook result tables shared by the pages."""

from __future__ import annotations

import streamlit as st


@st.cache_data(show_spinner="Memuat referensi eksperimen...")
def reference_tables(snapshot_version: tuple) -> dict:
    """Load the result tables once per data version."""
    del snapshot_version
    from gtex_explorer.experiments.reference_tables import load_reference_tables

    return load_reference_tables()


def file_version(directory) -> tuple:
    """Refresh caches after replacing exported tables or model files."""
    return tuple(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size)
        for path in sorted(directory.iterdir())
        if path.is_file()
    )


def load_tables_or_stop() -> dict:
    """Return the result tables, or explain the problem and stop the page."""
    from gtex_explorer.experiments.reference_tables import DATA_DIR

    try:
        return reference_tables(file_version(DATA_DIR))
    except (ValueError, FileNotFoundError) as data_error:
        st.error(f"Ekspor Part A tidak konsisten atau belum tersedia: {data_error}")
        st.info("Sinkronkan tabel dari results/notebook_run_500_1000_2000_l1_grid/part_a_only ke folder data aplikasi.")
        st.stop()
