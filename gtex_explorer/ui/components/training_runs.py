"""Run selector and fit diagnostics shared by the training pages."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from gtex_explorer.display import METHODS, count, decimal, percent
from gtex_explorer.training.gallery import list_gallery
from gtex_explorer.training.jobs import ACTIVE_STATUSES, read_job
from gtex_explorer.training.run_labels import (
    data_settings,
    model_settings,
    parameter_settings,
    run_created,
    run_label,
)
from gtex_explorer.ui.shell.error_messages import user_error


def _run_status(root, state_root):
    """A run's status for the picker. Only an active run is checked against its worker, which may have stopped."""
    try:
        status = json.loads((root / "status.json").read_text(encoding="utf-8"))["status"]
        return read_job(root.name, state_root).status if status in ACTIVE_STATUSES else status
    except (OSError, ValueError, KeyError):
        return None


def choose_run(state_root, key, *, gallery_only=False):
    """Show the run selector and return the chosen run."""
    paths = sorted(
        (Path(state_root) / "runs").glob("run_*/status.json"), key=lambda p: run_created(p.parent), reverse=True
    )
    if gallery_only:
        issues = []
        ready = {entry["run_id"] for entry in list_gallery(state_root, issues=issues)}
        paths = [path for path in paths if path.parent.name in ready]
        for issue in issues:
            st.warning(f"Hasil {issue['run_id'][-6:]} belum lolos verifikasi: {user_error(issue['reason'])}")
    if not paths:
        st.info(
            "Belum ada model terverifikasi di gallery. Periksa Status proses."
            if gallery_only
            else "Belum ada pelatihan pengguna. Mulai dari halaman Latih Model."
        )
        return None
    roots = {p.parent.name: p.parent for p in paths}
    # Every run in a gallery list is finished, so only the full history shows each run's status.
    statuses = {} if gallery_only else {run_id: _run_status(path, state_root) for run_id, path in roots.items()}
    remembered = st.session_state.get("training_selected_run")
    if st.session_state.get(key) not in roots:
        st.session_state[key] = remembered if remembered in roots else next(iter(roots))
    selected = st.selectbox(
        "Pilih hasil pelatihan", list(roots), format_func=lambda x: run_label(roots[x], statuses.get(x)), key=key
    )
    st.session_state.training_selected_run = selected
    return selected


def _settings_table(rows):
    """Show (label, value) rows as a settings table."""
    st.table(pd.DataFrame(rows, columns=["Pengaturan", "Nilai"]).set_index("Pengaturan"))


def show_settings(config, candidate_count=None, *, compact=False):
    """Show a run's settings in the order and with the labels of the Latih Model form.

    `compact` is for use inside an expander: smaller headings, and the parameters shown in place
    because expanders cannot hold another expander.
    """
    heading = (lambda text: st.markdown(f"**{text}**")) if compact else st.subheader
    data_column, model_column = st.columns(2, gap="large")
    with data_column:
        # Pembagian data utama and Pra-pemrosesan.
        heading("Data dan pra-pemrosesan")
        _settings_table(data_settings(config, candidate_count))
    with model_column:
        # Seleksi fitur, Evaluasi tambahan, and Parameter XGBoost.
        heading("Seleksi fitur dan model")
        _settings_table(model_settings(config))
    title = "Parameter XGBoost" + (" dan opsi selector" if config["selector"] == "l1" else "")
    parameters = pd.DataFrame(parameter_settings(config), columns=["Parameter", "Nilai"])
    if compact:
        heading(title)
        st.dataframe(parameters, hide_index=True)
    else:
        with st.expander(title):
            st.dataframe(parameters, hide_index=True)


def show_diagnostics(info):
    """Show the fit diagnostics of a selector, named as on the Latih Model form where it has the setting."""
    labels = {
        "selector": "Metode seleksi fitur",
        "candidate_count": "Gen kandidat",
        "requested_k": "Top-k gen",
        "n_neighbors": "n_neighbors",
        "positive_score_gene_count": "Gen dengan skor MI positif",
        "converged": "Solver konvergen",
        "iterations": "Iterasi solver",
        "l1_c": "C (regularisasi)",
        "nonzero_gene_count": "Gen berkoefisien nonzero",
        "sparsity": "Sparsitas gen",
        "training_samples": "Sampel training",
        "validation_samples": "Sampel validation",
    }
    rows = []
    for key, label in labels.items():
        if key not in info:
            continue
        value = info[key]
        if key == "selector":
            value = METHODS.get(value, value)
        elif key == "sparsity":
            value = percent(value)
        elif type(value) is int:
            value = count(value)
        elif type(value) is float:
            value = decimal(value)
        elif type(value) is bool:
            value = "Ya" if value else "Tidak"
        rows.append({"Diagnostik": label, "Nilai": str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if info.get("legacy_scores_unavailable"):
        st.caption("Riwayat ini hanya menyimpan daftar gen. Skor dan diagnostik fold tidak tersedia.")
    if info.get("selector") == "baseline":
        st.caption("Baseline menggunakan seluruh gen kandidat. Urutan daftar bukan peringkat kepentingan.")
