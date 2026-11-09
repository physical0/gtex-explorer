"""Run selector and fit diagnostics shared by the training pages."""

from pathlib import Path

import pandas as pd
import streamlit as st

from gtex_explorer.training.gallery import list_gallery
from gtex_explorer.training.run_labels import METHODS, run_created, run_label
from gtex_explorer.ui.shell.error_messages import user_error


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
    remembered = st.session_state.get("training_selected_run")
    if st.session_state.get(key) not in roots:
        st.session_state[key] = remembered if remembered in roots else next(iter(roots))
    selected = st.selectbox("Pilih hasil pelatihan", list(roots), format_func=lambda x: run_label(roots[x]), key=key)
    st.session_state.training_selected_run = selected
    return selected


def show_diagnostics(info):
    """Show the fit diagnostics of a selector."""
    labels = {
        "selector": "Metode",
        "candidate_count": "Gen kandidat",
        "requested_k": "Gen dipilih",
        "n_neighbors": "Jumlah tetangga MI",
        "positive_score_gene_count": "Gen dengan skor MI positif",
        "converged": "Solver konvergen",
        "iterations": "Iterasi solver",
        "l1_c": "C",
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
            value = f"{value:.2%}"
        elif type(value) is bool:
            value = "Ya" if value else "Tidak"
        rows.append({"Diagnostik": label, "Nilai": str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if info.get("legacy_scores_unavailable"):
        st.caption("Riwayat ini hanya menyimpan daftar gen. Skor dan diagnostik fold tidak tersedia.")
    if info.get("selector") == "baseline":
        st.caption("Baseline menggunakan seluruh gen kandidat. Urutan daftar bukan peringkat kepentingan.")
