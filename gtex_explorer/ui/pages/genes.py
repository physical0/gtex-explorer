"""Explore the deployed panel and the saved panels from all CV fits."""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st
from gtex_explorer.platform.cloud.media import download_button

from gtex_explorer.experiments.gene_panels import cv_gene_panel
from gtex_explorer.experiments.selector_diagnostics import render_selector_metrics, render_diagnostic_details, render_stability_details


GENE_PAGE_STYLE = """
<style>
[data-testid="stAppViewContainer"]:has(.gene-page-marker) { background: #fff; }
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-testid="stMain"] .block-container {
    padding: 1.4rem 2rem 1.5rem;
}
[data-testid="stElementContainer"]:has(.gene-page-marker),
[data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style) { display: none; }
.gene-identity { margin: .6rem 0 .15rem; font-size: 1.1rem; font-weight: 650; color: var(--ink); }
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab-list"] {
    gap: 1.5rem; border-bottom: 1px solid var(--line); margin: .5rem 0 .4rem;
}
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab"] {
    padding: .75rem .4rem; font-size: 1rem; font-weight: 600; background: transparent;
}
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab"][aria-selected="true"] { color: #0f8b8d; }
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab-highlight"] { background-color: #0f8b8d; }
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-testid="stSubheader"] h3 { font-size: 1.2rem; }
@media (max-width: 640px) {
    [data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-testid="stMain"] .block-container { padding: 1rem; }
    [data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab-list"] { gap: .5rem; }
    [data-testid="stAppViewContainer"]:has(.gene-page-marker) [data-baseweb="tab"] { font-size: .85rem; padding: .6rem .2rem; }
}
</style>
"""


def render_gene_page(tables: dict) -> None:
    st.markdown(GENE_PAGE_STYLE, unsafe_allow_html=True)
    st.markdown('<div class="gene-page-marker"></div>', unsafe_allow_html=True)
    st.title("Panel gen")
    source = st.radio(
        "Sumber panel", ["Model final", "Cross-validation"],
        horizontal=True, key="gene_panel_source",
    )
    summary = tables["run_summary"]
    baseline = False
    fit = None
    if source == "Model final":
        genes = tables["selected_genes"].copy()
        selector = summary["best_selector_id"]
        c_value = summary.get("best_l1_c")
        context = f"Data development: {summary['development_samples']:,} sampel - {summary['development_donors']:,} donor".replace(",", ".")
    else:
        rankings = tables.get("cv_gene_rankings", pd.DataFrame())
        if rankings.empty:
            st.warning("Panel CV belum diekspor. Jalankan export_cv_gene_panels.py dari checkpoint notebook yang sudah selesai.")
            return
        fits = tables["cv_folds"]
        choices = summary["selector_ids"] + [summary["baseline_id"]]
        labels = tables["cv_summary"].dropna(subset=["selector_id"]).set_index("selector_id")["selector_label"].to_dict()
        labels[summary["baseline_id"]] = "Baseline — tanpa seleksi fitur"
        controls = st.columns([2, 1, 1])
        selector = controls[0].selectbox(
            "Metode seleksi fitur", choices, format_func=lambda value: labels[value],
            index=choices.index(summary["best_selector_id"]), key="gene_cv_method",
        )
        baseline = selector == summary["baseline_id"]
        method_fits = fits.loc[
            fits["configuration_id"].eq(selector) if baseline else fits["selector_id"].eq(selector)
        ]
        counts = sorted(method_fits["genes"].astype(int).unique().tolist())
        count = controls[1].selectbox(
            "Jumlah gen", counts, index=counts.index(summary["best_feature_count"]) if summary["best_feature_count"] in counts else 0,
            format_func=lambda value: f"{value:,}".replace(",", "."), key="gene_cv_count",
        )
        folds = sorted(method_fits["fold"].astype(int).unique().tolist())
        fold = controls[2].selectbox("Fold", folds, key="gene_cv_fold")
        matching_fits = method_fits.loc[method_fits["genes"].eq(count) & method_fits["fold"].eq(fold)]
        if len(matching_fits) != 1:
            st.error("Konfigurasi/fold tidak tersedia atau duplikat pada hasil CV.")
            return
        fit = matching_fits.iloc[0]
        genes = cv_gene_panel(rankings, fit)
        c_value = fit.get("l1_c")
        context = f"Data training dan data validation fold {fold}"
        # All selectors and the baseline share these saved CV partitions.
        fold_metadata = tables.get("selector_diagnostics", {}).get("anova_summary", pd.DataFrame())
        if not fold_metadata.empty:
            sizes = fold_metadata.loc[fold_metadata["fold"].eq(fold), "training_samples"].dropna().unique()
            if len(sizes) == 1 and 0 < int(sizes[0]) < int(summary["development_samples"]):
                training_samples = int(sizes[0])
                validation_samples = int(summary["development_samples"]) - training_samples
                context = f"Data training: {training_samples:,} sampel - Data validation: {validation_samples:,} sampel".replace(",", ".")

    method = "Tanpa seleksi fitur" if baseline else (
        "Regresi logistik L1" if selector.startswith("l1_") else
        "Mutual Information" if selector == "mutual_information" else "ANOVA"
    )
    pills = [method]
    if selector.startswith("l1_"):
        pills.append(f"C = {float(c_value):g}")
    pills.extend([f"{len(genes):,} gen".replace(",", "."), "XGBoost"])
    if fit is not None:
        pills.append(f"Fold {int(fit['fold'])}")
    st.markdown(
        '<div class="gene-identity">'
        + ', '.join(
            f'<strong>{html.escape(pill)}</strong>' for pill in pills
        ) + '</div>', unsafe_allow_html=True,
    )
    st.caption(context)
    if fit is not None:
        if not baseline:
            zero_count = int(fit["zero_scores_in_top_k"]) if pd.notna(fit.get("zero_scores_in_top_k")) else int(genes["selection_score"].le(0).sum())
            if zero_count:
                zero_label = f"{zero_count:,}".replace(",", ".")
                count_label = f"{len(genes):,}".replace(",", ".")
                interpretation = (
                    "Jumlah gen berkoefisien nonzero kurang dari top-k pada fold ini. "
                    if selector.startswith("l1_") else "Jumlah gen berskor positif kurang dari top-k pada fold ini. "
                )
                st.warning(
                    interpretation + f"{zero_label} dari {count_label} gen dalam top-k memiliki skor nol menurut batas numerik notebook. "
                    "Gen tersebut tetap masuk matriks fit untuk memenuhi jumlah top-k, "
                    "tetapi tidak dihitung sebagai gen berskor nonzero oleh selektor. Urutan skor seri mengikuti aturan notebook."
                )
            if not bool(fit["eligible_for_primary_comparison"]):
                st.caption("Konfigurasi ini tersimpan untuk pemeriksaan, tetapi tidak memenuhi syarat perbandingan utama pada notebook.")

    is_l1 = selector.startswith("l1_")
    tab_labels = ["Daftar gen", "Kinerja XGBoost" if baseline else "Diagnostik metode"]
    if fit is not None and not baseline:
        tab_labels.append("Stabilitas antar-fold")
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        heading, download = st.columns([4, 1])
        heading.subheader("Gen kandidat" if baseline else "Gen terpilih")
        download_slot = download.container()
        filters = st.columns([2, 1, 1] if is_l1 else [3, 1])
        query = filters[0].text_input(
            "Cari ID gen atau kelas dominan" if is_l1 else "Cari ID gen",
            placeholder="Contoh: ENSG00000170549", key="gene_panel_search",
            help="Pencarian mengikuti rentang peringkat dan filter kelas di halaman ini. Naikkan peringkat maksimum untuk mencari lebih jauh dalam panel.",
        )
        max_rank = filters[1].number_input(
            "Jumlah gen ditampilkan" if baseline else "Peringkat maksimum",
            min_value=1, max_value=len(genes), value=min(100, len(genes)), step=25,
            key=f"gene_panel_limit_{source}_{selector}_{len(genes)}",
        )
        dominant = "Semua"
        if is_l1:
            options = ["Semua"] + sorted(genes["dominant_tissue"].dropna().unique().tolist())
            if genes["dominant_tissue"].isna().any():
                options.append("Tanpa koefisien nonzero")
            dominant = filters[2].selectbox("Kelas dominan (koefisien)", options, key="gene_panel_dominant")

        filtered = genes.head(int(max_rank)).copy() if baseline else genes.loc[genes["selection_rank"].le(max_rank)].copy()
        if query:
            matches = filtered["gene"].astype(str).str.contains(query, case=False, regex=False)
            if is_l1:
                matches |= filtered["dominant_tissue"].fillna("").str.contains(query, case=False, regex=False)
            filtered = filtered.loc[matches]
        if dominant != "Semua":
            filtered = filtered.loc[filtered["dominant_tissue"].isna() if dominant == "Tanpa koefisien nonzero" else filtered["dominant_tissue"].eq(dominant)]
        status = f"Menampilkan {len(filtered):,} dari {len(genes):,} gen"
        if not baseline and not query and dominant == "Semua":
            status += f" - Peringkat 1–{int(max_rank):,}"
        elif not baseline:
            status += f" dalam peringkat 1–{int(max_rank):,}"
        st.caption(status.replace(",", "."))
        columns = ["gene"] if baseline else ["selection_rank", "gene", "selection_score"]
        if is_l1:
            columns.append("dominant_tissue")
        score_label = "Skor L1" if is_l1 else "Statistik F" if selector == "anova" else "Skor MI (nat)"
        shown = filtered[columns].style.format({"selection_score": lambda value: f"{value:.6f}".replace(".", ",")} if not baseline else {})
        st.dataframe(
            shown, width="stretch", hide_index=True, height=360,
            column_config={
                "selection_rank": st.column_config.NumberColumn("Peringkat", format="%d", width="small"),
                "gene": st.column_config.TextColumn("ID gen Ensembl", width="large"),
                "selection_score": st.column_config.NumberColumn(score_label, width="small"),
                "dominant_tissue": st.column_config.TextColumn("Kelas dominan (koefisien)", width="large"),
            },
        )
        export = filtered.assign(panel_source="cv_training" if fit is not None else "final_development")
        filename = f"gene_panel_{fit['configuration_id']}_fold_{int(fit['fold'])}.csv" if fit is not None else f"gene_panel_final_{selector}_{len(genes)}.csv"
        with download_slot:
            download_button("Unduh CSV", data=export.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", icon=":material/download:")
        if is_l1:
            with st.expander("Detail koefisien L1"):
                st.caption("Kelas dominan mengikuti koefisien absolut terbesar, bukan TPM tertinggi. Gen tanpa koefisien nonzero tidak diberi kelas dominan.")
                st.dataframe(
                    filtered[["gene", "l2_coefficient_norm", "nonzero_class_count"]],
                    width="stretch", hide_index=True,
                    column_config={
                        "gene": "ID gen Ensembl", "l2_coefficient_norm": st.column_config.NumberColumn("Norma L2 koefisien", format="%.6f"),
                        "nonzero_class_count": "Jumlah kelas dengan koefisien nonzero",
                    },
                )
            st.caption("Skor L1: koefisien absolut maksimum lintas kelas setelah standardisasi.")
        elif baseline:
            st.caption("Seluruh kandidat gen pada fold, diurutkan menurut ID gen tanpa peringkat seleksi.")
        elif selector == "anova":
            st.caption("Gen diurutkan menurut statistik F ANOVA pada data training.")
        else:
            st.caption(f"Gen diurutkan menurut Mutual Information (n_neighbors = {summary['mi_primary_n_neighbors']}, satuan nat).")
        if filtered.empty:
            st.info("Tidak ada gen yang sesuai dengan filter pada rentang peringkat ini.")

    with tabs[1]:
        if not baseline:
            render_selector_metrics(tables, selector, genes, fit)
        if fit is not None:
            if not baseline:
                st.divider()
            st.subheader(f"XGBoost - data validation fold {int(fit['fold'])}")
            for column, (label, field) in zip(st.columns(3), [
                ("Macro-F1 validasi", "macro_f1"), ("Macro-precision validasi", "macro_precision"),
                ("Macro-recall validasi", "macro_recall"),
            ]):
                column.metric(label, f"{fit[field]:.2%}".replace(".", ","))
        if not baseline:
            render_diagnostic_details(tables, selector, filtered, fit)

    if len(tabs) == 3:
        with tabs[2]:
            render_stability_details(tables, selector, fit)
