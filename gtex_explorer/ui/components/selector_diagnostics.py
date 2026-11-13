"""Selector diagnostics panels for the Panel Gen page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from gtex_explorer.display import count as format_count, decimal, percent


def _number(value, digits=4) -> str:
    """A number with a decimal comma, or that it is unavailable."""
    return decimal(value, digits, missing="Tidak tersedia")


def _percent(value) -> str:
    """A share as a percentage, or that it is unavailable."""
    return percent(value, missing="Tidak tersedia")


def render_selector_metrics(tables: dict, selector: str, genes: pd.DataFrame, fit: pd.Series | None) -> None:
    """Training-data selector metrics, separate from XGBoost validation scores."""
    diagnostics = tables.get("selector_diagnostics", {})
    if fit is not None and not bool(fit.uses_feature_selection):
        return
    st.subheader("Diagnostik seleksi fitur")
    count = len(genes)

    # The final model has only its saved fit. A CV fold has the full exported diagnostics.
    if fit is None:
        saved = tables["selector_fit"].loc[tables["selector_fit"].selector_id.eq(selector)]
        if len(saved) != 1 or not selector.startswith("l1_"):
            st.info("Diagnostik tambahan belum tersedia untuk selector final ini.")
            return
        row = saved.iloc[0]
        n_features = int(tables["run_summary"]["variance_prefilter_genes"])
        nonzero = int(row.positive_or_nonzero_gene_count)
        st.caption(
            f"Data development: {format_count(int(row.training_samples))} sampel - {format_count(int(row.training_donors))} donor."
        )
    elif not diagnostics:
        st.info("Diagnostik tambahan belum tersedia. Ekspor hasil notebook dengan export_selector_diagnostics.py.")
        return
    else:
        candidates = int(tables["run_summary"]["variance_prefilter_genes"])
        if selector in {"anova", "mutual_information"}:
            st.caption(
                f"Data training fold {int(fit.fold)} - panel top-{format_count(count)} dari {format_count(candidates)} gen kandidat."
            )

        # ANOVA metrics.
        if selector == "anova":
            st.caption(
                f"η² dan flag variansi: gen top-k. Spearman: seluruh {format_count(candidates)} gen kandidat. Tidak mengikuti filter pencarian."
            )
            saved = diagnostics["anova_summary"]
            row = saved.loc[saved.fold.eq(fit.fold) & saved.top_k.eq(count)].iloc[0]
            columns = st.columns(4)
            columns[0].metric(
                "Median η²",
                _percent(row.median_eta_squared),
                help="Proporsi variasi ekspresi terkait perbedaan rerata jaringan, diringkas pada gen top-k; bukan akurasi.",
            )
            columns[1].metric(
                "Variansi berbeda (BF)",
                _percent(row.fraction_bf_fdr_005),
                help="Persentase gen top-k dengan adjusted p-value Brown–Forsythe < 0,05. Bersifat eksploratif, bukan uji kelayakan lulus/gagal.",
            )
            columns[2].metric(
                "Rasio variansi ≥ 4",
                _percent(row.fraction_variance_ratio_ge_4),
                help="Persentase gen top-k yang melewati flag deskriptif notebook; variansi nol dicatat terpisah.",
            )
            columns[3].metric(
                "Spearman ANOVA–Welch",
                _number(row.anova_welch_spearman_all_genes),
                help="Kesamaan urutan seluruh gen kandidat saat asumsi kesamaan variansi dilonggarkan. Koefisien −1 sampai 1.",
            )
            return

        # Mutual information metrics.
        if selector == "mutual_information":
            st.caption(
                f"Gen berskor positif dan Spearman: seluruh {format_count(candidates)} gen kandidat. Cutoff: gen top-k. Tidak mengikuti filter pencarian."
            )
            columns = st.columns(4)
            columns[0].metric(
                "Gen berskor MI positif",
                f"{format_count(int(fit.positive_score_gene_count))} / {format_count(int(tables['run_summary']['variance_prefilter_genes']))}",
                help="Jumlah gen kandidat berskor positif pada n_neighbors utama, bukan jumlah gen yang terbukti signifikan.",
            )
            columns[1].metric("Cutoff MI top-k (nat)", _number(fit.score_cutoff, 6))
            saved = diagnostics["mi_neighbors"]
            for index, neighbor in enumerate([5, 10], start=2):
                row = saved.loc[
                    saved.fold.eq(fit.fold)
                    & saved.top_k.eq(count)
                    & saved.neighbors_a.eq(3)
                    & saved.neighbors_b.eq(neighbor)
                ].iloc[0]
                columns[index].metric(
                    f"Spearman 3 vs {neighbor}",
                    _number(row.spearman_all_genes),
                    help="Sensitivitas ranking seluruh gen kandidat terhadap n_neighbors; bukan perbandingan fold atau skor klasifikasi.",
                )
            return

        # L1 metrics.
        saved = diagnostics["l1_fits"]
        row = saved.loc[saved.selector_id.eq(selector) & saved.fold.eq(fit.fold)].iloc[0]
        n_features, nonzero = int(row.n_features), int(row.nonzero_gene_count)

    # Candidate genes, nonzero genes, sparsity, and convergence of the fit.
    scope = f"Data training fold {int(fit.fold)} - " if fit is not None else ""
    st.caption(f"{scope}{format_count(n_features)} gen kandidat sebelum pemilihan top-k.")
    columns = st.columns(4)
    columns[0].metric(
        "Gen nonzero",
        f"{format_count(nonzero)} / {format_count(n_features)}",
        help="Jumlah gen nonzero di antara seluruh kandidat sebelum pemilihan top-k. Koefisien absolut maksimum lintas kelas > 10⁻¹²; satu gen dihitung satu kali.",
    )
    columns[1].metric(
        "Sparsitas gen",
        _percent(1 - nonzero / n_features),
        help="Proporsi kandidat gen tanpa koefisien nonzero di kelas mana pun; komplemen proporsi gen nonzero.",
    )
    columns[2].metric(
        "Konvergen",
        "Ya" if bool(row.converged) else "Tidak",
        help="Status kriteria penghentian solver, bukan ukuran kinerja prediksi.",
    )
    columns[3].metric(
        "Iterasi solver",
        format_count(row.n_iter_max),
        help=f"Batas iterasi {format_count(int(row.max_iter))}; toleransi {decimal(row.tol)}.",
    )
    # Panel counts use the full saved top-k, not the filtered gene-table view.
    panel_nonzero = int(genes["selection_score"].gt(1e-12).sum())
    panel_zero = count - panel_nonzero
    st.caption(
        f"Panel top-{format_count(count)}: {format_count(panel_nonzero)} gen nonzero dan {format_count(panel_zero)} gen berskor nol. "
        "Tidak mengikuti filter pencarian."
    )


def render_diagnostic_details(tables: dict, selector: str, filtered: pd.DataFrame, fit: pd.Series | None) -> None:
    """Show the detailed selector diagnostics for one CV fit."""
    diagnostics = tables.get("selector_diagnostics", {})
    if fit is None or not bool(fit.uses_feature_selection) or not diagnostics:
        return
    fold, count = int(fit.fold), int(fit.genes)
    with st.expander("Detail diagnostik metode"):
        # Details for ANOVA and mutual information.
        if selector in {"anova", "mutual_information"}:
            saved = diagnostics["anova_genes" if selector == "anova" else "mi_genes"]
            detail = filtered[["gene"]].merge(
                saved.loc[saved.fold.eq(fold)], on="gene", how="left", validate="one_to_one"
            )
            if selector == "anova":
                st.caption(
                    "Nilai F dan η² menggambarkan pemisahan jaringan. P-value/BH ANOVA, Brown–Forsythe, dan Welch tidak memodelkan korelasi sampel dalam donor. Welch melonggarkan asumsi variansi, bukan independensi donor."
                )
                fields = [
                    "gene",
                    "f_statistic",
                    "anova_p_value_naive",
                    "anova_fdr_bh_naive",
                    "eta_squared",
                    "brown_forsythe_fdr_bh_naive",
                    "variance_ratio_max_min_positive",
                    "has_zero_within_tissue_variance",
                    "residual_skewness",
                    "residual_excess_kurtosis",
                    "welch_f_statistic",
                    "welch_rank",
                ]
                saved_summary = diagnostics["anova_summary"]
                row = saved_summary.loc[saved_summary.fold.eq(fold) & saved_summary.top_k.eq(count)].iloc[0]
                st.caption(
                    f"Residual pada top-k: nilai absolut skewness > 2 = {_percent(row.fraction_abs_skew_gt_2)}, excess kurtosis > 7 = {_percent(row.fraction_excess_kurtosis_gt_7)}. Flag ini deskriptif, bukan bukti normalitas atau batas universal."
                )
                st.caption(
                    f"Subset ANOVA dan Welch pada data training fold yang sama memiliki {format_count(int(row.anova_welch_selected_overlap))} gen bersama dari masing-masing {format_count(count)} gen. Ini adalah pemeriksaan sensitivitas terhadap variansi, bukan perbandingan fold."
                )
            else:
                st.caption(
                    "Ranking utama memakai n_neighbors = 3. Nilai 5 dan 10 hanya pemeriksaan sensitivitas. Skor dalam nat, bukan probabilitas atau p-value; skor nol tidak membuktikan tidak adanya hubungan biologis."
                )
                fields = [
                    "gene",
                    "mi_n3",
                    "mi_n5",
                    "mi_n10",
                    "rank_n3",
                    "rank_n5",
                    "rank_n10",
                    "abs_rank_change_3_vs_5",
                    "abs_rank_change_3_vs_10",
                ]
            st.dataframe(
                detail[fields].style.format({"eta_squared": _percent} if selector == "anova" else {}),
                hide_index=True,
                width="stretch",
                column_config={
                    "gene": "ID gen Ensembl",
                    "f_statistic": "Statistik F",
                    "eta_squared": "η² (%)",
                    "anova_p_value_naive": st.column_config.NumberColumn("p ANOVA (naif)", format="%.3e"),
                    "anova_fdr_bh_naive": st.column_config.NumberColumn("p BH ANOVA (naif)", format="%.3e"),
                    "brown_forsythe_fdr_bh_naive": st.column_config.NumberColumn(
                        "p BH Brown–Forsythe (naif)", format="%.3e"
                    ),
                    "variance_ratio_max_min_positive": "Rasio variansi positif maks/min",
                    "has_zero_within_tissue_variance": "Ada variansi dalam jaringan nol",
                    "residual_skewness": "Skewness residual",
                    "residual_excess_kurtosis": "Excess kurtosis residual",
                    "welch_f_statistic": "Statistik Welch",
                    "welch_rank": "Peringkat Welch",
                    **{f"mi_n{n}": f"MI n={n} (nat)" for n in [3, 5, 10]},
                    **{f"rank_n{n}": f"Peringkat n={n}" for n in [3, 5, 10]},
                    "abs_rank_change_3_vs_5": "Perubahan absolut peringkat (3 vs 5)",
                    "abs_rank_change_3_vs_10": "Perubahan absolut peringkat (3 vs 10)",
                },
            )
            export = detail.assign(selector_id=selector, genes=count, panel_source="cv_training")
            st.download_button(
                "Unduh diagnostik gen terfilter",
                export.to_csv(index=False).encode("utf-8"),
                file_name=f"diagnostik_{selector}_{count}_fold_{fold}.csv",
                mime="text/csv",
            )

        # Extra details for mutual information.
        if selector == "mutual_information":
            neighbors = diagnostics["mi_neighbors"]
            neighbors = neighbors.loc[neighbors.fold.eq(fold) & neighbors.top_k.eq(count)]
            st.markdown("**Sensitivitas parameter pada fold yang sama**")
            st.caption(
                "Spearman memakai seluruh kandidat; overlap dan Jaccard memakai subset top-k pada masing-masing n_neighbors. Ini berbeda dari tab Stabilitas antar-fold."
            )
            st.dataframe(
                neighbors[
                    ["fold", "neighbors_a", "neighbors_b", "spearman_all_genes", "selected_overlap", "jaccard"]
                ].style.format({"jaccard": _percent}),
                hide_index=True,
                width="stretch",
                column_config={
                    "fold": "Fold",
                    "neighbors_a": "Neighbors A",
                    "neighbors_b": "Neighbors B",
                    "spearman_all_genes": "Spearman seluruh kandidat",
                    "selected_overlap": "Gen bersama",
                    "jaccard": "Jaccard antar-neighbors (%)",
                },
            )

        # Details for L1.
        if selector.startswith("l1_"):
            count_label = format_count(count)
            st.caption(
                f"Cutoff koefisien absolut top-k: {_number(fit.coefficient_cutoff, 6)}. Gen nonzero dinilai dengan toleransi 10⁻¹². Konfigurasi utama memerlukan konvergensi dan sedikitnya {count_label} gen nonzero pada setiap fold."
            )
            ties_label = format_count(fit.tied_at_cutoff_count)
            st.caption(
                f"Skor seri pada cutoff (seluruh kandidat): {ties_label}. Urutan seri mengikuti notebook; koefisien L1 bukan importance internal XGBoost."
            )
            comparisons = (
                diagnostics["l1_fits"]
                .loc[diagnostics["l1_fits"].fold.eq(fold)]
                .sort_values("l1_c", ascending=False)
                .copy()
            )
            comparisons["gene_sparsity"] = 1 - comparisons.nonzero_gene_count / comparisons.n_features
            comparisons["top_k_supported"] = comparisons[f"eligible_k_{count}"]
            st.markdown("**Pengaruh C pada data training fold yang sama**")
            st.dataframe(
                comparisons[
                    [
                        "fold",
                        "l1_c",
                        "nonzero_gene_count",
                        "gene_sparsity",
                        "converged",
                        "n_iter_max",
                        "top_k_supported",
                    ]
                ].style.format({"gene_sparsity": _percent}),
                hide_index=True,
                width="stretch",
                column_config={
                    "fold": "Fold",
                    "l1_c": "C",
                    "nonzero_gene_count": "Gen nonzero",
                    "gene_sparsity": "Sparsitas gen (%)",
                    "converged": "Konvergen",
                    "n_iter_max": "Iterasi",
                    "top_k_supported": f"Mendukung top-{count_label}",
                },
            )


def render_stability_details(tables: dict, selector: str, fit: pd.Series | None) -> None:
    """All fold pairs for this configuration, independent of the gene-table filters."""
    if fit is None or not bool(fit.uses_feature_selection):
        return
    diagnostics = tables.get("selector_diagnostics", {})
    if not diagnostics:
        st.info("Stabilitas antar-fold belum tersedia pada ekspor ini.")
        return
    fold, count = int(fit.fold), int(fit.genes)
    pairs = diagnostics["stability_pairs"]
    pairs = pairs.loc[pairs.selector_id.eq(selector) & pairs.genes.eq(count)].sort_values(["fold_a", "fold_b"])
    labels = (
        tables["cv_summary"].loc[tables["cv_summary"]["selector_id"].eq(selector), "selector_label"].dropna().unique()
    )
    selector_label = str(labels[0]) if len(labels) == 1 else selector
    st.subheader("Stabilitas antar-fold")
    st.caption(
        f"{selector_label} - {format_count(count)} gen - {tables['run_summary']['cv_folds']} fold - {len(pairs)} pasangan unik"
    )
    st.caption(f"Ringkasan seluruh fold, bukan hanya fold {fold}, dan tidak mengikuti filter gen.")
    columns = st.columns(2)
    columns[0].metric("Rerata Jaccard", _percent(pairs.jaccard.mean()))
    columns[1].metric("Minimum Jaccard", _percent(pairs.jaccard.min()))
    st.caption(
        "Jaccard = jumlah gen pada irisan ÷ jumlah gen pada gabungan. Nilai tinggi menunjukkan panel lebih serupa, bukan akurasi atau bukti kesesuaian biologis. Bandingkan metode pada top-k yang sama."
    )
    if not bool(fit.eligible_for_primary_comparison):
        st.caption(
            "Konfigurasi ini tidak memenuhi syarat perbandingan utama. Gen berskor nol dan aturan pemecahan skor seri dapat memengaruhi overlap top-k."
        )
    with st.expander("Pasangan fold dan overlap terkoreksi"):
        st.dataframe(
            pairs[
                [
                    "fold_a",
                    "fold_b",
                    "selected_overlap",
                    "jaccard",
                    "expected_random_overlap",
                    "chance_adjusted_overlap",
                ]
            ].style.format({"jaccard": _percent}),
            hide_index=True,
            width="stretch",
            column_config={
                "fold_a": "Fold A",
                "fold_b": "Fold B",
                "selected_overlap": "Gen bersama",
                "jaccard": "Jaccard (%)",
                "expected_random_overlap": "Overlap harapan acak",
                "chance_adjusted_overlap": "Overlap terkoreksi peluang",
            },
        )
        st.caption(
            "Overlap terkoreksi memperhitungkan ukuran dan irisan ruang kandidat dengan asumsi pemilihan acak seragam. Angka negatif berarti overlap di bawah harapan acak; bukan Jaccard."
        )
