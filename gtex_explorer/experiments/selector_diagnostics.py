"""Read saved selector diagnostics and display their training-data scope."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


DIAGNOSTIC_FILES = {
    "anova_genes": "anova_gene_diagnostics_by_fold.csv.gz",
    "anova_summary": "anova_diagnostics_summary.csv",
    "mi_genes": "mutual_information_gene_diagnostics.csv.gz",
    "mi_neighbors": "mutual_information_neighbor_stability.csv",
    "l1_fits": "l1_fit_diagnostics.csv",
    "stability_pairs": "feature_selection_stability_pairs.csv",
}


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_selector_diagnostics(data_dir: Path, summary: dict) -> dict:
    manifest_path = data_dir / "selector_diagnostics_manifest.json"
    if not manifest_path.exists():
        if any((data_dir / name).exists() for name in DIAGNOSTIC_FILES.values()):
            raise ValueError("Ekspor diagnostik selektor tidak lengkap: manifest hilang.")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("experiment_fingerprint") != summary["experiment_fingerprint"]:
        raise ValueError("Diagnostik selektor berasal dari eksperimen berbeda.")
    if set(manifest["files"]) != set(DIAGNOSTIC_FILES.values()):
        raise ValueError("Daftar berkas diagnostik selektor tidak cocok.")
    for filename, expected_hash in manifest["bindings"].items():
        if filename not in {"experiment_cv_folds.csv", "cv_gene_rankings.csv.gz", "final_selector_fit_diagnostics.csv"}:
            raise ValueError("Binding diagnostik selektor tidak dikenal.")
        if not (data_dir / filename).is_file() or file_hash(data_dir / filename) != expected_hash:
            raise ValueError("Diagnostik tidak cocok dengan hasil fold/panel aplikasi.")
    if set(manifest["bindings"]) != {"experiment_cv_folds.csv", "cv_gene_rankings.csv.gz", "final_selector_fit_diagnostics.csv"}:
        raise ValueError("Binding diagnostik selektor tidak lengkap.")
    result = {}
    for key, filename in DIAGNOSTIC_FILES.items():
        path = data_dir / filename
        if not path.is_file() or file_hash(path) != manifest["files"][filename]["sha256"]:
            raise ValueError(f"Hash/berkas diagnostik tidak cocok: {filename}")
        result[key] = pd.read_csv(path)
        if len(result[key]) != manifest["files"][filename]["rows"]:
            raise ValueError(f"Jumlah baris diagnostik tidak cocok: {filename}")
    return result


def validate_diagnostic_tables(diagnostics: dict, fits: pd.DataFrame, rankings: pd.DataFrame) -> None:
    """Check exports against actual rankings/fit rows before packaging them."""
    anova_fits = fits.loc[fits.selector_id.eq("anova")]
    anova_summary = diagnostics["anova_summary"]
    if anova_summary.duplicated(["fold", "top_k"]).any() or set(zip(anova_summary.fold, anova_summary.top_k)) != set(zip(anova_fits.fold, anova_fits.genes)):
        raise ValueError("Ringkasan ANOVA tidak lengkap atau duplikat.")
    mi_fits = fits.loc[fits.selector_id.eq("mutual_information")]
    mi_neighbors = diagnostics["mi_neighbors"]
    expected_neighbors = {(fold, count, a, b) for fold, count in zip(mi_fits.fold, mi_fits.genes) for a, b in itertools.combinations([3, 5, 10], 2)}
    neighbor_keys = ["fold", "top_k", "neighbors_a", "neighbors_b"]
    if mi_neighbors.duplicated(neighbor_keys).any() or set(mi_neighbors[neighbor_keys].itertuples(index=False, name=None)) != expected_neighbors:
        raise ValueError("Diagnostik neighbor MI tidak lengkap atau duplikat.")
    l1_fits = diagnostics["l1_fits"]
    if l1_fits.duplicated(["selector_id", "fold"]).any() or set(zip(l1_fits.selector_id, l1_fits.fold)) != set(zip(fits.loc[fits.feature_selection.eq("l1"), "selector_id"], fits.loc[fits.feature_selection.eq("l1"), "fold"])):
        raise ValueError("Diagnostik L1 tidak lengkap atau duplikat.")
    for key, selector, rank_column, score_column in [
        ("anova_genes", "anova", "anova_rank", "f_statistic"),
        ("mi_genes", "mutual_information", "rank_n3", "mi_n3"),
    ]:
        saved = diagnostics[key]
        reference = rankings.loc[rankings.selector_id.eq(selector)]
        if saved.duplicated(["fold", "gene"]).any() or len(saved) != len(reference):
            raise ValueError(f"Cakupan gen diagnostik tidak cocok: {selector}")
        joined = reference.merge(saved, on=["fold", "gene"], how="left", validate="one_to_one")
        if not np.array_equal(joined.selection_rank, joined[rank_column]) or not np.allclose(joined.selection_score, joined[score_column], rtol=1e-6, atol=1e-10):
            raise ValueError(f"Peringkat/skor diagnostik tidak cocok: {selector}")
    for _, fit in fits.loc[fits.feature_selection.eq("l1")].iterrows():
        selected = diagnostics["l1_fits"].loc[
            diagnostics["l1_fits"].selector_id.eq(fit.selector_id)
            & diagnostics["l1_fits"].fold.eq(fit.fold)
        ]
        if len(selected) != 1:
            raise ValueError("Diagnostik L1 hilang atau duplikat.")
        row = selected.iloc[0]
        if row.nonzero_gene_count != fit.nonzero_gene_count or bool(row.converged) != bool(fit.selector_converged):
            raise ValueError("Diagnostik L1 tidak cocok dengan hasil fit.")
    expected_keys = set(zip(fits.loc[fits.uses_feature_selection, "selector_id"], fits.loc[fits.uses_feature_selection, "genes"]))
    pairs = diagnostics["stability_pairs"]
    if set(zip(pairs.selector_id, pairs.genes)) != expected_keys:
        raise ValueError("Cakupan konfigurasi Jaccard tidak cocok.")
    for selector, count in expected_keys:
        subset = pairs.loc[pairs.selector_id.eq(selector) & pairs.genes.eq(count)]
        folds = sorted(fits.loc[fits.selector_id.eq(selector) & fits.genes.eq(count), "fold"].unique())
        if len(subset) != len(list(itertools.combinations(folds, 2))) or set(zip(subset.fold_a, subset.fold_b)) != set(itertools.combinations(folds, 2)):
            raise ValueError("Pasangan fold Jaccard tidak lengkap atau duplikat.")
        sets = {fold: set(rankings.loc[rankings.selector_id.eq(selector) & rankings.fold.eq(fold) & rankings.selection_rank.le(count), "gene"]) for fold in folds}
        for row in subset.itertuples():
            a, b = sets[row.fold_a], sets[row.fold_b]
            if row.selected_overlap != len(a & b) or not np.isclose(row.jaccard, len(a & b) / len(a | b)):
                raise ValueError("Jaccard tidak cocok dengan panel gen tersimpan.")
    for row in diagnostics["anova_summary"].itertuples():
        selected = diagnostics["anova_genes"].loc[diagnostics["anova_genes"].fold.eq(row.fold) & diagnostics["anova_genes"].anova_rank.le(row.top_k)]
        if len(selected) != row.top_k or not np.isclose(selected.eta_squared.median(), row.median_eta_squared):
            raise ValueError("Ringkasan ANOVA tidak cocok dengan subset top-k.")
        for flag, summary_field in [
            ("flag_bf_fdr_005", "fraction_bf_fdr_005"),
            ("flag_variance_ratio_ge_4", "fraction_variance_ratio_ge_4"),
            ("has_zero_within_tissue_variance", "fraction_has_zero_within_tissue_variance"),
            ("flag_abs_skew_gt_2", "fraction_abs_skew_gt_2"),
            ("flag_excess_kurtosis_gt_7", "fraction_excess_kurtosis_gt_7"),
        ]:
            if not np.isclose(selected[flag].mean(), getattr(row, summary_field)):
                raise ValueError("Flag ANOVA tidak cocok dengan subset top-k.")
    for row in mi_neighbors.itertuples():
        candidate_genes = diagnostics["mi_genes"].loc[diagnostics["mi_genes"].fold.eq(row.fold)]
        a = set(candidate_genes.loc[candidate_genes[f"rank_n{row.neighbors_a}"].le(row.top_k), "gene"])
        b = set(candidate_genes.loc[candidate_genes[f"rank_n{row.neighbors_b}"].le(row.top_k), "gene"])
        rho = candidate_genes[f"mi_n{row.neighbors_a}"].rank().corr(candidate_genes[f"mi_n{row.neighbors_b}"].rank())
        if row.selected_overlap != len(a & b) or not np.isclose(row.jaccard, len(a & b) / len(a | b)) or not np.isclose(row.spearman_all_genes, rho, equal_nan=True):
            raise ValueError("Sensitivitas MI tidak cocok dengan ranking tersimpan.")


def _number(value, digits=4) -> str:
    return "Tidak tersedia" if pd.isna(value) else f"{float(value):.{digits}f}".replace(".", ",")


def _percent(value) -> str:
    return "Tidak tersedia" if pd.isna(value) else f"{float(value):.2%}".replace(".", ",")


def render_selector_metrics(tables: dict, selector: str, genes: pd.DataFrame, fit: pd.Series | None) -> None:
    """Training-data selector metrics, separate from XGBoost validation scores."""
    diagnostics = tables.get("selector_diagnostics", {})
    if fit is not None and not bool(fit.uses_feature_selection):
        return
    st.subheader("Diagnostik seleksi fitur")
    count = len(genes)
    if fit is None:
        saved = tables["selector_fit"].loc[tables["selector_fit"].selector_id.eq(selector)]
        if len(saved) != 1 or not selector.startswith("l1_"):
            st.info("Diagnostik tambahan belum tersedia untuk selector final ini.")
            return
        row = saved.iloc[0]
        n_features = int(tables["run_summary"]["variance_prefilter_genes"])
        nonzero = int(row.positive_or_nonzero_gene_count)
        st.caption(f"Data development: {int(row.training_samples):,} sampel - {int(row.training_donors):,} donor.".replace(",", "."))
    elif not diagnostics:
        st.info("Diagnostik tambahan belum tersedia. Ekspor hasil notebook dengan export_selector_diagnostics.py.")
        return
    else:
        candidates = int(tables["run_summary"]["variance_prefilter_genes"])
        if selector in {"anova", "mutual_information"}:
            st.caption(f"Data training fold {int(fit.fold)} - panel top-{count:,} dari {candidates:,} gen kandidat.".replace(",", "."))
        if selector == "anova":
            st.caption(f"η² dan flag variansi: gen top-k. Spearman: seluruh {candidates:,} gen kandidat. Tidak mengikuti filter pencarian.".replace(",", "."))
            saved = diagnostics["anova_summary"]
            row = saved.loc[saved.fold.eq(fit.fold) & saved.top_k.eq(count)].iloc[0]
            columns = st.columns(4)
            columns[0].metric("Median η²", _percent(row.median_eta_squared), help="Proporsi variasi ekspresi terkait perbedaan rerata jaringan, diringkas pada gen top-k; bukan akurasi.")
            columns[1].metric("Variansi berbeda (BF)", _percent(row.fraction_bf_fdr_005), help="Persentase gen top-k dengan adjusted p-value Brown–Forsythe < 0,05. Bersifat eksploratif, bukan uji kelayakan lulus/gagal.")
            columns[2].metric("Rasio variansi ≥ 4", _percent(row.fraction_variance_ratio_ge_4), help="Persentase gen top-k yang melewati flag deskriptif notebook; variansi nol dicatat terpisah.")
            columns[3].metric("Spearman ANOVA–Welch", _number(row.anova_welch_spearman_all_genes), help="Kesamaan urutan seluruh gen kandidat saat asumsi kesamaan variansi dilonggarkan. Koefisien −1 sampai 1.")
            return
        if selector == "mutual_information":
            st.caption(f"Gen berskor positif dan Spearman: seluruh {candidates:,} gen kandidat. Cutoff: gen top-k. Tidak mengikuti filter pencarian.".replace(",", "."))
            columns = st.columns(4)
            columns[0].metric("Gen berskor MI positif", f"{int(fit.positive_score_gene_count):,} / {int(tables['run_summary']['variance_prefilter_genes']):,}".replace(",", "."), help="Jumlah gen kandidat berskor positif pada n_neighbors utama, bukan jumlah gen yang terbukti signifikan.")
            columns[1].metric("Cutoff MI top-k (nat)", _number(fit.score_cutoff, 6))
            saved = diagnostics["mi_neighbors"]
            for index, neighbor in enumerate([5, 10], start=2):
                row = saved.loc[saved.fold.eq(fit.fold) & saved.top_k.eq(count) & saved.neighbors_a.eq(3) & saved.neighbors_b.eq(neighbor)].iloc[0]
                columns[index].metric(f"Spearman 3 vs {neighbor}", _number(row.spearman_all_genes), help="Sensitivitas ranking seluruh gen kandidat terhadap n_neighbors; bukan perbandingan fold atau skor klasifikasi.")
            return
        saved = diagnostics["l1_fits"]
        row = saved.loc[saved.selector_id.eq(selector) & saved.fold.eq(fit.fold)].iloc[0]
        n_features, nonzero = int(row.n_features), int(row.nonzero_gene_count)
    scope = f"Data training fold {int(fit.fold)} - " if fit is not None else ""
    st.caption(f"{scope}{n_features:,} gen kandidat sebelum pemilihan top-k.".replace(",", "."))
    columns = st.columns(4)
    columns[0].metric("Gen nonzero", f"{nonzero:,} / {n_features:,}".replace(",", "."), help="Jumlah gen nonzero di antara seluruh kandidat sebelum pemilihan top-k. Koefisien absolut maksimum lintas kelas > 10⁻¹²; satu gen dihitung satu kali.")
    columns[1].metric("Sparsitas gen", _percent(1 - nonzero / n_features), help="Proporsi kandidat gen tanpa koefisien nonzero di kelas mana pun; komplemen proporsi gen nonzero.")
    columns[2].metric("Konvergen", "Ya" if bool(row.converged) else "Tidak", help="Status kriteria penghentian solver, bukan ukuran kinerja prediksi.")
    columns[3].metric("Iterasi solver", f"{int(row.n_iter_max):,}".replace(",", "."), help=f"Batas iterasi {int(row.max_iter):,}; toleransi {row.tol:g}.".replace(",", "."))
    # Panel counts use the full saved top-k, not the filtered gene-table view.
    panel_nonzero = int(genes["selection_score"].gt(1e-12).sum())
    panel_zero = count - panel_nonzero
    st.caption(
        f"Panel top-{count:,}: {panel_nonzero:,} gen nonzero dan {panel_zero:,} gen berskor nol. "
        "Tidak mengikuti filter pencarian.".replace(",", ".")
    )


def render_diagnostic_details(tables: dict, selector: str, filtered: pd.DataFrame, fit: pd.Series | None) -> None:
    diagnostics = tables.get("selector_diagnostics", {})
    if fit is None or not bool(fit.uses_feature_selection) or not diagnostics:
        return
    fold, count = int(fit.fold), int(fit.genes)
    with st.expander("Detail diagnostik metode"):
        if selector in {"anova", "mutual_information"}:
            saved = diagnostics["anova_genes" if selector == "anova" else "mi_genes"]
            detail = filtered[["gene"]].merge(saved.loc[saved.fold.eq(fold)], on="gene", how="left", validate="one_to_one")
            if selector == "anova":
                st.caption("Nilai F dan η² menggambarkan pemisahan jaringan. P-value/BH ANOVA, Brown–Forsythe, dan Welch tidak memodelkan korelasi sampel dalam donor. Welch melonggarkan asumsi variansi, bukan independensi donor.")
                fields = ["gene", "f_statistic", "anova_p_value_naive", "anova_fdr_bh_naive", "eta_squared", "brown_forsythe_fdr_bh_naive", "variance_ratio_max_min_positive", "has_zero_within_tissue_variance", "residual_skewness", "residual_excess_kurtosis", "welch_f_statistic", "welch_rank"]
                saved_summary = diagnostics["anova_summary"]
                row = saved_summary.loc[saved_summary.fold.eq(fold) & saved_summary.top_k.eq(count)].iloc[0]
                st.caption(f"Residual pada top-k: nilai absolut skewness > 2 = {_percent(row.fraction_abs_skew_gt_2)}, excess kurtosis > 7 = {_percent(row.fraction_excess_kurtosis_gt_7)}. Flag ini deskriptif, bukan bukti normalitas atau batas universal.")
                st.caption(f"Subset ANOVA dan Welch pada data training fold yang sama memiliki {int(row.anova_welch_selected_overlap):,} gen bersama dari masing-masing {count:,} gen. Ini adalah pemeriksaan sensitivitas terhadap variansi, bukan perbandingan fold.".replace(",", "."))
            else:
                st.caption("Ranking utama memakai n_neighbors = 3. Nilai 5 dan 10 hanya pemeriksaan sensitivitas. Skor dalam nat, bukan probabilitas atau p-value; skor nol tidak membuktikan tidak adanya hubungan biologis.")
                fields = ["gene", "mi_n3", "mi_n5", "mi_n10", "rank_n3", "rank_n5", "rank_n10", "abs_rank_change_3_vs_5", "abs_rank_change_3_vs_10"]
            st.dataframe(detail[fields].style.format({"eta_squared": _percent} if selector == "anova" else {}), hide_index=True, width="stretch", column_config={
                "gene": "ID gen Ensembl", "f_statistic": "Statistik F", "eta_squared": "η² (%)",
                "anova_p_value_naive": st.column_config.NumberColumn("p ANOVA (naif)", format="%.3e"),
                "anova_fdr_bh_naive": st.column_config.NumberColumn("p BH ANOVA (naif)", format="%.3e"),
                "brown_forsythe_fdr_bh_naive": st.column_config.NumberColumn("p BH Brown–Forsythe (naif)", format="%.3e"),
                "variance_ratio_max_min_positive": "Rasio variansi positif maks/min",
                "has_zero_within_tissue_variance": "Ada variansi dalam jaringan nol",
                "residual_skewness": "Skewness residual", "residual_excess_kurtosis": "Excess kurtosis residual",
                "welch_f_statistic": "Statistik Welch", "welch_rank": "Peringkat Welch",
                **{f"mi_n{n}": f"MI n={n} (nat)" for n in [3, 5, 10]},
                **{f"rank_n{n}": f"Peringkat n={n}" for n in [3, 5, 10]},
                "abs_rank_change_3_vs_5": "Perubahan absolut peringkat (3 vs 5)",
                "abs_rank_change_3_vs_10": "Perubahan absolut peringkat (3 vs 10)",
            })
            export = detail.assign(selector_id=selector, genes=count, panel_source="cv_training")
            st.download_button("Unduh diagnostik gen terfilter", export.to_csv(index=False).encode("utf-8"), file_name=f"diagnostik_{selector}_{count}_fold_{fold}.csv", mime="text/csv")
        if selector == "mutual_information":
            neighbors = diagnostics["mi_neighbors"]
            neighbors = neighbors.loc[neighbors.fold.eq(fold) & neighbors.top_k.eq(count)]
            st.markdown("**Sensitivitas parameter pada fold yang sama**")
            st.caption("Spearman memakai seluruh kandidat; overlap dan Jaccard memakai subset top-k pada masing-masing n_neighbors. Ini berbeda dari tab Stabilitas antar-fold.")
            st.dataframe(neighbors[["fold", "neighbors_a", "neighbors_b", "spearman_all_genes", "selected_overlap", "jaccard"]].style.format({"jaccard": _percent}), hide_index=True, width="stretch", column_config={"fold": "Fold", "neighbors_a": "Neighbors A", "neighbors_b": "Neighbors B", "spearman_all_genes": "Spearman seluruh kandidat", "selected_overlap": "Gen bersama", "jaccard": "Jaccard antar-neighbors (%)"})
        if selector.startswith("l1_"):
            count_label = f"{count:,}".replace(",", ".")
            st.caption(f"Cutoff koefisien absolut top-k: {_number(fit.coefficient_cutoff, 6)}. Gen nonzero dinilai dengan toleransi 10⁻¹². Konfigurasi utama memerlukan konvergensi dan sedikitnya {count_label} gen nonzero pada setiap fold.")
            ties_label = f"{int(fit.tied_at_cutoff_count):,}".replace(",", ".")
            st.caption(f"Skor seri pada cutoff (seluruh kandidat): {ties_label}. Urutan seri mengikuti notebook; koefisien L1 bukan importance internal XGBoost.")
            comparisons = diagnostics["l1_fits"].loc[diagnostics["l1_fits"].fold.eq(fold)].sort_values("l1_c", ascending=False).copy()
            comparisons["gene_sparsity"] = 1 - comparisons.nonzero_gene_count / comparisons.n_features
            comparisons["top_k_supported"] = comparisons[f"eligible_k_{count}"]
            st.markdown("**Pengaruh C pada data training fold yang sama**")
            st.dataframe(comparisons[["fold", "l1_c", "nonzero_gene_count", "gene_sparsity", "converged", "n_iter_max", "top_k_supported"]].style.format({"gene_sparsity": _percent}), hide_index=True, width="stretch", column_config={"fold": "Fold", "l1_c": "C", "nonzero_gene_count": "Gen nonzero", "gene_sparsity": "Sparsitas gen (%)", "converged": "Konvergen", "n_iter_max": "Iterasi", "top_k_supported": f"Mendukung top-{count_label}"})


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
    labels = tables["cv_summary"].loc[tables["cv_summary"]["selector_id"].eq(selector), "selector_label"].dropna().unique()
    selector_label = str(labels[0]) if len(labels) == 1 else selector
    st.subheader("Stabilitas antar-fold")
    st.caption(f"{selector_label} - {count:,} gen - {tables['run_summary']['cv_folds']} fold - {len(pairs)} pasangan unik".replace(",", "."))
    st.caption(f"Ringkasan seluruh fold, bukan hanya fold {fold}, dan tidak mengikuti filter gen.")
    columns = st.columns(2)
    columns[0].metric("Rerata Jaccard", _percent(pairs.jaccard.mean()))
    columns[1].metric("Minimum Jaccard", _percent(pairs.jaccard.min()))
    st.caption("Jaccard = jumlah gen pada irisan ÷ jumlah gen pada gabungan. Nilai tinggi menunjukkan panel lebih serupa, bukan akurasi atau bukti kesesuaian biologis. Bandingkan metode pada top-k yang sama.")
    if not bool(fit.eligible_for_primary_comparison):
        st.caption("Konfigurasi ini tidak memenuhi syarat perbandingan utama. Gen berskor nol dan aturan pemecahan skor seri dapat memengaruhi overlap top-k.")
    with st.expander("Pasangan fold dan overlap terkoreksi"):
        st.dataframe(pairs[["fold_a", "fold_b", "selected_overlap", "jaccard", "expected_random_overlap", "chance_adjusted_overlap"]].style.format({"jaccard": _percent}), hide_index=True, width="stretch", column_config={"fold_a": "Fold A", "fold_b": "Fold B", "selected_overlap": "Gen bersama", "jaccard": "Jaccard (%)", "expected_random_overlap": "Overlap harapan acak", "chance_adjusted_overlap": "Overlap terkoreksi peluang"})
        st.caption("Overlap terkoreksi memperhitungkan ukuran dan irisan ruang kandidat dengan asumsi pemilihan acak seragam. Angka negatif berarti overlap di bawah harapan acak; bukan Jaccard.")
