"""Read and validate the saved selector diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIAGNOSTIC_FILES = {
    "anova_genes": "anova_gene_diagnostics_by_fold.csv.gz",
    "anova_summary": "anova_diagnostics_summary.csv",
    "mi_genes": "mutual_information_gene_diagnostics.csv.gz",
    "mi_neighbors": "mutual_information_neighbor_stability.csv",
    "l1_fits": "l1_fit_diagnostics.csv",
    "stability_pairs": "feature_selection_stability_pairs.csv",
}


def file_hash(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_selector_diagnostics(data_dir: Path, summary: dict) -> dict:
    """Load the exported diagnostics after checking them against their manifest."""
    manifest_path = data_dir / "selector_diagnostics_manifest.json"
    if not manifest_path.exists():
        if any((data_dir / name).exists() for name in DIAGNOSTIC_FILES.values()):
            raise ValueError("Ekspor diagnostik selektor tidak lengkap: manifest hilang.")
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != 1
        or manifest.get("experiment_fingerprint") != summary["experiment_fingerprint"]
    ):
        raise ValueError("Diagnostik selektor berasal dari eksperimen berbeda.")
    if set(manifest["files"]) != set(DIAGNOSTIC_FILES.values()):
        raise ValueError("Daftar berkas diagnostik selektor tidak cocok.")
    for filename, expected_hash in manifest["bindings"].items():
        if filename not in {"experiment_cv_folds.csv", "cv_gene_rankings.csv.gz", "final_selector_fit_diagnostics.csv"}:
            raise ValueError("Binding diagnostik selektor tidak dikenal.")
        if not (data_dir / filename).is_file() or file_hash(data_dir / filename) != expected_hash:
            raise ValueError("Diagnostik tidak cocok dengan hasil fold/panel aplikasi.")
    if set(manifest["bindings"]) != {
        "experiment_cv_folds.csv",
        "cv_gene_rankings.csv.gz",
        "final_selector_fit_diagnostics.csv",
    }:
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
    if anova_summary.duplicated(["fold", "top_k"]).any() or set(zip(anova_summary.fold, anova_summary.top_k)) != set(
        zip(anova_fits.fold, anova_fits.genes)
    ):
        raise ValueError("Ringkasan ANOVA tidak lengkap atau duplikat.")
    mi_fits = fits.loc[fits.selector_id.eq("mutual_information")]
    mi_neighbors = diagnostics["mi_neighbors"]
    expected_neighbors = {
        (fold, count, a, b)
        for fold, count in zip(mi_fits.fold, mi_fits.genes)
        for a, b in itertools.combinations([3, 5, 10], 2)
    }
    neighbor_keys = ["fold", "top_k", "neighbors_a", "neighbors_b"]
    if (
        mi_neighbors.duplicated(neighbor_keys).any()
        or set(mi_neighbors[neighbor_keys].itertuples(index=False, name=None)) != expected_neighbors
    ):
        raise ValueError("Diagnostik neighbor MI tidak lengkap atau duplikat.")
    l1_fits = diagnostics["l1_fits"]
    if l1_fits.duplicated(["selector_id", "fold"]).any() or set(zip(l1_fits.selector_id, l1_fits.fold)) != set(
        zip(fits.loc[fits.feature_selection.eq("l1"), "selector_id"], fits.loc[fits.feature_selection.eq("l1"), "fold"])
    ):
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
        if not np.array_equal(joined.selection_rank, joined[rank_column]) or not np.allclose(
            joined.selection_score, joined[score_column], rtol=1e-6, atol=1e-10
        ):
            raise ValueError(f"Peringkat/skor diagnostik tidak cocok: {selector}")
    for _, fit in fits.loc[fits.feature_selection.eq("l1")].iterrows():
        selected = diagnostics["l1_fits"].loc[
            diagnostics["l1_fits"].selector_id.eq(fit.selector_id) & diagnostics["l1_fits"].fold.eq(fit.fold)
        ]
        if len(selected) != 1:
            raise ValueError("Diagnostik L1 hilang atau duplikat.")
        row = selected.iloc[0]
        if row.nonzero_gene_count != fit.nonzero_gene_count or bool(row.converged) != bool(fit.selector_converged):
            raise ValueError("Diagnostik L1 tidak cocok dengan hasil fit.")
    expected_keys = set(
        zip(fits.loc[fits.uses_feature_selection, "selector_id"], fits.loc[fits.uses_feature_selection, "genes"])
    )
    pairs = diagnostics["stability_pairs"]
    if set(zip(pairs.selector_id, pairs.genes)) != expected_keys:
        raise ValueError("Cakupan konfigurasi Jaccard tidak cocok.")
    for selector, count in expected_keys:
        subset = pairs.loc[pairs.selector_id.eq(selector) & pairs.genes.eq(count)]
        folds = sorted(fits.loc[fits.selector_id.eq(selector) & fits.genes.eq(count), "fold"].unique())
        if len(subset) != len(list(itertools.combinations(folds, 2))) or set(zip(subset.fold_a, subset.fold_b)) != set(
            itertools.combinations(folds, 2)
        ):
            raise ValueError("Pasangan fold Jaccard tidak lengkap atau duplikat.")
        sets = {
            fold: set(
                rankings.loc[
                    rankings.selector_id.eq(selector) & rankings.fold.eq(fold) & rankings.selection_rank.le(count),
                    "gene",
                ]
            )
            for fold in folds
        }
        for row in subset.itertuples():
            a, b = sets[row.fold_a], sets[row.fold_b]
            if row.selected_overlap != len(a & b) or not np.isclose(row.jaccard, len(a & b) / len(a | b)):
                raise ValueError("Jaccard tidak cocok dengan panel gen tersimpan.")
    for row in diagnostics["anova_summary"].itertuples():
        selected = diagnostics["anova_genes"].loc[
            diagnostics["anova_genes"].fold.eq(row.fold) & diagnostics["anova_genes"].anova_rank.le(row.top_k)
        ]
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
        if (
            row.selected_overlap != len(a & b)
            or not np.isclose(row.jaccard, len(a & b) / len(a | b))
            or not np.isclose(row.spearman_all_genes, rho, equal_nan=True)
        ):
            raise ValueError("Sensitivitas MI tidak cocok dengan ranking tersimpan.")
