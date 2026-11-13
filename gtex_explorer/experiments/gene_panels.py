"""Read-only gene panels from the notebook's saved cross-validation rankings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

L1_INSUFFICIENT_GENES = "Gen berkoefisien nonzero kurang dari jumlah top-k pada satu atau lebih fold"


def clarify_eligibility_reasons(summary: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Clarify L1 display copy only when saved fold diagnostics establish the cause."""
    result = summary.copy()
    for index, row in result.iterrows():
        if row.get("feature_selection") != "l1":
            continue
        matching = fits.loc[fits["selector_id"].eq(row["selector_id"]) & fits["genes"].eq(row["genes"])]
        if matching.empty or not matching["nonzero_gene_count"].lt(row["genes"]).any():
            continue
        reasons = [part.strip() for part in str(row["eligibility_reason"]).split("|")]
        reasons = [
            part
            for part in reasons
            if part
            not in {
                "top-k mencakup skor nol atau tidak didukung",
                "flag eligibility fold gagal",
                "eligible",
            }
        ]
        result.loc[index, "eligibility_reason"] = " | ".join([L1_INSUFFICIENT_GENES, *reasons])
    return result


# The exports mix English and Indonesian reasons; these are shown in Indonesian.
REASON_TEXT = {
    "eligible": "Memenuhi syarat seleksi",
    "reference baseline, not a selector candidate": "Baseline pembanding, bukan kandidat seleksi",
    "flag eligibility fold gagal": "syarat seleksi tidak terpenuhi pada satu atau lebih fold",
}


def readable_eligibility_reason(text) -> str:
    """Write the saved "a | b" reasons as an Indonesian list."""
    return ", ".join(REASON_TEXT.get(part.strip(), part.strip()) for part in str(text).split("|"))


def load_cv_gene_rankings(data_dir: Path, summary: dict, fits: pd.DataFrame) -> pd.DataFrame:
    """Require a matching, complete export; never reconstruct rankings in the UI."""
    ranking_path = data_dir / "cv_gene_rankings.csv.gz"
    manifest_path = data_dir / "cv_gene_panel_manifest.json"
    if not ranking_path.exists() and not manifest_path.exists():
        return pd.DataFrame()
    if not ranking_path.is_file() or not manifest_path.is_file():
        raise ValueError("Ekspor panel gen CV tidak lengkap.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["experiment_fingerprint"] != summary["experiment_fingerprint"]:
        raise ValueError("Panel gen CV berasal dari eksperimen berbeda.")
    if hashlib.sha256(ranking_path.read_bytes()).hexdigest() != manifest["ranking_sha256"]:
        raise ValueError("Hash panel gen CV tidak cocok dengan manifest.")
    if hashlib.sha256((data_dir / "experiment_cv_folds.csv").read_bytes()).hexdigest() != manifest["cv_fits_sha256"]:
        raise ValueError("Hasil fit CV telah berubah. Ekspor ulang panel gen CV.")

    rankings = pd.read_csv(ranking_path)
    required = {"selector_id", "fold", "gene", "selection_rank", "selection_score"}
    if not required.issubset(rankings.columns):
        raise ValueError("Kolom panel gen CV tidak lengkap.")
    selector_fits = fits.loc[fits["uses_feature_selection"].eq(True)]
    expected = set(zip(selector_fits["selector_id"], selector_fits["fold"]))
    actual = set(zip(rankings["selector_id"], rankings["fold"]))
    if expected != actual or len(rankings) != manifest["ranking_rows"]:
        raise ValueError("Cakupan metode/fold panel gen CV tidak cocok.")
    if len(fits) != manifest["cv_fit_count"]:
        raise ValueError("Jumlah fit pada ekspor panel gen CV tidak cocok.")
    if rankings["gene"].isna().any() or rankings.duplicated(["selector_id", "fold", "gene"]).any():
        raise ValueError("Gen duplikat pada panel metode/fold CV.")
    candidate_count = int(summary["variance_prefilter_genes"])
    candidates_by_fold = {}
    for (selector, fold), group in rankings.groupby(["selector_id", "fold"]):
        if not np.array_equal(np.sort(group["selection_rank"]), np.arange(1, candidate_count + 1)):
            raise ValueError("Peringkat gen CV tidak lengkap atau duplikat.")
        candidates = set(group["gene"])
        if fold in candidates_by_fold and candidates != candidates_by_fold[fold]:
            raise ValueError("Metode memakai kandidat gen yang berbeda pada fold yang sama.")
        candidates_by_fold[fold] = candidates
    return rankings


def cv_gene_panel(rankings: pd.DataFrame, fit: pd.Series) -> pd.DataFrame:
    """Return exactly the top-k genes used in this fit, or the unranked baseline."""
    if bool(fit["uses_feature_selection"]):
        panel = (
            rankings.loc[
                rankings["selector_id"].eq(fit["selector_id"])
                & rankings["fold"].eq(fit["fold"])
                & rankings["selection_rank"].le(int(fit["genes"]))
            ]
            .sort_values("selection_rank")
            .copy()
        )
    else:
        # All methods share the same prefiltered candidates within a fold.
        # ANOVA is only the source of their IDs, not a selector for the baseline.
        panel = (
            rankings.loc[rankings["selector_id"].eq("anova") & rankings["fold"].eq(fit["fold"])]
            .sort_values("gene")
            .copy()
        )
        panel["selector_id"] = "none"
        for field in [
            "selection_rank",
            "selection_score",
            "dominant_tissue",
            "l2_coefficient_norm",
            "nonzero_class_count",
        ]:
            panel[field] = np.nan
    if len(panel) != int(fit["genes"]):
        raise ValueError("Jumlah gen panel tidak cocok dengan konfigurasi fit.")
    panel["configuration_id"] = fit["configuration_id"]
    panel["genes"] = int(fit["genes"])
    panel["classifier"] = fit["classifier"]
    panel["uses_feature_selection"] = bool(fit["uses_feature_selection"])
    panel["l1_c"] = fit.get("l1_c", np.nan)
    return panel.reset_index(drop=True)
