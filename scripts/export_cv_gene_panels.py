"""Export existing rankings for all CV fits, without fitting any model.

Run from the app root: python -m scripts.export_cv_gene_panels
Only trusted, locally generated notebook checkpoints are read with joblib.
The desktop application reads the compact CSV export, never these checkpoints.
"""

from __future__ import annotations

import hashlib
import json

import joblib
import numpy as np
import pandas as pd

from gtex_explorer.experiments.gene_panels import cv_gene_panel, load_cv_gene_rankings


def export_panels() -> None:
    """Export the gene panel of every CV fold from the notebook results."""
    from gtex_explorer.platform.paths import APP_ROOT as app_dir

    # Read the exported summary and CV fits.
    data_dir = app_dir / "data"
    checkpoint_dir = app_dir.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_progress_v4"
    summary = json.loads((data_dir / "run_summary.json").read_text(encoding="utf-8"))
    fits = pd.read_csv(data_dir / "experiment_cv_folds.csv")
    selector_fits = fits.loc[fits["uses_feature_selection"].eq(True)]
    fingerprint = summary["experiment_fingerprint"]
    source_hashes = {}

    def read_checkpoint(filename: str, fingerprint_key: str = "experiment_fingerprint") -> dict:
        """Load a notebook checkpoint after checking that it belongs to this experiment."""
        path = checkpoint_dir / filename
        payload = joblib.load(path)
        if payload[fingerprint_key] != fingerprint:
            raise ValueError(f"Checkpoint from a different experiment: {filename}")
        with path.open("rb") as handle:
            source_hashes[filename.replace("\\", "/")] = hashlib.file_digest(handle, "sha256").hexdigest()
        return payload

    # Rankings and combined results use different documented hash formats.
    # Recompute both from the actual prepared fold instead of comparing them directly.
    fold_checkpoint = read_checkpoint("step_29_folds.joblib")
    fold_identity = {}
    fold_candidates = {}
    for prepared in fold_checkpoint["variables"]["prepared_cv_folds"]:
        fold = int(prepared["fold"])
        genes = list(map(str, prepared["gene_names"]))
        training_ids = list(map(str, prepared["train_sample_ids"]))
        ranking_hash = hashlib.sha256(
            ("\n".join(genes) + "|samples|" + "\n".join(training_ids)).encode("utf-8")
        ).hexdigest()
        combined_payload = {"fold": fold, "gene_names": genes}
        for field in ["train_sample_ids", "validation_sample_ids", "train_donors", "validation_donors"]:
            combined_payload[field] = list(map(str, prepared[field]))
        combined_hash = hashlib.sha256(
            json.dumps(combined_payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        fold_identity[fold] = (ranking_hash, combined_hash)
        fold_candidates[fold] = genes
    del prepared, fold_checkpoint

    # Collect the gene sets and scores saved by each selector step.
    saved_sets = {}
    saved_results = {}
    for filename in ["step_30_anova.joblib", "step_31_mutual_information.joblib", "step_32_l1.joblib"]:
        checkpoint = read_checkpoint(filename)
        if not checkpoint["complete"]:
            raise ValueError(f"Incomplete checkpoint: {filename}")
        for row in checkpoint["selected_gene_sets"]:
            key = (row["selector_id"], int(row["genes"]), int(row["fold"]))
            if key in saved_sets:
                raise ValueError(f"Duplicate checkpoint panel: {key}")
            saved_sets[key] = set(map(str, row["selected_genes"]))
        for row in checkpoint["selection_results"]:
            saved_results[(row["selector_id"], int(row["genes"]), int(row["fold"]))] = row

    # Rebuild the ranking of each selector in each fold.
    frames = []
    for selector, fold in sorted(set(zip(selector_fits["selector_id"], selector_fits["fold"]))):
        prefix = "mi" if selector == "mutual_information" else selector
        checkpoint = read_checkpoint(f"rankings_v4/{prefix}_fold_{int(fold)}.joblib")
        fold_fits = selector_fits.loc[selector_fits["selector_id"].eq(selector) & selector_fits["fold"].eq(fold)]
        ranking_hash, combined_hash = fold_identity[int(fold)]
        if checkpoint["fold_identity"] != ranking_hash or not fold_fits["fold_identity_sha256"].eq(combined_hash).all():
            raise ValueError(f"Different fold identity: {selector}/{fold}")
        diagnostics = checkpoint["gene_diagnostics"]
        if list(diagnostics["gene"].astype(str)) != fold_candidates[int(fold)]:
            raise ValueError(f"Different fold candidates: {selector}/{fold}")
        if selector == "mutual_information":
            neighbors = summary["mi_primary_n_neighbors"]
            order = checkpoint["order_by_neighbors"][neighbors]
            scores = checkpoint["score_by_neighbors"][neighbors]
            rank_column = f"rank_n{neighbors}"
        else:
            order, scores = checkpoint["ranked_indices"], checkpoint["scores"]
            rank_column = "anova_rank" if selector == "anova" else "l1_rank"
        ordered = diagnostics.iloc[order].reset_index(drop=True)
        expected_ranks = np.arange(1, len(ordered) + 1)
        if not np.array_equal(ordered[rank_column], expected_ranks):
            raise ValueError(f"Ranking mismatch: {selector}/{fold}")
        frame = pd.DataFrame(
            {
                "selector_id": selector,
                "fold": int(fold),
                "gene": ordered["gene"].astype(str),
                "selection_rank": expected_ranks,
                "selection_score": np.asarray(scores)[order],
            }
        )
        for column in ["dominant_tissue", "l2_coefficient_norm", "nonzero_class_count"]:
            if column == "dominant_tissue":
                frame[column] = (
                    ordered[column].astype(object)
                    if column in ordered
                    else pd.Series(None, index=frame.index, dtype=object)
                )
            else:
                frame[column] = ordered[column].astype(float) if column in ordered else np.nan
        frames.append(frame)
    rankings = pd.concat(frames, ignore_index=True)

    # Check the rankings against every saved fit.
    expected_keys = set(zip(selector_fits["selector_id"], selector_fits["genes"], selector_fits["fold"]))
    if expected_keys != set(saved_sets) or expected_keys != set(saved_results):
        raise ValueError("Checkpoint fits do not cover all selector configurations.")
    for _, fit in selector_fits.iterrows():
        key = (fit["selector_id"], int(fit["genes"]), int(fit["fold"]))
        if set(cv_gene_panel(rankings, fit)["gene"]) != saved_sets[key]:
            raise ValueError(f"Panel does not match actual selected genes: {key}")
        for metric in ["macro_f1", "macro_precision", "macro_recall", "balanced_accuracy", "accuracy"]:
            if not np.isclose(fit[metric], saved_results[key][metric], atol=1e-12, rtol=0):
                raise ValueError(f"CV result mismatch: {key}/{metric}")

    # Check the baseline the same way.
    baseline = read_checkpoint("step_32b_xgboost_all_5000.joblib", "selector_experiment_fingerprint")
    if baseline["baseline_fingerprint"] != summary["baseline_fingerprint"]:
        raise ValueError("Different baseline experiment.")
    baseline_fits = fits.loc[fits["uses_feature_selection"].eq(False)]
    if len(baseline_fits) != len(baseline["fold_results"]):
        raise ValueError("Incomplete baseline fits.")
    for row in baseline["fold_results"]:
        fit = baseline_fits.loc[baseline_fits["fold"].eq(row["fold"])].iloc[0]
        if fit["fold_identity_sha256"] != row["fold_identity_sha256"]:
            raise ValueError("Different baseline fold identity.")
        if not np.isclose(fit["macro_f1"], row["macro_f1"], atol=1e-12, rtol=0):
            raise ValueError("Different baseline result.")
        cv_gene_panel(rankings, fit)

    # Publish only after every saved fit and selected gene set has been checked.
    path = data_dir / "cv_gene_rankings.csv.gz"
    temporary = path.with_name(path.name + ".part")
    rankings.to_csv(temporary, index=False, compression={"method": "gzip", "mtime": 0})
    temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "experiment_fingerprint": fingerprint,
        "ranking_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "cv_fits_sha256": hashlib.sha256((data_dir / "experiment_cv_folds.csv").read_bytes()).hexdigest(),
        "ranking_rows": len(rankings),
        "method_fold_rankings": len(frames),
        "cv_fit_count": len(fits),
        "selector_fit_count": len(selector_fits),
        "baseline_fit_count": len(baseline_fits),
        "mi_primary_n_neighbors": summary["mi_primary_n_neighbors"],
        "selected_sets_verified": len(saved_sets),
        "source_checkpoint_sha256": source_hashes,
        "baseline_gene_source": "Same-fold candidates shared by all saved selector rankings; no selection ranking is assigned to the baseline.",
    }
    manifest_path = data_dir / "cv_gene_panel_manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.part")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    load_cv_gene_rankings(data_dir, summary, fits)
    print(
        f"Verified {len(saved_sets)} selector panels + {len(baseline_fits)} baseline panels; exported {len(frames)} reusable rankings ({len(rankings):,} rows). No models refitted."
    )


if __name__ == "__main__":
    export_panels()
