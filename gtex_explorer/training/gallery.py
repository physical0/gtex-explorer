"""Publish a gallery receipt only after the complete saved run is verified."""

import json
import time
from pathlib import Path

import pandas as pd
from xgboost.core import XGBoostError

from gtex_explorer.platform.cloud.state import operation
from gtex_explorer.training.data import file_hash, stable_hash, verify_files_cached
from gtex_explorer.training.jobs import atomic_json, job_path


def finalize_gallery(run_id, state_root):
    """Verify disk artifacts, then atomically index them. Never fit a model here."""
    from gtex_explorer.platform.model_registry import _entry, load_saved_model
    from gtex_explorer.training.core import aggregate_cv
    from gtex_explorer.training.panels import load_run_panel

    root = job_path(run_id, state_root)
    saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
    config = saved["config"]
    bundle = Path(state_root) / "models" / run_id
    entry = _entry(bundle)
    pre, _ = load_saved_model(entry)
    manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
    if (
        entry.model_id != run_id
        or pre.get("run_id") != run_id
        or manifest["provenance"]["identity"] != saved["identity"]
        or manifest["provenance"]["data_fingerprint"] != saved["data_fingerprint"]
        or manifest["config"] != config
    ):
        raise ValueError("Saved model does not match training run")
    final_genes = pd.read_csv(bundle / "selected_genes.csv")
    if final_genes.gene.tolist() != pre["selected_genes"] or final_genes.selection_rank.tolist() != list(
        range(1, config["top_k"] + 1)
    ):
        raise ValueError("Final gene panel does not match saved model")

    run_files = ["config.json"]
    rows, sets = [], []
    summary_path = bundle / "cv_summary.json"
    if config["evaluate_cv"]:
        for fold in range(1, config["cv_folds"] + 1):
            # Old gene-only checkpoints remain history, not fully verified gallery results.
            if not (root / f"fold_{fold}_panel" / "manifest.json").is_file():
                raise ValueError(f"Fold {fold} gene panel is incomplete")
            genes, info = load_run_panel(state_root, run_id, fold)
            payload = json.loads((root / f"fold_{fold}.json").read_text(encoding="utf-8"))
            row = payload["result"]
            if (
                len(genes) != config["top_k"]
                or not genes.gene.is_unique
                or info["training_samples"] != row["training_samples"]
                or info["validation_samples"] != row["validation_samples"]
            ):
                raise ValueError(f"Fold {fold} panel/metric mismatch")
            rows.append(row)
            sets.append(set(row["selected_genes"]))
            run_files += [f"fold_{fold}.json", f"fold_{fold}_panel/manifest.json", f"fold_{fold}_panel/genes.csv"]
        summary = aggregate_cv(
            rows, sets, expected_folds=config["cv_folds"], supervised_selector=config["selector"] != "baseline"
        )
        recorded = json.loads(summary_path.read_text(encoding="utf-8"))
        if stable_hash(summary) != stable_hash(recorded):
            raise ValueError("Saved CV summary does not match fold results")
    else:
        if summary_path.exists():
            raise ValueError("Training-only run cannot claim CV results")
        summary = None
    holdout_path = bundle / "holdout_summary.json"
    holdout = None
    if config.get("holdout_percent"):
        holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
        if holdout["holdout_percent"] != config["holdout_percent"] or not holdout["validation_samples"]:
            raise ValueError("Saved hold-out score does not match the configuration")
    elif holdout_path.exists():
        raise ValueError("Run without a hold-out cannot claim hold-out results")

    receipt = {
        "schema_version": 1,
        "run_id": run_id,
        "identity": saved["identity"],
        "saved_at": time.time(),
        "feature_count": config["top_k"],
        "cv_summary": summary,
        "holdout_summary": holdout,
        "run_files": {name: file_hash(root / name) for name in run_files},
        "model_files": {**manifest["files"], "bundle_manifest.json": file_hash(bundle / "bundle_manifest.json")},
    }
    atomic_json(root / "gallery.json", receipt)
    return receipt


@operation
def list_gallery(state_root, *, issues=None):
    """Only finalized, completed runs are visible, including verified older runs."""
    entries = []
    for path in (Path(state_root) / "runs").glob("run_*/status.json"):
        run_id = path.parent.name
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("status") != "completed":
                continue
            root = job_path(run_id, state_root)
            index = root / "gallery.json"
            receipt = (
                json.loads(index.read_text(encoding="utf-8"))
                if index.exists()
                else finalize_gallery(run_id, state_root)
            )
            if receipt["schema_version"] != 1 or receipt["run_id"] != run_id:
                raise ValueError("Invalid gallery record")
            verify_files_cached(root, receipt["run_files"])
            verify_files_cached(Path(state_root) / "models" / run_id, receipt["model_files"])
            entries.append(receipt)
        except (OSError, ValueError, KeyError, TypeError, XGBoostError) as error:
            if issues is not None:
                issues.append({"run_id": run_id, "reason": str(error)})
    return sorted(entries, key=lambda entry: entry["saved_at"], reverse=True)
