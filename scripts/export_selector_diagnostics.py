"""Package existing notebook CSV diagnostics. Never fits models or loads joblib."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gtex_explorer.analysis.selector_diagnostics import DIAGNOSTIC_FILES, file_hash, validate_diagnostic_tables


def export_diagnostics() -> None:
    from gtex_explorer.paths import APP_ROOT as app_dir
    data_dir = app_dir / "data"
    source_dir = app_dir.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_only"
    summary = json.loads((data_dir / "run_summary.json").read_text(encoding="utf-8"))
    source_summary = json.loads((source_dir / "run_summary.json").read_text(encoding="utf-8"))
    if source_summary["experiment_fingerprint"] != summary["experiment_fingerprint"]:
        raise ValueError("Sumber diagnostik dan aplikasi berasal dari eksperimen berbeda.")
    fits = pd.read_csv(data_dir / "experiment_cv_folds.csv")
    source_fits = pd.read_csv(source_dir / "experiment_cv_folds.csv")
    keys = ["configuration_id", "fold"]
    pd.testing.assert_frame_equal(
        fits.sort_values(keys).reset_index(drop=True)[source_fits.columns],
        source_fits.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12,
    )
    rankings = pd.read_csv(data_dir / "cv_gene_rankings.csv.gz")
    diagnostics = {key: pd.read_csv(source_dir / filename.removesuffix(".gz")) for key, filename in DIAGNOSTIC_FILES.items()}
    validate_diagnostic_tables(diagnostics, fits, rankings)
    manifest = {
        "schema_version": 1, "experiment_fingerprint": summary["experiment_fingerprint"],
        "source": "results/notebook_run_500_1000_2000_l1_grid/part_a_only",
        "files": {},
        "bindings": {name: file_hash(data_dir / name) for name in ["experiment_cv_folds.csv", "cv_gene_rankings.csv.gz", "final_selector_fit_diagnostics.csv"]},
    }
    for key, filename in DIAGNOSTIC_FILES.items():
        target = data_dir / filename
        partial = target.with_name(target.name + ".part")
        diagnostics[key].to_csv(partial, index=False, compression="gzip" if filename.endswith(".gz") else None)
        partial.replace(target)
        manifest["files"][filename] = {"sha256": file_hash(target), "rows": len(diagnostics[key]), "source_sha256": file_hash(source_dir / filename.removesuffix(".gz"))}
    target = data_dir / "selector_diagnostics_manifest.json"
    partial = target.with_name(target.name + ".part")
    partial.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    partial.replace(target)
    print("Exported verified diagnostics for 25 method/fold rankings and 150 fold pairs; no model was retrained.")


if __name__ == "__main__":
    export_diagnostics()
