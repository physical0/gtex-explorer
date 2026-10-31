from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier


from gtex_explorer.paths import APP_ROOT as APP_DIR
DEFAULT_CACHE = APP_DIR.parent / 'results/notebook_run_500_1000_2000_l1_grid/part_a_cache.joblib'


def macro_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_count: int) -> dict:
    precision_values = []
    recall_values = []
    f1_values = []
    for label in range(class_count):
        true_positive = int(((y_true == label) & (y_pred == label)).sum())
        false_positive = int(((y_true != label) & (y_pred == label)).sum())
        false_negative = int(((y_true == label) & (y_pred != label)).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, np.finfo(float).eps)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    return {
        "macro_f1": float(np.mean(f1_values)),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the frozen Part A preprocessing and XGBoost prediction bundle."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=APP_DIR / "model_bundle")
    args = parser.parse_args()

    payload = joblib.load(args.cache)
    variables = payload["variables"]
    if int(payload["cache_metadata"].get("schema_version", 0)) < 3:
        raise ValueError("Part A cache schema 3 or newer is required.")

    X_development_log = variables["X_development_log"].copy()
    X_test_log = variables["X_test_log"].copy()
    metadata = variables["metadata"].copy()
    selected_genes = (
        variables["selected_genes"]
        .sort_values("selection_rank", kind="stable")["gene"]
        .astype(str)
        .tolist()
    )
    best_feature_count = int(variables["best_feature_count"])
    if len(selected_genes) != best_feature_count:
        raise ValueError("Selected gene count does not match the winning configuration.")
    missing_selected = sorted(set(selected_genes) - set(X_development_log.columns))
    if missing_selected:
        raise ValueError(f"Selected genes absent from development matrix: {missing_selected[:5]}")

    sample_column = variables["SAMPLE_COLUMN"]
    tissue_column = variables["TISSUE_COLUMN"]
    metadata = metadata.set_index(sample_column, drop=False)
    classes = np.asarray(sorted(metadata[tissue_column].dropna().unique()), dtype=object)
    class_to_index = {str(label): index for index, label in enumerate(classes)}

    y_development = np.asarray(
        [class_to_index[str(metadata.loc[sample_id, tissue_column])] for sample_id in X_development_log.index],
        dtype=int,
    )
    y_test = np.asarray(
        [class_to_index[str(metadata.loc[sample_id, tissue_column])] for sample_id in X_test_log.index],
        dtype=int,
    )

    selected_development_log = X_development_log.loc[:, selected_genes].to_numpy(dtype=float)
    selected_test_log = X_test_log.loc[:, selected_genes].to_numpy(dtype=float)
    scaler_mean = selected_development_log.mean(axis=0)
    scaler_scale = selected_development_log.std(axis=0, ddof=0)
    scaler_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    selected_development_scaled = (
        selected_development_log - scaler_mean
    ) / scaler_scale
    selected_test_scaled = (selected_test_log - scaler_mean) / scaler_scale

    parameters = dict(variables["XGB_BASE_PARAMETERS"])
    classifier = XGBClassifier(num_class=len(classes), **parameters)
    classifier.fit(selected_development_scaled, y_development)
    test_predictions = classifier.predict(selected_test_scaled).astype(int)
    measured_metrics = macro_metrics(y_test, test_predictions, len(classes))

    expected_summary = variables["final_test_summary"]
    expected_row = expected_summary.loc[
        expected_summary["classifier"].eq("xgboost")
    ].iloc[0]
    maximum_metric_difference = max(
        abs(measured_metrics[metric] - float(expected_row[metric]))
        for metric in measured_metrics
    )
    if maximum_metric_difference > 1e-10:
        raise RuntimeError(
            "Retrained bundle did not reproduce the notebook test metrics. "
            f"Maximum difference: {maximum_metric_difference:.3e}"
        )

    # The monotonic inverse of the log transform reconstructs the imputed TPM scale.
    selected_raw_tpm = np.exp2(selected_development_log) - 1.0
    raw_tpm_medians = np.median(selected_raw_tpm, axis=0)

    args.output.mkdir(parents=True, exist_ok=True)
    classifier.save_model(args.output / "xgboost_model.json")
    preprocessor = {
        "bundle_schema_version": 1,
        "experiment_fingerprint": variables["PART_A_EXPERIMENT_FINGERPRINT"],
        "selector_pipeline_version": variables["PART_A_SELECTOR_PIPELINE_VERSION"],
        "selector_id": variables["best_selector_id"],
        "feature_selection": variables["best_selector"],
        "l1_c": float(variables["best_l1_c"]),
        "feature_count": best_feature_count,
        "selected_genes": selected_genes,
        "raw_tpm_medians": raw_tpm_medians.astype(np.float32),
        "scaler_mean": scaler_mean.astype(np.float32),
        "scaler_scale": scaler_scale.astype(np.float32),
        "classes": classes.tolist(),
        "xgboost_parameters": parameters,
    }
    joblib.dump(preprocessor, args.output / "preprocessor.joblib", compress=3)

    demo_target_tissues = [
        "Thyroid",
        "Heart - Atrial Appendage",
        "Artery - Tibial",
    ]
    demo_positions: list[int] = []
    for target_tissue in demo_target_tissues:
        target_positions = np.flatnonzero(
            y_test == class_to_index[target_tissue]
        )
        if target_positions.size:
            demo_positions.append(int(target_positions[0]))
    for candidate_position in range(len(X_test_log)):
        if len(demo_positions) >= 3:
            break
        if candidate_position not in demo_positions:
            demo_positions.append(candidate_position)

    demo_tpm = pd.DataFrame(
        np.exp2(selected_test_log[demo_positions]) - 1.0,
        index=X_test_log.index[demo_positions],
        columns=selected_genes,
    )
    demo_tpm.index.name = "SAMPID"
    demo_tpm.reset_index().to_csv(APP_DIR / "data" / "demo_tpm.csv", index=False)

    model_sha256 = hashlib.sha256(
        (args.output / "xgboost_model.json").read_bytes()
    ).hexdigest()
    validation = {
        "cache_path": str(args.cache.resolve()),
        "cache_schema_version": payload["cache_metadata"].get("schema_version"),
        "development_samples": int(len(X_development_log)),
        "test_samples": int(len(X_test_log)),
        "selected_genes": best_feature_count,
        "classes": classes.tolist(),
        "measured_test_metrics": measured_metrics,
        "maximum_metric_difference_from_notebook": maximum_metric_difference,
        "xgboost_model_sha256": model_sha256,
        "status": "validated",
    }
    (args.output / "bundle_manifest.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
