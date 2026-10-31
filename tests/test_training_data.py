from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler


def checkpoint_sources(tmp_path):
    genes = [f"ENSG{i:011d}.1" for i in range(1, 7)]
    ids = [f"GTEX-D{i}-SM-A" for i in range(12)]
    raw = pd.DataFrame(np.arange(72, dtype=np.float32).reshape(12, 6) + 1,
                       index=ids, columns=genes)
    log = np.log2(raw + 1)
    scaler = StandardScaler().fit(log)
    metadata = pd.DataFrame({"SAMPID": ids + ["HOLDOUT"],
                             "SUBJID": [f"D{i}" for i in range(12)] + ["SECRET"],
                             "SMTSD": ["A", "B"] * 6 + ["A"]})
    common = {"metadata": metadata, "SAMPLE_COLUMN": "SAMPID",
              "DONOR_COLUMN": "SUBJID", "TISSUE_COLUMN": "SMTSD"}
    fold_list = []
    for fold in range(3):
        val = np.arange(fold * 4, fold * 4 + 4)
        train = np.setdiff1d(np.arange(12), val)
        fold_list.append({"fold": fold + 1, "X_train": scaler.transform(log)[train].astype('float32'),
                          "X_validation": scaler.transform(log)[val].astype('float32'),
                          "y_train": np.array([0, 1] * 6)[train],
                          "y_validation": np.array([0, 1] * 6)[val], "gene_names": genes,
                          "train_sample_ids": np.array(ids)[train], "validation_sample_ids": np.array(ids)[val]})
    fingerprint = "verified-test-fingerprint"
    values = {**common, "X_development_log": log, "X_test_log": log.iloc[:1].rename(index={ids[0]: "HOLDOUT"})}
    paths = [tmp_path / name for name in ["cache.joblib", "pre.joblib", "fold.joblib"]]
    joblib.dump({"cache_metadata": {"schema_version": 5, "experiment_fingerprint": fingerprint}, "variables": values}, paths[0])
    joblib.dump({"experiment_fingerprint": fingerprint, "variables": {"X_development_log": log,
                 "X_development_scaled": scaler.transform(log).astype('float32'), "final_scaler": scaler}}, paths[1])
    joblib.dump({"experiment_fingerprint": fingerprint, "variables": {**common,
                 "X_development_raw": raw, "prepared_cv_folds": fold_list}}, paths[2])
    return paths


def test_atomic_publish_retries_transient_windows_lock_without_overwriting(tmp_path, monkeypatch):
    from gtex_explorer.training.data import atomic_publish_directory
    import os
    source, target = tmp_path / 'staging', tmp_path / 'published'
    source.mkdir()
    (source / 'complete.txt').write_text('complete')
    real_replace = os.replace
    attempts = []
    def locked_replace(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError('temporary scanner lock')
            error.winerror = 5
            raise error
        return real_replace(src, dst)
    monkeypatch.setattr(os, 'replace', locked_replace)
    atomic_publish_directory(source, target)
    assert (target / 'complete.txt').read_text() == 'complete'
    assert len(attempts) == 3
    source.mkdir()
    with pytest.raises(FileExistsError):
        atomic_publish_directory(source, target)


def test_export_contains_only_development_and_rejects_mixed_fingerprints(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.data import load_training_data
    paths = checkpoint_sources(tmp_path)
    out = tmp_path / "export"
    report = export_training_data(*paths, out)
    data = load_training_data(out)
    assert len(data.sample_ids) == 12
    assert "HOLDOUT" not in data.sample_ids
    assert "SECRET" not in data.donor_ids
    assert data.raw_tpm.dtype == np.float32
    assert isinstance(data.raw_tpm, np.memmap)
    assert data.classes == ("A", "B")
    np.testing.assert_array_equal(data.labels, [0, 1] * 6)
    assert report["holdout_included"] is False
    assert np.load(out / "raw_tpm.npy", allow_pickle=False).shape == (12, 6)
    payload = joblib.load(paths[2])
    payload["experiment_fingerprint"] = "different"
    joblib.dump(payload, paths[2])
    with pytest.raises(ValueError, match="fingerprint"):
        export_training_data(*paths, tmp_path / "bad")


def test_missing_corrupt_or_unsafe_export_is_rejected(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.data import load_training_data
    paths = checkpoint_sources(tmp_path)
    out = tmp_path / "export"
    export_training_data(*paths, out)
    with (out / "raw_tpm.npy").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        load_training_data(out)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["files"]["../outside.npy"] = "ignored"
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_training_data(out)
    with pytest.raises(FileNotFoundError):
        load_training_data(tmp_path / "missing")


def test_donor_assignment_coverage_and_validation_cannot_change_preprocessing(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.data import load_training_data, make_cv_assignments, prepare_partition
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'export')
    data = load_training_data(tmp_path / 'export')
    assignments = make_cv_assignments(data, 3)
    assert sorted(np.concatenate([a.validation_indices for a in assignments])) == list(range(12))
    for assignment in assignments:
        assert not set(np.array(data.donor_ids)[assignment.train_indices]) & set(np.array(data.donor_ids)[assignment.validation_indices])
    assignment = assignments[0]
    first = prepare_partition(data, assignment, tmp_path / 'first')
    data.raw_tpm = np.array(data.raw_tpm)
    data.raw_tpm[assignment.validation_indices] = 1000000
    second = prepare_partition(data, assignment, tmp_path / 'second')
    assert first.gene_names == second.gene_names
    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.raw_tpm_medians, second.raw_tpm_medians)
    np.testing.assert_array_equal(first.scaler_mean, second.scaler_mean)
    assert not np.array_equal(first.X_validation, second.X_validation)
    assert assignments[0].split_hash != make_cv_assignments(data, 2)[0].split_hash


def test_missing_class_fold_and_missing_raw_data_are_not_silently_used(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.data import load_training_data, make_cv_assignments
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'export')
    data = load_training_data(tmp_path / 'export')
    data.raw_tpm = None
    with pytest.raises(ValueError, match='raw|cached'):
        make_cv_assignments(data, 2)
    data.donor_ids = tuple('onlyA' if y == 0 else 'onlyB' for y in data.labels)
    with pytest.raises(ValueError, match='class|tissue'):
        make_cv_assignments(data, 2)
