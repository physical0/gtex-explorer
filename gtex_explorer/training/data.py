"""Validated, non-pickle development data for local model training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- File integrity ---


def file_hash(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_directory(source: Path, target: Path) -> None:
    """Move a finished folder into place, retrying while Windows holds it."""
    # Windows scanners can briefly hold a newly closed file. Never mask a
    # persistent permission failure or overwrite an already published bundle.
    for attempt in range(6):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def discard_directory(path: Path, root: Path) -> None:
    """Remove a directory in one step. Hidden names are never published as durable state.

    A failed rename leaves everything in place. A file still locked during cleanup
    only leaves a hidden leftover, which the next removal clears.
    """
    trash = Path(root) / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    atomic_publish_directory(Path(path), trash / uuid.uuid4().hex)
    for leftover in trash.iterdir():
        shutil.rmtree(leftover, ignore_errors=True)


def atomic_replace_file(source: Path, target: Path) -> None:
    """Replace a file in one step, retrying while Windows holds it."""
    for attempt in range(6):
        try:
            os.replace(source, target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32) or attempt == 5:
                raise
            time.sleep(0.05 * (attempt + 1))


def stable_hash(value: object) -> str:
    """Return a hash of JSON-compatible data that does not depend on key order."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    ).hexdigest()


def safe_path(root: Path, relative: str) -> Path:
    """Resolve a relative artifact path and refuse anything outside the root."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or ":" in relative:
        raise ValueError("Unsafe artifact path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Unsafe artifact path")
    return resolved


def verify_files(root: Path, files: dict) -> None:
    """Check that files match their recorded hashes."""
    # Validate paths before opening any file, including symlink resolution.
    paths = [(safe_path(root, name), expected) for name, expected in files.items()]
    for path, expected in paths:
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {path.name}")


_known_hashes = {}


def cached_file_hash(path: Path) -> str:
    """For paths that run on every rerun or poll: rehash only after a file's time or size changes."""
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    known = _known_hashes.get(str(path))
    if known and known[0] == signature:
        return known[1]
    digest = file_hash(path)
    _known_hashes[str(path)] = (signature, digest)
    return digest


def verify_files_cached(root: Path, files: dict) -> None:
    """Check file hashes, reusing earlier results for unchanged files."""
    paths = [(safe_path(root, name), expected) for name, expected in files.items()]
    for path, expected in paths:
        if not path.is_file() or cached_file_hash(path) != expected:
            raise ValueError(f"Artifact hash mismatch: {path.name}")


# --- Identifiers ---


def unique_ids(values: list, *, genes: bool = False) -> tuple[str, ...]:
    """Check that identifiers are present and not duplicated."""
    ids = tuple(str(x) for x in values)
    normalized = [x.split(".")[0].upper() if genes else x for x in ids]
    if not ids or any(not x.strip() for x in ids) or len(set(normalized)) != len(ids):
        raise ValueError("Empty or duplicate identifiers")
    return ids


# --- Data containers ---


@dataclass
class PreparedData:
    """One partition after preprocessing, ready for selection and training."""

    X_train: np.ndarray
    y_train: np.ndarray
    gene_names: tuple[str, ...]
    train_sample_ids: tuple[str, ...]
    train_donor_ids: tuple[str, ...]
    fingerprint: str
    X_validation: np.ndarray | None = None
    y_validation: np.ndarray | None = None
    validation_sample_ids: tuple[str, ...] = ()
    validation_donor_ids: tuple[str, ...] = ()
    raw_tpm_medians: np.ndarray | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None
    training_log: np.ndarray | None = None


@dataclass
class FoldAssignment:
    """The sample positions used for training and validation."""

    fold: int
    train_indices: np.ndarray
    validation_indices: np.ndarray
    split_hash: str


@dataclass
class TrainingData:
    """A training dataset with its samples, donors, labels, and TPM values."""

    root: Path
    manifest: dict
    sample_ids: tuple[str, ...]
    donor_ids: tuple[str, ...]
    classes: tuple[str, ...]
    labels: np.ndarray
    raw_gene_names: tuple[str, ...]
    raw_tpm: np.ndarray | None

    @property
    def fingerprint(self) -> str:
        return stable_hash(self.manifest)


# --- Loading ---


def _array(root: Path, manifest: dict, name: str) -> np.ndarray:
    """Load one array listed in the manifest and check its type."""
    if name not in manifest["files"]:
        raise ValueError(f"Unregistered array: {name}")
    array = np.load(safe_path(root, name), allow_pickle=False, mmap_mode="r")
    if array.dtype.kind not in "fiu" or array.ndim not in (1, 2):
        raise ValueError("Invalid numeric array")
    return array


def load_training_data(root: Path) -> TrainingData:
    """Load a training dataset after verifying its manifest and files."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in (1, 2) or manifest.get("holdout_included") is not False:
        raise ValueError("Unsupported training data schema")
    verify_files(root, manifest["files"])
    samples = unique_ids(manifest["sample_ids"])
    donors = tuple(manifest["donor_ids"])
    classes = unique_ids(manifest["classes"])
    labels = _array(root, manifest, "labels.npy")
    if len(donors) != len(samples) or any(not str(x).strip() for x in donors):
        raise ValueError("Donor axis mismatch")
    if (
        labels.dtype.kind not in "iu"
        or len(classes) < 2
        or labels.shape != (len(samples),)
        or set(labels.tolist()) != set(range(len(classes)))
    ):
        raise ValueError("Class mapping mismatch")
    raw_genes = unique_ids(json.loads((root / "raw_genes.json").read_text()), genes=True)
    raw = _array(root, manifest, "raw_tpm.npy") if "raw_tpm.npy" in manifest["files"] else None
    if raw is not None:
        if raw.shape != (len(samples), len(raw_genes)) or raw.dtype != np.float32:
            raise ValueError("Raw TPM shape/dtype mismatch")
        for start in range(0, len(raw), 64):
            block = raw[start : start + 64]
            if np.isinf(block).any() or (block < 0).any():
                raise ValueError("Invalid raw TPM")
    data = TrainingData(root, manifest, samples, donors, classes, labels, raw_genes, raw)
    if manifest["schema_version"] == 1:
        load_final_prepared(data)
    elif raw is None or not 1 <= manifest.get("candidate_count", 0) <= min(5000, len(raw_genes)):
        raise ValueError("Invalid uploaded training data/candidate count")
    # Fold membership is checked again against requested assignments before reuse.
    for spec in manifest["cached_folds"]:
        _load_prepared(data, spec)
    return data


def _load_prepared(data: TrainingData, spec: dict) -> PreparedData:
    """Load a partition that the notebook already preprocessed."""
    genes = unique_ids(spec["genes"], genes=True)
    train = unique_ids(spec["train_samples"])
    lookup = {sample: i for i, sample in enumerate(data.sample_ids)}
    if not set(train).issubset(lookup):
        raise ValueError("Unknown training sample")
    X = _array(data.root, data.manifest, spec["X_train"])
    y = np.asarray([data.labels[lookup[s]] for s in train])
    if X.shape != (len(train), len(genes)) or not np.isfinite(X).all():
        raise ValueError("Prepared training shape/values mismatch")
    result = PreparedData(
        X,
        y,
        genes,
        train,
        tuple(data.donor_ids[lookup[s]] for s in train),
        stable_hash({"data": data.fingerprint, "partition": spec}),
    )
    if "validation_samples" in spec:
        val = unique_ids(spec["validation_samples"])
        if not set(val).issubset(lookup) or set(train) & set(val):
            raise ValueError("Invalid validation membership")
        result.X_validation = _array(data.root, data.manifest, spec["X_validation"])
        result.y_validation = np.asarray([data.labels[lookup[s]] for s in val])
        result.validation_sample_ids = val
        result.validation_donor_ids = tuple(data.donor_ids[lookup[s]] for s in val)
        if set(result.train_donor_ids) & set(result.validation_donor_ids):
            raise ValueError("Donor leakage")
        if result.X_validation.shape != (len(val), len(genes)) or not np.isfinite(result.X_validation).all():
            raise ValueError("Prepared validation shape/values mismatch")
    else:
        if train != data.sample_ids:
            raise ValueError("Final fit must use all development samples")
        for key in ["raw_tpm_medians", "scaler_mean", "scaler_scale"]:
            values = _array(data.root, data.manifest, spec[key])
            if values.shape != (len(genes),) or not np.isfinite(values).all():
                raise ValueError("Invalid preprocessing constants")
            setattr(result, key, values)
        if (result.raw_tpm_medians < 0).any() or (result.scaler_scale <= 0).any():
            raise ValueError("Invalid medians/scales")
        result.training_log = _array(data.root, data.manifest, spec["training_log"])
        if result.training_log.shape != X.shape or not np.isfinite(result.training_log).all():
            raise ValueError("Invalid final log matrix")
    return result


def load_final_prepared(data: TrainingData) -> PreparedData:
    """Return the preprocessed data for training the final model."""
    if data.manifest["schema_version"] == 2:
        assignment = FoldAssignment(
            0, np.arange(len(data.labels)), np.array([], dtype=int), stable_hash({"final": data.fingerprint})
        )
        return prepare_partition(data, assignment, data.root / "preprocessing_cache")
    return _load_prepared(data, data.manifest["final"])


# --- Data splits ---


def make_cv_assignments(data: TrainingData, fold_count: int, within: np.ndarray | None = None) -> list[FoldAssignment]:
    """Folds over all samples, or only over the positions in `within` when a hold-out was set aside first."""
    from sklearn.model_selection import StratifiedGroupKFold

    if isinstance(fold_count, bool) or not isinstance(fold_count, int) or not 2 <= fold_count <= 10:
        raise ValueError("Fold count must be an integer from 2 to 10")
    splitter = StratifiedGroupKFold(n_splits=fold_count, shuffle=True, random_state=42)
    samples = np.arange(len(data.labels)) if within is None else np.asarray(within)
    assignments = []
    observed = np.zeros(len(data.labels), dtype=int)
    donor_fold = {}
    for number, (train, val) in enumerate(
        splitter.split(np.zeros(len(samples)), np.asarray(data.labels)[samples], [data.donor_ids[i] for i in samples]),
        1,
    ):
        train, val = samples[train], samples[val]
        if not len(train) or not len(val):
            raise ValueError(f"Fold {number} is empty")
        for side, positions in [("training", train), ("validation", val)]:
            missing = set(range(len(data.classes))) - set(data.labels[positions].tolist())
            if missing:
                names = ", ".join(data.classes[x] for x in sorted(missing))
                raise ValueError(f"Fold {number} {side} missing tissue classes: {names}")
        train_donors = {data.donor_ids[i] for i in train}
        val_donors = {data.donor_ids[i] for i in val}
        if train_donors & val_donors:
            raise ValueError("Donor leakage")
        for donor in val_donors:
            if donor in donor_fold:
                raise ValueError("Donor validates more than once")
            donor_fold[donor] = number
        observed[val] += 1
        identity = stable_hash(
            {
                "train": [data.sample_ids[i] for i in train],
                "validation": [data.sample_ids[i] for i in val],
                "fold_count": fold_count,
                "seed": 42,
            }
        )
        assignments.append(FoldAssignment(number, train, val, identity))
    if (
        not np.all(observed[samples] == 1)
        or observed.sum() != len(samples)
        or set(donor_fold) != {data.donor_ids[i] for i in samples}
    ):
        raise ValueError("Incomplete validation assignment")
    if data.raw_tpm is None and not _cached_assignment_matches(data, assignments):
        raise ValueError("Raw development data required for non-cached assignments")
    return assignments


def make_holdout_assignment(data: TrainingData, percent: int) -> FoldAssignment:
    """Set whole donors aside, keeping each tissue close to the chosen share."""
    import pandas as pd

    labels = np.asarray(data.labels)
    donor_index, donors = pd.factorize(np.asarray(data.donor_ids), sort=True)
    counts = np.zeros((len(donors), len(data.classes)), dtype=int)
    np.add.at(counts, (donor_index, labels), 1)
    targets = np.maximum(1, np.round(counts.sum(axis=0) * percent / 100)).astype(int)
    order = np.random.default_rng(42).permutation(len(donors))
    taken = np.zeros(len(data.classes), dtype=int)
    chosen = set()
    for donor in order:
        gain = np.minimum(counts[donor], np.maximum(targets - taken, 0)).sum()
        if gain > counts[donor].sum() - gain:
            chosen.add(donor)
            taken += counts[donor]
    # A donor with many tissues may overshoot, but every tissue must still be tested.
    for donor in order:
        if donor not in chosen and counts[donor][taken == 0].any():
            chosen.add(donor)
            taken += counts[donor]
    in_holdout = np.isin(donor_index, list(chosen))
    train, validation = np.flatnonzero(~in_holdout), np.flatnonzero(in_holdout)
    for side, positions in [("development", train), ("holdout", validation)]:
        missing = set(range(len(data.classes))) - set(labels[positions].tolist())
        if missing:
            names = ", ".join(data.classes[x] for x in sorted(missing))
            raise ValueError(
                f"Data {side} dengan porsi holdout {percent}% tidak mencakup jaringan: {names}. "
                "Ubah porsi holdout, atau pilih 0% untuk melatih dengan semua sampel tanpa skor."
            )
    identity = stable_hash(
        {
            "train": [data.sample_ids[i] for i in train],
            "validation": [data.sample_ids[i] for i in validation],
            "holdout_percent": percent,
            "seed": 42,
        }
    )
    return FoldAssignment(0, train, validation, identity)


def run_split(data: TrainingData, config) -> tuple[list[FoldAssignment], FoldAssignment | None, str | None]:
    """The CV folds and hold-out of a run, with the hash that identifies them."""
    # The hold-out is fixed first, so its donors can never enter a fold.
    holdout = make_holdout_assignment(data, config.holdout_percent) if config.holdout_percent else None
    assignments = (
        make_cv_assignments(data, config.cv_folds, holdout.train_indices if holdout else None)
        if config.evaluate_cv
        else []
    )
    parts = assignments + ([holdout] if holdout else [])
    return assignments, holdout, stable_hash([a.split_hash for a in parts]) if parts else None


def _cached_assignment_matches(data, assignments):
    """Return whether the requested folds match the cached ones."""
    specs = data.manifest["cached_folds"]
    if len(specs) != len(assignments):
        return False
    return all(
        spec["train_samples"] == [data.sample_ids[i] for i in assignment.train_indices]
        and spec["validation_samples"] == [data.sample_ids[i] for i in assignment.validation_indices]
        for spec, assignment in zip(specs, assignments)
    )


# --- Preprocessing ---


def _impute_missing(train):
    """Fill missing TPM with medians of the training part. A gene with no values at all is dropped."""
    if not train.isna().to_numpy().any():
        return train, None
    medians = train.median(axis=0)
    valid = medians.notna()
    return train.loc[:, valid].fillna(medians[valid]), medians


def _technical_filter(train):
    """Keep genes with at least 1 TPM in 20% of samples and some variation."""
    return ((train >= 1).mean(axis=0) >= 0.2) & (train.var(axis=0, ddof=0) > 0)


def usable_candidate_genes(data: TrainingData, holdout_percent: int | None = None, cv_folds: int | None = None) -> int:
    """How many genes pass the technical filter on every part a run would train on."""
    import pandas as pd

    parts = [np.arange(len(data.labels))]
    # A refused split is reported by its own check, so count what can still be formed.
    try:
        if holdout_percent:
            parts = [make_holdout_assignment(data, holdout_percent).train_indices]
        if cv_folds:
            parts += [fold.train_indices for fold in make_cv_assignments(data, cv_folds, parts[0])]
    except ValueError:
        pass
    return min(
        int(
            _technical_filter(
                _impute_missing(pd.DataFrame(np.array(data.raw_tpm[part], copy=True), columns=data.raw_gene_names))[0]
            ).sum()
        )
        for part in parts
    )


def prepare_partition(data: TrainingData, assignment: FoldAssignment, cache_dir: Path) -> PreparedData:
    """Preprocess one partition, reusing a saved copy when it still matches."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Reuse only verified original memberships under compatible preprocessing libraries.
    specs = data.manifest["cached_folds"]
    versions = {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]}
    compatible = all(data.manifest["versions"][p] == v for p, v in versions.items())
    for spec in specs:
        if (
            assignment.fold
            and compatible
            and spec["train_samples"] == [data.sample_ids[i] for i in assignment.train_indices]
            and spec["validation_samples"] == [data.sample_ids[i] for i in assignment.validation_indices]
        ):
            return _load_prepared(data, spec)
    if data.raw_tpm is None:
        raise ValueError("Raw development data unavailable for this split")
    fingerprint = stable_hash(
        {
            "data": data.fingerprint,
            "split": assignment.split_hash,
            "preprocessing": data.manifest["preprocessing_version"],
            "versions": versions,
        }
    )
    target = Path(cache_dir) / fingerprint
    if target.is_dir():
        return _read_partition(target, fingerprint)
    train = pd.DataFrame(np.array(data.raw_tpm[assignment.train_indices], copy=True), columns=data.raw_gene_names)
    val = pd.DataFrame(np.array(data.raw_tpm[assignment.validation_indices], copy=True), columns=data.raw_gene_names)
    train, medians = _impute_missing(train)
    if medians is not None:
        val = val.loc[:, medians.notna()]
    technical = _technical_filter(train)
    count = int(data.manifest["candidate_count"])
    if technical.sum() < count:
        raise ValueError(
            f"Hanya {technical.sum()} gen lolos filter teknis pada data training partisi ini, "
            f"sedangkan {count} gen kandidat diminta. Kurangi Gen kandidat dan top-k bila perlu, lalu mulai pelatihan baru."
        )
    train_log = np.log2(train.loc[:, technical] + 1.0)
    variance = train_log.var(axis=0, ddof=0).to_numpy()
    indices = np.lexsort((np.arange(len(variance)), -variance))[:count]
    genes = tuple(train_log.columns[indices])
    selected_medians = train.loc[:, list(genes)].median(axis=0) if medians is None else medians.loc[list(genes)]
    selected_validation = val.loc[:, list(genes)]
    if selected_validation.isna().to_numpy().any():
        selected_validation = selected_validation.fillna(selected_medians)
    train_log = train_log.iloc[:, indices].to_numpy(copy=True)
    val_log = np.log2(selected_validation + 1.0).to_numpy(copy=True)
    scaler = StandardScaler()
    X = scaler.fit_transform(train_log).astype(np.float32, copy=False)
    Xval = scaler.transform(val_log).astype(np.float32, copy=False) if len(val_log) else None
    result = PreparedData(
        X,
        data.labels[assignment.train_indices],
        genes,
        tuple(data.sample_ids[i] for i in assignment.train_indices),
        tuple(data.donor_ids[i] for i in assignment.train_indices),
        fingerprint,
        Xval,
        data.labels[assignment.validation_indices] if len(val_log) else None,
        tuple(data.sample_ids[i] for i in assignment.validation_indices),
        tuple(data.donor_ids[i] for i in assignment.validation_indices),
        selected_medians.to_numpy(),
        scaler.mean_,
        scaler.scale_,
        train_log,
    )
    _write_partition(result, target)
    return result


# --- Partition cache ---


def _write_partition(prepared: PreparedData, target: Path):
    """Save a preprocessed partition in one step."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preparing-", dir=target.parent) as directory:
        root = Path(directory) / "data"
        root.mkdir()
        meta = {}
        for name, value in vars(prepared).items():
            if isinstance(value, np.ndarray):
                np.save(root / f"{name}.npy", value, allow_pickle=False)
                meta[name] = {"array": f"{name}.npy"}
            else:
                meta[name] = value
        files = {p.name: file_hash(p) for p in root.iterdir()}
        (root / "partition.json").write_text(json.dumps({"values": meta, "files": files}), encoding="utf-8")
        atomic_publish_directory(root, target)


def _read_partition(root: Path, fingerprint: str) -> PreparedData:
    """Read a saved partition after verifying it."""
    payload = json.loads((root / "partition.json").read_text(encoding="utf-8"))
    verify_files(root, payload["files"])
    values = payload["values"]
    if values["fingerprint"] != fingerprint:
        raise ValueError("Preprocessing cache identity mismatch")
    for key, value in values.items():
        if isinstance(value, dict):
            if value["array"] not in payload["files"]:
                raise ValueError("Unregistered cache array")
            values[key] = np.load(safe_path(root, value["array"]), allow_pickle=False, mmap_mode="r")
        elif key.endswith("_ids") or key == "gene_names":
            values[key] = tuple(value)
    result = PreparedData(**values)
    if not np.isfinite(result.X_train).all() or (
        result.X_validation is not None and not np.isfinite(result.X_validation).all()
    ):
        raise ValueError("Invalid preprocessing cache values")
    return result
