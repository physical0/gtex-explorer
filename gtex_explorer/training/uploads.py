"""Audit user TPM/metadata and persist immutable, raw-only training datasets."""

import csv
import hashlib
import importlib.metadata
import io
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from gtex_explorer.platform.cloud.state import operation
from gtex_explorer.training.data import TrainingData, atomic_publish_directory, file_hash, stable_hash, unique_ids


@dataclass
class UploadAudit:
    """The checked contents of an uploaded TPM matrix and its metadata."""

    raw_tpm: np.ndarray
    gene_names: tuple
    sample_ids: tuple
    donor_ids: tuple
    classes: tuple
    labels: np.ndarray
    source: dict


BLANK_VALUES = ["", "NA", "NaN", "nan", "N/A"]


# --- Reading ---


def read_upload_table(content: bytes, filename: str, *, numeric: bool = False) -> pd.DataFrame:
    """Read metadata as text, or an expression matrix with text IDs and numeric values.

    numeric=True lets the C parser read value columns as floats. Reading every cell
    as a Python string costs roughly 15x the file size in memory.
    """
    if not content.strip():
        raise ValueError("Berkas kosong. Unggah matriks TPM atau metadata yang berisi data.")
    stream = io.BytesIO(content)
    first = stream.readline().decode("utf-8-sig").strip()
    skip = 2 if first in {"#1.2", "#1.3"} else 0
    stream.seek(0)
    sep = "\t" if skip or "\t" in first else ","
    # pandas renames duplicate column labels, so inspect the actual header first.
    # Only the first lines are read; decoding the whole file here doubled its memory.
    lines = []
    for _ in range(skip + 1):
        line = stream.readline()
        if not line:
            break
        lines.append(line)
    stream.seek(0)
    if len(lines) <= skip:
        raise ValueError("Header GCT tidak lengkap. Berkas harus memuat versi, dimensi, header, dan data.")
    header = lines[skip].decode("utf-8-sig").rstrip("\r\n")
    columns = next(csv.reader([header], delimiter=sep))
    if len(set(columns)) != len(columns):
        raise ValueError("ID atau nama kolom duplikat pada header berkas.")
    if numeric:
        frame = pd.read_csv(
            stream,
            sep=sep,
            skiprows=skip,
            low_memory=False,
            keep_default_na=False,
            dtype={0: str},
            na_values={column: BLANK_VALUES for column in columns[1:]},
        )
    else:
        frame = pd.read_csv(stream, sep=sep, skiprows=skip, dtype=str, keep_default_na=False)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Berkas harus memiliki baris data dan sedikitnya dua kolom.")
    return frame


# --- Validation ---


def audit_uploads(
    expression: bytes,
    expression_name: str,
    metadata: bytes,
    metadata_name: str,
    sample_column="SAMPID",
    donor_column="SUBJID",
    tissue_column="SMTSD",
) -> UploadAudit:
    """Check that an uploaded matrix and its metadata can be used for training."""
    meta = read_upload_table(metadata, metadata_name)
    required = [sample_column, donor_column, tissue_column]
    if len(set(required)) != 3 or not set(required).issubset(meta.columns):
        raise ValueError("Pilih tiga kolom metadata yang berbeda untuk ID sampel, donor, dan jaringan.")
    meta = meta[required].copy()
    for column, label in [(sample_column, "sampel"), (donor_column, "donor"), (tissue_column, "jaringan")]:
        meta[column] = meta[column].str.strip()
        if meta[column].eq("").any():
            raise ValueError(f"ID atau label {label} kosong pada metadata.")
    if meta[sample_column].duplicated().any():
        raise ValueError("ID sampel duplikat pada metadata.")
    frame = read_upload_table(expression, expression_name, numeric=True)
    ids = set(meta[sample_column])
    first_values = frame.iloc[:, 0].str.strip()
    row_matches = first_values.isin(ids).any()
    col_matches = bool(set(frame.columns[1:]) & ids)
    if row_matches and col_matches:
        raise ValueError("Orientasi matriks ambigu. Gunakan ID sampel hanya pada satu sumbu.")
    if not row_matches and not col_matches:
        raise ValueError("ID sampel ekspresi tidak ditemukan pada metadata. Pencocokan ID harus persis sama.")
    if col_matches:
        if "Description" in frame.columns:
            frame = frame.drop(columns="Description")
        body = frame.set_index(frame.columns[0])
        sample_axis, gene_axis = body.columns, body.index
        orientation = "gen x sampel"
    else:
        body = frame.set_index(frame.columns[0])
        sample_axis, gene_axis = body.index, body.columns
        orientation = "sampel x gen"
    try:
        samples = unique_ids(sample_axis.tolist())
        unique_ids(gene_axis.tolist(), genes=True)
    except ValueError as error:
        raise ValueError("ID sampel atau ID gen kosong/duplikat setelah suffix versi dihapus.") from error
    missing = set(samples) - ids
    if missing:
        raise ValueError(f'{len(missing)} ID sampel tidak memiliki metadata: {", ".join(sorted(missing)[:3])}.')
    values = _expression_values(body)
    if col_matches:
        values = np.ascontiguousarray(values.T)
    if np.isinf(values).any():
        raise ValueError("Nilai tak hingga ditemukan pada ekspresi TPM.")
    if (values < 0).any():
        raise ValueError("Nilai TPM negatif ditemukan. Gunakan ekspresi TPM sebelum transformasi.")
    if np.isnan(values).all(axis=1).any():
        raise ValueError("Ada sampel tanpa pengukuran ekspresi. Lengkapi data sampel tersebut.")
    aligned = meta.set_index(sample_column).loc[list(samples)]
    classes = tuple(sorted(aligned[tissue_column].unique()))
    if len(classes) < 2:
        raise ValueError("Training memerlukan sedikitnya dua kelas jaringan.")
    labels = aligned[tissue_column].map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
    source = {
        "kind": "user_upload",
        "expression_file": Path(expression_name).name,
        "metadata_file": Path(metadata_name).name,
        "input_orientation": orientation,
        "expression_sha256": hashlib.sha256(expression).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "sample_column": sample_column,
        "donor_column": donor_column,
        "tissue_column": tissue_column,
        "extra_metadata_rows": len(meta) - len(samples),
    }
    return UploadAudit(
        values, tuple(gene_axis.astype(str)), samples, tuple(aligned[donor_column]), classes, labels, source
    )


def _expression_values(body: pd.DataFrame) -> np.ndarray:
    """Float32 values in the file's own orientation; only text columns take the slow path."""
    kinds = np.array([dtype.kind for dtype in body.dtypes])
    numeric_columns = np.isin(kinds, list("fiu"))
    values = np.empty(body.shape, dtype=np.float32)
    if numeric_columns.any():
        values[:, numeric_columns] = body.iloc[:, numeric_columns].to_numpy(dtype=np.float32, na_value=np.nan)
    if not numeric_columns.all():
        text = body.iloc[:, ~numeric_columns]
        flat = pd.Series(text.to_numpy(dtype=object).reshape(-1))
        blank = flat.isna() | flat.astype(str).str.strip().isin(BLANK_VALUES)
        numeric = pd.to_numeric(flat.mask(blank), errors="coerce")
        if (numeric.isna() & ~blank).any():
            raise ValueError("Nilai ekspresi nonnumerik ditemukan. Gunakan angka TPM atau sel kosong.")
        values[:, ~numeric_columns] = numeric.to_numpy(dtype=np.float32).reshape(text.shape)
    return values


# --- Saving ---


@operation
def save_training_upload(audit: UploadAudit, datasets_root: Path, candidate_count: int) -> Path:
    """Save a checked upload as a training dataset."""
    if not 1 <= candidate_count <= min(5000, len(audit.gene_names)):
        raise ValueError("Jumlah kandidat harus antara 1 dan jumlah gen input, maksimum 5.000.")
    # Include actual numeric values, not merely filenames, in the immutable identity.
    identity = stable_hash(
        {
            "source": audit.source,
            "genes": audit.gene_names,
            "samples": audit.sample_ids,
            "donors": audit.donor_ids,
            "classes": audit.classes,
            "candidate_count": candidate_count,
            "values": hashlib.sha256(audit.raw_tpm.tobytes()).hexdigest(),
            "labels": audit.labels.tolist(),
        }
    )
    datasets_root = Path(datasets_root)
    target = datasets_root / identity
    if target.is_dir():
        return target
    datasets_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="upload-", dir=datasets_root) as temporary:
        root = Path(temporary) / "data"
        root.mkdir()
        np.save(root / "raw_tpm.npy", audit.raw_tpm, allow_pickle=False)
        np.save(root / "labels.npy", audit.labels, allow_pickle=False)
        (root / "raw_genes.json").write_text(json.dumps(audit.gene_names), encoding="utf-8")
        manifest = {
            "schema_version": 2,
            "holdout_included": False,
            "candidate_count": candidate_count,
            "sample_ids": audit.sample_ids,
            "donor_ids": audit.donor_ids,
            "classes": audit.classes,
            "source": audit.source,
            "preprocessing_version": "user-training-v1",
            "cached_folds": [],
            "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn"]},
            "files": {p.name: file_hash(p) for p in root.iterdir()},
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        atomic_publish_directory(root, target)
    return target


def upload_preview(audit):
    """Wrap a checked upload so it can be split before saving."""
    # Splitting requires only sample/label/donor axes, never fitted gene values.
    return TrainingData(
        Path("."),
        {"cached_folds": []},
        audit.sample_ids,
        audit.donor_ids,
        audit.classes,
        audit.labels,
        audit.gene_names,
        audit.raw_tpm,
    )
