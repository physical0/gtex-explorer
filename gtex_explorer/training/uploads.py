"""Audit user TPM/metadata and persist immutable, raw-only training datasets."""
from dataclasses import dataclass
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from gtex_explorer.training.data import atomic_publish_directory, file_hash, stable_hash, unique_ids
from gtex_explorer.cloud.state import operation


@dataclass
class UploadAudit:
    raw_tpm: np.ndarray
    gene_names: tuple
    sample_ids: tuple
    donor_ids: tuple
    classes: tuple
    labels: np.ndarray
    source: dict


def read_upload_table(content: bytes, filename: str) -> pd.DataFrame:
    if not content.strip():
        raise ValueError('Berkas kosong. Unggah matriks TPM atau metadata yang berisi data.')
    stream = io.BytesIO(content)
    first = stream.readline().decode('utf-8-sig').strip()
    skip = 2 if first in {'#1.2', '#1.3'} else 0
    stream.seek(0)
    sep = '\t' if skip or '\t' in first else ','
    # pandas renames duplicate column labels, so inspect the actual header first.
    lines = content.decode('utf-8-sig').splitlines()
    if len(lines) <= skip:
        raise ValueError('Header GCT tidak lengkap. Berkas harus memuat versi, dimensi, header, dan data.')
    header = lines[skip]
    columns = next(csv.reader([header], delimiter=sep))
    if len(set(columns)) != len(columns):
        raise ValueError('ID atau nama kolom duplikat pada header berkas.')
    frame = pd.read_csv(stream, sep=sep, skiprows=skip, dtype=str, keep_default_na=False)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError('Berkas harus memiliki baris data dan sedikitnya dua kolom.')
    return frame


def audit_uploads(expression: bytes, expression_name: str, metadata: bytes, metadata_name: str,
                  sample_column='SAMPID', donor_column='SUBJID', tissue_column='SMTSD') -> UploadAudit:
    meta = read_upload_table(metadata, metadata_name)
    required = [sample_column, donor_column, tissue_column]
    if len(set(required)) != 3 or not set(required).issubset(meta.columns):
        raise ValueError('Pilih tiga kolom metadata yang berbeda untuk ID sampel, donor, dan jaringan.')
    meta = meta[required].copy()
    for column, label in [(sample_column, 'sampel'), (donor_column, 'donor'), (tissue_column, 'jaringan')]:
        meta[column] = meta[column].str.strip()
        if meta[column].eq('').any():
            raise ValueError(f'ID atau label {label} kosong pada metadata.')
    if meta[sample_column].duplicated().any():
        raise ValueError('ID sampel duplikat pada metadata.')
    frame = read_upload_table(expression, expression_name)
    ids = set(meta[sample_column])
    first_values = frame.iloc[:, 0].str.strip()
    row_matches = first_values.isin(ids).any()
    col_matches = bool(set(frame.columns[1:]) & ids)
    if row_matches and col_matches:
        raise ValueError('Orientasi matriks ambigu. Gunakan ID sampel hanya pada satu sumbu.')
    if not row_matches and not col_matches:
        raise ValueError('ID sampel ekspresi tidak ditemukan pada metadata. Pencocokan ID harus persis sama.')
    if col_matches:
        if 'Description' in frame.columns:
            frame = frame.drop(columns='Description')
        matrix = frame.set_index(frame.columns[0]).T
        orientation = 'gen x sampel'
    else:
        matrix = frame.set_index(frame.columns[0])
        orientation = 'sampel x gen'
    try:
        samples = unique_ids(matrix.index.tolist())
        genes = unique_ids(matrix.columns.tolist(), genes=True)
    except ValueError as error:
        raise ValueError('ID sampel atau ID gen kosong/duplikat setelah suffix versi dihapus.') from error
    missing = set(samples) - ids
    if missing:
        raise ValueError(f'{len(missing)} ID sampel tidak memiliki metadata: {", ".join(sorted(missing)[:3])}.')
    # Convert on the short (sample) axis. This avoids 74,628 Python column conversions.
    flat = pd.Series(matrix.to_numpy().reshape(-1))
    blank = flat.astype(str).str.strip().isin(['', 'NA', 'NaN', 'nan', 'N/A'])
    numeric = pd.to_numeric(flat.mask(blank), errors='coerce')
    if (numeric.isna() & ~blank).any():
        raise ValueError('Nilai ekspresi nonnumerik ditemukan. Gunakan angka TPM atau sel kosong.')
    values = numeric.to_numpy(dtype=np.float32).reshape(matrix.shape)
    if np.isinf(values).any():
        raise ValueError('Nilai tak hingga ditemukan pada ekspresi TPM.')
    if (values < 0).any():
        raise ValueError('Nilai TPM negatif ditemukan. Gunakan ekspresi TPM sebelum transformasi.')
    if np.isnan(values).all(axis=1).any():
        raise ValueError('Ada sampel tanpa pengukuran ekspresi. Lengkapi data sampel tersebut.')
    aligned = meta.set_index(sample_column).loc[list(samples)]
    classes = tuple(sorted(aligned[tissue_column].unique()))
    if len(classes) < 2:
        raise ValueError('Training memerlukan sedikitnya dua kelas jaringan.')
    labels = aligned[tissue_column].map({name: i for i, name in enumerate(classes)}).to_numpy(dtype=np.int64)
    source = {'kind': 'user_upload', 'expression_file': Path(expression_name).name,
              'metadata_file': Path(metadata_name).name, 'input_orientation': orientation,
              'expression_sha256': hashlib.sha256(expression).hexdigest(),
              'metadata_sha256': hashlib.sha256(metadata).hexdigest(),
              'sample_column': sample_column, 'donor_column': donor_column, 'tissue_column': tissue_column,
              'extra_metadata_rows': len(meta) - len(samples)}
    return UploadAudit(values, tuple(matrix.columns.astype(str)), samples,
                       tuple(aligned[donor_column]), classes, labels, source)


@operation
def save_training_upload(audit: UploadAudit, datasets_root: Path, candidate_count: int) -> Path:
    if not 1 <= candidate_count <= min(5000, len(audit.gene_names)):
        raise ValueError('Jumlah kandidat harus antara 1 dan jumlah gen input, maksimum 5.000.')
    # Include actual numeric values, not merely filenames, in the immutable identity.
    identity = stable_hash({'source': audit.source, 'genes': audit.gene_names,
                            'samples': audit.sample_ids, 'donors': audit.donor_ids,
                            'classes': audit.classes, 'candidate_count': candidate_count,
                            'values': hashlib.sha256(audit.raw_tpm.tobytes()).hexdigest(),
                            'labels': audit.labels.tolist()})
    datasets_root = Path(datasets_root)
    target = datasets_root / identity
    if target.is_dir():
        return target
    datasets_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='upload-', dir=datasets_root) as temporary:
        root = Path(temporary) / 'data'
        root.mkdir()
        np.save(root / 'raw_tpm.npy', audit.raw_tpm, allow_pickle=False)
        np.save(root / 'labels.npy', audit.labels, allow_pickle=False)
        (root / 'raw_genes.json').write_text(json.dumps(audit.gene_names), encoding='utf-8')
        manifest = {'schema_version': 2, 'holdout_included': False, 'candidate_count': candidate_count,
                    'sample_ids': audit.sample_ids, 'donor_ids': audit.donor_ids, 'classes': audit.classes,
                    'source': audit.source, 'preprocessing_version': 'user-training-v1', 'cached_folds': [],
                    'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn']},
                    'files': {p.name: file_hash(p) for p in root.iterdir()}}
        (root / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        atomic_publish_directory(root, target)
    return target
