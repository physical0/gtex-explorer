"""Read an uploaded expression file into a sample-by-gene matrix."""

from __future__ import annotations

import io
from typing import BinaryIO

import numpy as np
import pandas as pd

from gtex_explorer.display import count


def strip_ensembl_version(gene_id: str) -> str:
    """Return the stable Ensembl identifier before its optional version suffix."""

    return str(gene_id).strip().split(".", 1)[0].upper()


def _read_delimited_bytes(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV, TSV, or a small GCT file without relying on its extension alone."""

    lower_name = filename.lower()
    text_start = raw_bytes[:200].decode("utf-8", errors="ignore")
    is_gct = lower_name.endswith(".gct") or text_start.lstrip().startswith("#1.2")
    separator = "\t" if lower_name.endswith((".tsv", ".txt", ".gct")) else ","
    skiprows = 2 if is_gct else 0
    return pd.read_csv(
        io.BytesIO(raw_bytes),
        sep=separator,
        skiprows=skiprows,
        # Identifiers stay text; value columns use the fast C float parser.
        dtype={0: str},
        low_memory=False,
    )


def parse_expression_file(file_object: BinaryIO, filename: str, *, model_genes=None) -> pd.DataFrame:
    """Convert sample-by-gene or gene-by-sample input into sample-by-gene form."""

    raw_bytes = file_object.read()
    try:
        table = _read_delimited_bytes(raw_bytes, filename)
    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "Berkas kosong atau tidak berisi tabel. Unggah berkas dengan kolom ID dan nilai ekspresi TPM."
        ) from error
    if table.empty or table.shape[1] < 2:
        raise ValueError("Berkas ekspresi kosong atau tidak memiliki kolom gen/sampel.")

    first_column = str(table.columns[0])
    if table.iloc[:, 0].isna().any() or table.iloc[:, 0].astype(str).str.strip().eq("").any():
        raise ValueError("Identifier gen/sampel pada kolom pertama tidak boleh kosong.")
    first_values = table.iloc[:, 0].astype(str).str.strip()
    gene_row_fraction = first_values.str.upper().str.startswith("ENSG").mean()
    model_ids = {strip_ensembl_version(gene) for gene in (model_genes or [])}
    row_matches = any(strip_ensembl_version(value) in model_ids for value in first_values)
    column_matches = any(strip_ensembl_version(value) in model_ids for value in table.columns[1:])
    if row_matches and column_matches:
        raise ValueError("Orientasi matriks ambigu. ID gen model ditemukan pada kedua sumbu.")

    if row_matches or (not column_matches and gene_row_fraction >= 0.5):
        # GCT commonly includes a Description column that is not an expression sample.
        non_expression_columns = {
            first_column,
            "Description",
            "description",
            "gene_name",
        }
        expression_columns = [column for column in table.columns if column not in non_expression_columns]
        # Convert before transposing: one text column would make the transposed frame all-object.
        matrix, invalid_count = _numeric_expression(table.set_index(first_column)[expression_columns])
        matrix = matrix.T
        matrix.index.name = "SAMPID"
        source_orientation = "gen × sampel"
    else:
        matrix, invalid_count = _numeric_expression(table.set_index(first_column))
        matrix.index.name = "SAMPID"
        source_orientation = "sampel × gen"

    matrix.index = matrix.index.astype(str).str.strip()
    matrix.columns = pd.Index([str(column).strip() for column in matrix.columns])
    if matrix.index.has_duplicates:
        duplicate_samples = matrix.index[matrix.index.duplicated()].unique().tolist()
        raise ValueError(f"SAMPID duplikat ditemukan: {', '.join(duplicate_samples[:5])}")

    if invalid_count:
        raise ValueError(
            f"Ditemukan {count(invalid_count)} nilai ekspresi non-numerik. Gunakan angka TPM atau nilai kosong, bukan teks."
        )
    matrix.attrs["source_orientation"] = source_orientation
    return matrix


def _numeric_expression(matrix):
    """Return a float matrix and the number of non-numeric cells.

    Columns the C parser already read as numbers are copied directly. Only text
    columns take the per-cell path, which costs roughly 30x the file size in memory.
    """
    kinds = np.array([dtype.kind for dtype in matrix.dtypes])
    numeric_columns = np.isin(kinds, list("fiu"))
    values = np.empty(matrix.shape, dtype=float)
    if numeric_columns.any():
        values[:, numeric_columns] = matrix.iloc[:, numeric_columns].to_numpy(dtype=float, na_value=np.nan)
    invalid = 0
    if not numeric_columns.all():
        text = matrix.iloc[:, ~numeric_columns]
        original = pd.Series(text.to_numpy(dtype=object).reshape(-1))
        original = original.mask(original.astype("string").str.fullmatch(r"\s*", na=False))
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = int((original.notna() & numeric.isna()).sum())
        values[:, ~numeric_columns] = numeric.to_numpy(dtype=float, na_value=np.nan).reshape(text.shape)
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns), invalid
