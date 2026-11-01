from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
import gtex_explorer.analysis.core as core

from gtex_explorer.analysis.core import (
    audit_and_align_input,
    load_reference_tables,
    parse_expression_file,
    strip_ensembl_version,
)


def test_strip_ensembl_version() -> None:
    assert strip_ensembl_version("ENSG00000170549.4") == "ENSG00000170549"


def test_parse_sample_by_gene_csv() -> None:
    raw = b"SAMPID,ENSG000001.1,ENSG000002.2\nS1,1.0,2.0\n"
    matrix = parse_expression_file(io.BytesIO(raw), "sample.csv")
    assert matrix.shape == (1, 2)
    assert matrix.index.tolist() == ["S1"]


def test_parse_gene_by_sample_tsv() -> None:
    raw = b"Name\tDescription\tS1\tS2\nENSG000001.1\tA\t1\t2\n"
    matrix = parse_expression_file(io.BytesIO(raw), "sample.tsv")
    assert matrix.shape == (2, 1)
    assert matrix.columns.tolist() == ["ENSG000001.1"]


def test_audit_matches_versionless_ids() -> None:
    preprocessor = {
        "selected_genes": ["ENSG000001.1", "ENSG000002.5"],
        "raw_tpm_medians": np.asarray([3.0, 4.0]),
    }
    matrix = pd.DataFrame(
        [[1.0, 2.0]],
        index=["S1"],
        columns=["ENSG000001.9", "ENSG000002"],
    )
    aligned, audit = audit_and_align_input(
        matrix,
        preprocessor,
    )
    assert audit.ready
    assert audit.matched_gene_count == 2
    assert aligned.loc["S1", "ENSG000001.1"] == 1.0


def test_audit_imputes_missing_values_with_training_median() -> None:
    preprocessor = {
        "selected_genes": [f"ENSG{i:06d}.1" for i in range(1, 11)],
        "raw_tpm_medians": np.full(10, 3.5),
    }
    matrix = pd.DataFrame([[np.nan, *([1.0] * 9)]], index=["S1"], columns=preprocessor["selected_genes"])
    aligned, audit = audit_and_align_input(
        matrix,
        preprocessor,
    )
    assert audit.ready
    assert audit.missing_value_count == 1
    assert aligned.loc["S1", "ENSG000001.1"] == 3.5
    assert audit.sample_quality.iloc[0]["observed_fraction"] == 0.9


@pytest.mark.parametrize("value", [np.nan, 0.0])
def test_audit_allows_missing_or_zero_samples(value) -> None:
    preprocessor = {"selected_genes": ["ENSG1.1", "ENSG2.1"], "raw_tpm_medians": np.array([3., 4.])}
    matrix = pd.DataFrame([[value, value]], index=["S1"], columns=preprocessor["selected_genes"])
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    assert aligned.attrs["input_audit_ready"] is True
    np.testing.assert_allclose(aligned.iloc[0], [3., 4.] if np.isnan(value) else [0., 0.])
    assert audit.below_threshold_sample_count == (1 if np.isnan(value) else 0)


def test_audit_checks_each_sample_not_batch_average() -> None:
    genes = [f"ENSG{i}" for i in range(10)]
    preprocessor = {"selected_genes": genes, "raw_tpm_medians": np.ones(10)}
    matrix = pd.DataFrame([[1.] * 10, [np.nan] * 2 + [1.] * 8], index=["good", "bad"], columns=genes)
    _, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    assert audit.below_threshold_sample_count == 1
    assert "bad" in audit.message


def test_nullable_integer_missing_value_accepts_fractional_median() -> None:
    genes = [f"ENSG{i}" for i in range(10)]
    preprocessor = {"selected_genes": genes, "raw_tpm_medians": np.full(10, 3.5)}
    matrix = pd.DataFrame([[pd.NA] + [1] * 9], columns=genes, index=["S1"], dtype="Int64")
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    assert aligned.iloc[0, 0] == 3.5


def test_audit_zero_matching_genes_uses_medians_without_blocking() -> None:
    preprocessor = {"selected_genes": ["ENSG1", "ENSG2"], "raw_tpm_medians": [3.5, 4.5]}
    matrix = pd.DataFrame([[8.]], index=["S1"], columns=["ENSG_OTHER"])
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    assert audit.matched_gene_count == 0
    assert audit.missing_gene_count == 2
    assert audit.below_threshold_sample_count == 1
    np.testing.assert_allclose(aligned.iloc[0], [3.5, 4.5])
    assert "median" in audit.message


@pytest.mark.parametrize("threshold,below", [(0., 0), (.8, 0), (.9, 1), (1., 1)])
def test_completeness_threshold_changes_reminder_only(threshold, below) -> None:
    genes = [f"ENSG{i}" for i in range(10)]
    preprocessor = {"selected_genes": genes, "raw_tpm_medians": np.full(10, 3.5)}
    matrix = pd.DataFrame([[0.] * 8 + [np.nan] * 2], columns=genes, index=["S1"])
    aligned, audit = audit_and_align_input(matrix, preprocessor, minimum_observed_fraction=threshold)
    assert audit.ready
    assert audit.below_threshold_sample_count == below
    assert audit.completeness_threshold == threshold
    np.testing.assert_allclose(aligned.iloc[0], [0.] * 8 + [3.5] * 2)


@pytest.mark.parametrize("threshold", [-.1, 1.1, np.nan, np.inf, True, "90"])
def test_audit_rejects_invalid_reminder_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="ambang"):
        audit_and_align_input(pd.DataFrame([[1.]], columns=["ENSG1"]),
                              {"selected_genes": ["ENSG1"], "raw_tpm_medians": [1.]},
                              minimum_observed_fraction=threshold)


def test_audit_rejects_nonnumeric_measurements() -> None:
    _, audit = audit_and_align_input(pd.DataFrame([["abc"]], columns=["ENSG1"]),
                                    {"selected_genes": ["ENSG1"], "raw_tpm_medians": [1.]})
    assert not audit.ready
    assert audit.invalid_numeric_value_count == 1


def test_zero_tpm_still_uses_frozen_log_and_scaling() -> None:
    values = core.transform_aligned_tpm(
        pd.DataFrame([[0., 0.]], columns=["ENSG1", "ENSG2"]),
        {"scaler_mean": [2., 3.], "scaler_scale": [2., 6.]},
    )
    np.testing.assert_allclose(values, [[-1., -.5]])


@pytest.mark.parametrize("threshold,below", [(0., 0), (.5, 0), (.9, 1), (1., 1)])
def test_positive_expression_reminder_is_independent_of_model_match(threshold, below) -> None:
    preprocessor = {"selected_genes": ["ENSG1", "ENSG2"], "raw_tpm_medians": [3.5, 4.5]}
    # No model IDs match, but half the uploaded genes have positive measured TPM.
    matrix = pd.DataFrame([[2., 0., 5., np.nan]], columns=["OTHER1", "OTHER2", "OTHER3", "OTHER4"])
    aligned, audit = audit_and_align_input(matrix, preprocessor, minimum_positive_fraction=threshold)
    assert audit.ready
    assert audit.matched_fraction == 0
    assert audit.positive_expression_threshold == threshold
    assert audit.low_positive_sample_count == below
    assert audit.sample_quality.iloc[0]["uploaded_positive_genes"] == 2
    assert audit.sample_quality.iloc[0]["uploaded_positive_fraction"] == .5
    np.testing.assert_allclose(aligned.iloc[0], [3.5, 4.5])


@pytest.mark.parametrize("threshold", [-.1, 1.1, np.nan, np.inf, True, "90"])
def test_audit_rejects_invalid_positive_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="ambang"):
        audit_and_align_input(pd.DataFrame([[1.]], columns=["ENSG1"]),
                              {"selected_genes": ["ENSG1"], "raw_tpm_medians": [1.]},
                              minimum_positive_fraction=threshold)


def test_audit_missing_columns_and_values_share_imputation_budget() -> None:
    genes = [f"ENSG{i}" for i in range(10)]
    preprocessor = {"selected_genes": genes, "raw_tpm_medians": np.ones(10)}
    matrix = pd.DataFrame([[np.nan] + [1.] * 8], index=["S1"], columns=genes[:9])
    _, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.matched_fraction == 0.9
    assert audit.ready  # Missing measurements are advisory, including the combined 20%.
    assert audit.below_threshold_sample_count == 1
    assert audit.sample_quality.iloc[0]["imputed_model_genes"] == 2


@pytest.mark.parametrize("filename,raw", [
    ("bad.csv", b"SAMPID,ENSG1\nS1,abc\n"),
    ("bad.tsv", b"Name\tDescription\tS1\nENSG1\tGeneA\tabc\n"),
    ("bad.gct", b"#1.2\n1\t1\nName\tDescription\tS1\nENSG1\tGeneA\tabc\n"),
])
def test_parser_rejects_nonnumeric_expression(filename, raw) -> None:
    with pytest.raises(ValueError, match="non-numerik"):
        parse_expression_file(io.BytesIO(raw), filename)


def test_parser_rejects_blank_identifier() -> None:
    with pytest.raises(ValueError, match="tidak boleh kosong"):
        parse_expression_file(io.BytesIO(b"SAMPID,ENSG1\n,1\n"), "bad.csv")


def test_final_overview_matches_part_a_export() -> None:
    tables = load_reference_tables()
    assert tables["final_test"]["classifier"].tolist() == ["xgboost"]
    canonical = APP_DIR.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_only/final_test_summary.csv"
    pd.testing.assert_frame_equal(tables["final_test"], pd.read_csv(canonical))


@pytest.mark.parametrize("defect", ["dummy_row", "wrong_selector"])
def test_reference_loader_rejects_mixed_or_stale_final_export(monkeypatch, defect) -> None:
    original_reader = core.pd.read_csv
    def altered_reader(path, *args, **kwargs):
        frame = original_reader(path, *args, **kwargs)
        if Path(path).name == "final_test_summary.csv":
            if defect == "dummy_row":
                frame = pd.concat([frame, frame.assign(classifier="dummy")], ignore_index=True)
            else:
                frame["selector_id"] = "old_selector"
        return frame
    monkeypatch.setattr(core.pd, "read_csv", altered_reader)
    with pytest.raises(ValueError):
        load_reference_tables()


def test_audit_rejects_infinite_values() -> None:
    preprocessor = {
        "selected_genes": ["ENSG000001.1"],
        "raw_tpm_medians": np.asarray([3.5]),
    }
    matrix = pd.DataFrame([[np.inf]], index=["S1"], columns=["ENSG000001.9"])
    _, audit = audit_and_align_input(
        matrix,
        preprocessor,
    )
    assert not audit.ready
    assert audit.infinite_value_count == 1
    assert "tak hingga" in audit.message


def test_audit_rejects_duplicate_versionless_gene_ids() -> None:
    preprocessor = {
        "selected_genes": ["ENSG000001.1"],
        "raw_tpm_medians": np.asarray([3.5]),
    }
    matrix = pd.DataFrame(
        [[1.0, 2.0]],
        index=["S1"],
        columns=["ENSG000001.8", "ENSG000001.9"],
    )
    _, audit = audit_and_align_input(
        matrix,
        preprocessor,
    )
    assert not audit.ready
    assert audit.duplicate_gene_count == 1


def test_parse_rejects_duplicate_sample_ids() -> None:
    raw = b"SAMPID,ENSG000001.1\nS1,1.0\nS1,2.0\n"
    with pytest.raises(ValueError, match="SAMPID duplikat"):
        parse_expression_file(io.BytesIO(raw), "sample.csv")


def test_reference_tables_include_validated_all_gene_baseline() -> None:
    tables = load_reference_tables()
    folds = tables["cv_folds"]
    summary = tables["cv_summary"]
    validation = tables["experiment_validation"]

    assert validation["status"] == "passed"
    assert len(folds) == 80
    assert folds["configuration_id"].nunique() == 16
    assert len(summary) == 16

    baseline = folds.loc[
        folds["configuration_id"].eq("xgboost_all_5000")
    ]
    assert len(baseline) == 5
    assert set(baseline["fold"].astype(int)) == {1, 2, 3, 4, 5}
    assert baseline["genes"].eq(5000).all()
    assert (~baseline["uses_feature_selection"].astype(bool)).all()


def test_parse_keeps_sample_ids_as_written_and_reads_text_cells_slowly() -> None:
    # IDs such as 001 must not be read as numbers, and one text column must still be reported.
    raw = b"SAMPID,ENSG000001,ENSG000002\n001,1.5, \n002,2,3\n"
    matrix = parse_expression_file(io.BytesIO(raw), "sample.csv")
    assert matrix.index.tolist() == ["001", "002"]
    assert np.isnan(matrix.loc["001", "ENSG000002"])
    with pytest.raises(ValueError, match="1 nilai ekspresi non-numerik"):
        parse_expression_file(io.BytesIO(b"SAMPID,ENSG000001\nS1,abc\nS2,2\n"), "sample.csv")
