import io
import numpy as np
import pytest


def uploads():
    expression = b'gene,s1,s2,s3,s4,s5,s6\nOTHER_A,2,4,6,8,10,12\nOTHER_B,3,2,5,4,8,7\nOTHER_C,9,4,2,5,3,8\n'
    metadata = b'SAMPID,SUBJID,SMTSD\ns6,d6,B\ns5,d5,A\ns4,d4,B\ns3,d3,A\ns2,d2,B\ns1,d1,A\n'
    return expression, metadata


def test_uploaded_genes_labels_and_saved_final_are_independent_of_original_model(tmp_path):
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    from gtex_explorer.training.data import load_training_data, load_final_prepared
    expression, metadata = uploads()
    audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    assert audit.sample_ids == ('s1', 's2', 's3', 's4', 's5', 's6')
    assert audit.gene_names == ('OTHER_A', 'OTHER_B', 'OTHER_C')
    np.testing.assert_array_equal(audit.labels, [0, 1, 0, 1, 0, 1])
    root = save_training_upload(audit, tmp_path, candidate_count=2)
    data = load_training_data(root)
    assert data.manifest['source']['expression_file'] == 'own.csv'
    assert not (root / 'final').exists()  # No fitting merely to audit/load uploads.
    final = load_final_prepared(data)
    assert final.X_train.shape == (6, 2)
    assert final.X_validation is None
    np.testing.assert_allclose(final.X_train.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_array_equal(load_final_prepared(data).X_train, final.X_train)


def test_sample_by_gene_and_gene_by_sample_uploads_match():
    from gtex_explorer.training.uploads import audit_uploads
    expression, metadata = uploads()
    import pandas as pd
    transposed = pd.read_csv(io.BytesIO(expression), index_col=0).T.to_csv().encode()
    a = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    b = audit_uploads(transposed, 'transposed.csv', metadata, 'labels.csv')
    np.testing.assert_array_equal(a.raw_tpm, b.raw_tpm)
    assert a.gene_names == b.gene_names


@pytest.mark.parametrize('old,new,message', [
    (b'2,4,6', b'-2,4,6', 'negatif'),
    (b'2,4,6', b'inf,4,6', 'hingga'),
    (b'2,4,6', b'text,4,6', 'numerik'),
    (b'OTHER_B', b'OTHER_A', 'duplikat'),
    (b's2,s3', b's1,s3', 'duplikat'),
])
def test_invalid_uploads_have_specific_errors(old, new, message):
    from gtex_explorer.training.uploads import audit_uploads
    expression, metadata = uploads()
    with pytest.raises(ValueError, match=message):
        audit_uploads(expression.replace(old, new), 'own.csv', metadata, 'labels.csv')


def test_missing_metadata_and_empty_donors_are_rejected():
    from gtex_explorer.training.uploads import audit_uploads
    expression, metadata = uploads()
    for invalid, message in [(metadata.replace(b's1,d1,A\n', b''), 'metadata'),
                             (metadata.replace(b's1,d1,A', b's1,,A'), 'donor')]:
        with pytest.raises(ValueError, match=message):
            audit_uploads(expression, 'own.csv', invalid, 'labels.csv')


def test_partition_constants_ignore_validation_values(tmp_path):
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    from gtex_explorer.training.data import load_training_data, prepare_partition, FoldAssignment
    expression, metadata = uploads()
    audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    root = save_training_upload(audit, tmp_path, candidate_count=2)
    data = load_training_data(root)
    assignment = FoldAssignment(1, np.arange(4), np.array([4, 5]), 'split')
    a = prepare_partition(data, assignment, tmp_path / 'cache')
    audit.raw_tpm[4:] = 1e8
    changed = load_training_data(save_training_upload(audit, tmp_path, candidate_count=2))
    b = prepare_partition(changed, assignment, tmp_path / 'cache')
    np.testing.assert_array_equal(a.X_train, b.X_train)
    np.testing.assert_array_equal(a.scaler_mean, b.scaler_mean)
    np.testing.assert_array_equal(a.raw_tpm_medians, b.raw_tpm_medians)
    assert not np.array_equal(a.X_validation, b.X_validation)


@pytest.mark.parametrize('content', [b'#1.2\n', b'#1.2\n3\t2\n'])
def test_truncated_gct_has_descriptive_error(content):
    from gtex_explorer.training.uploads import read_upload_table
    with pytest.raises(ValueError, match='GCT'):
        read_upload_table(content, 'truncated.gct')


def test_custom_gene_prediction_keeps_gene_by_sample_orientation():
    from gtex_explorer.inference.parsing import parse_expression_file
    expression, _ = uploads()
    matrix = parse_expression_file(io.BytesIO(expression), 'own.csv', model_genes=['OTHER_A', 'OTHER_C'])
    assert matrix.shape == (6, 3)
    assert matrix.index.tolist() == [f's{i}' for i in range(1, 7)]
