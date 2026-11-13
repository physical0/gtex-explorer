"""The log2(TPM+1) and standardization choices: same candidate genes, honest features, and repeated at prediction."""
import json

import numpy as np
import pandas as pd
import pytest

from gtex_explorer.training.config import configuration_fingerprint, validate_training_config
from gtex_explorer.training.run_labels import method_description, preprocessing_label


def test_default_transforms_keep_saved_configs_and_identities_unchanged():
    implicit = validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=6)
    explicit = validate_training_config({'selector': 'anova', 'top_k': 2, 'log2_transform': True,
                                         'standardize': True}, candidate_count=6)
    assert 'log2_transform' not in implicit.to_dict() and 'standardize' not in implicit.to_dict()
    assert implicit.to_dict() == explicit.to_dict()
    fingerprint = lambda config: configuration_fingerprint(config, 'data', 'split', {'numpy': '1'})
    assert fingerprint(implicit) == fingerprint(explicit)
    raw = validate_training_config({'selector': 'anova', 'top_k': 2, 'log2_transform': False}, candidate_count=6)
    assert raw.to_dict()['log2_transform'] is False and 'standardize' not in raw.to_dict()
    assert fingerprint(raw) != fingerprint(implicit)
    # A saved config reads back to the same settings.
    assert validate_training_config(raw.to_dict(), candidate_count=6) == raw
    with pytest.raises(ValueError, match='boolean'):
        validate_training_config({'standardize': 'no'}, candidate_count=6)


def test_labels_name_only_non_default_transforms():
    base = {'selector': 'anova', 'top_k': 1000}
    assert method_description(base) == 'ANOVA - 1.000 gen'
    assert method_description({**base, 'standardize': True, 'log2_transform': True}) == 'ANOVA - 1.000 gen'
    assert method_description({**base, 'log2_transform': False}) == 'ANOVA - 1.000 gen - TPM + standardisasi'
    assert preprocessing_label({'standardize': False}) == 'log2(TPM+1) tanpa standardisasi'
    assert preprocessing_label({'log2_transform': False, 'standardize': False}) == 'TPM tanpa transformasi'


@pytest.fixture
def uploaded(tmp_path):
    from test_training_uploads import uploads
    from gtex_explorer.training.data import load_training_data
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    expression, metadata = uploads()
    audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    return load_training_data(save_training_upload(audit, tmp_path / 'datasets', candidate_count=2))


def test_each_transform_changes_only_the_features_and_uses_training_statistics(uploaded, tmp_path):
    from gtex_explorer.training.data import FoldAssignment, prepare_partition
    train, validation = np.array([0, 1, 2, 3]), np.array([4, 5])
    split = FoldAssignment(1, train, validation, 'split-1')
    prepared = {
        (log2, standardize): prepare_partition(uploaded, split, tmp_path / 'cache', log2=log2, standardize=standardize)
        for log2 in (True, False) for standardize in (True, False)
    }
    default = prepared[True, True]
    columns = [list(uploaded.raw_gene_names).index(gene) for gene in default.gene_names]
    raw_train = uploaded.raw_tpm[train][:, columns].astype(float)
    raw_validation = uploaded.raw_tpm[validation][:, columns].astype(float)
    for result in prepared.values():
        # Same candidate genes and the same log2 reference matrix whatever the transform.
        assert result.gene_names == default.gene_names
        np.testing.assert_allclose(result.training_log, np.log2(raw_train + 1))
    assert len({result.fingerprint for result in prepared.values()}) == 4

    raw = prepared[False, False]
    np.testing.assert_allclose(raw.X_train, raw_train, rtol=1e-6)
    np.testing.assert_allclose(raw.X_validation, raw_validation, rtol=1e-6)
    np.testing.assert_array_equal(raw.scaler_mean, 0)
    np.testing.assert_array_equal(raw.scaler_scale, 1)

    log_only = prepared[True, False]
    np.testing.assert_allclose(log_only.X_train, np.log2(raw_train + 1), rtol=1e-6)

    scaled = prepared[False, True]
    mean, scale = raw_train.mean(axis=0), raw_train.std(axis=0)
    np.testing.assert_allclose(scaled.X_train, (raw_train - mean) / scale, rtol=1e-5, atol=1e-6)
    # Validation is scaled with the training part's statistics, never its own.
    np.testing.assert_allclose(scaled.X_validation, (raw_validation - mean) / scale, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(scaled.scaler_mean, mean)

    log_train = np.log2(raw_train + 1)
    np.testing.assert_allclose(default.X_train, (log_train - log_train.mean(axis=0)) / log_train.std(axis=0),
                               rtol=1e-5, atol=1e-6)
    # Read back from the cache unchanged.
    again = prepare_partition(uploaded, split, tmp_path / 'cache', log2=False, standardize=True)
    np.testing.assert_array_equal(again.X_train, scaled.X_train)


def test_notebook_data_keeps_its_candidate_genes_when_rebuilt_from_raw_tpm(tmp_path):
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.data import load_final_prepared, load_training_data
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    data = load_training_data(tmp_path / 'data')
    notebook = load_final_prepared(data)
    raw = load_final_prepared(data, log2=False, standardize=False, cache_dir=tmp_path / 'cache')
    assert raw.gene_names == notebook.gene_names == tuple(data.manifest['final']['genes'])
    columns = [list(data.raw_gene_names).index(gene) for gene in raw.gene_names]
    np.testing.assert_allclose(raw.X_train, data.raw_tpm[:, columns], rtol=1e-6)
    np.testing.assert_allclose(raw.training_log, notebook.training_log, rtol=1e-5)
    # Nothing is written next to the bundled notebook data.
    assert not (tmp_path / 'data' / 'preprocessing_cache').exists()


def test_run_without_transforms_trains_saves_and_predicts_on_raw_tpm(tmp_path, monkeypatch):
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.inference.prediction import transform_aligned_tpm
    from gtex_explorer.platform.model_registry import list_models, load_saved_model
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.worker import run_job
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'evaluate_cv': True, 'cv_folds': 3,
                                       'log2_transform': False, 'standardize': False,
                                       'xgboost': {'n_estimators': 2}}, candidate_count=6)
    run_id = start_job(config, data, state).run_id
    assert run_job(run_id, data, state) == 0
    assert [entry['run_id'] for entry in list_gallery(state)] == [run_id]
    saved = json.loads((state / 'runs' / run_id / 'config.json').read_text())
    assert saved['config']['log2_transform'] is False and saved['config']['standardize'] is False

    entry = next(model for model in list_models(tmp_path / 'resources', state) if model.model_id == run_id)
    pre, classifier = load_saved_model(entry)
    assert pre['log2_transform'] is False and pre['standardize'] is False
    np.testing.assert_array_equal(pre['scaler_mean'], 0)
    np.testing.assert_array_equal(pre['scaler_scale'], 1)
    demo = pd.read_csv(state / 'models' / run_id / 'demo_tpm.csv', index_col=0)[pre['selected_genes']]
    features = transform_aligned_tpm(demo, pre)
    np.testing.assert_allclose(features, demo.to_numpy(), rtol=1e-5)
    assert classifier.predict_proba(features).shape == (len(demo), len(pre['classes']))


def test_prediction_applies_log2_for_older_models_and_skips_it_when_trained_without():
    from gtex_explorer.inference.prediction import transform_aligned_tpm
    tpm = pd.DataFrame([[0.0, 3.0], [7.0, 1.0]], columns=['g1', 'g2'])
    older = {'scaler_mean': np.zeros(2), 'scaler_scale': np.ones(2)}
    np.testing.assert_allclose(transform_aligned_tpm(tpm, older), np.log2(tpm.to_numpy() + 1))
    np.testing.assert_allclose(transform_aligned_tpm(tpm, {**older, 'log2_transform': False}), tpm.to_numpy())
    with pytest.raises(ValueError, match='tidak boleh negatif'):
        transform_aligned_tpm(-tpm, {**older, 'log2_transform': False})


def test_form_offers_both_transforms_on_by_default_and_reviews_the_choice(tmp_path, monkeypatch):
    from test_training_uploads import uploads
    from test_training_workflow_ui import setup_app
    from gtex_explorer.training.uploads import audit_uploads
    app = setup_app(tmp_path, monkeypatch)
    expression, metadata = uploads()
    app.session_state.training_upload_audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    app.run()
    app.button(key='training_to_configuration').click().run()
    assert app.checkbox(key='training_log2').value is True
    assert app.checkbox(key='training_standardize').value is True
    headings = [item.value for item in app.subheader]
    assert headings.index('Pra-pemrosesan') < headings.index('Seleksi fitur')
    app.checkbox(key='training_log2').uncheck().run()
    app.selectbox(key='training_selector').set_value('anova').run()
    app.number_input(key='training_top_k').set_value(2).run()
    app.button(key='training_to_review').click().run()
    assert not app.exception
    review = next(table.value for table in app.table if 'Pra-pemrosesan' in table.value.index)
    assert review.loc['Pra-pemrosesan', 'Nilai'] == 'TPM + standardisasi'
    assert not app.button(key='training_start').disabled
    # The choice survives a visit to another page.
    app.button(key='nav_Kamus Biologi').click().run()
    app.button(key='nav_Pelatihan Model').click().run()
    assert app.session_state.training_log2 is False


def test_comparison_lists_preprocessing_as_a_setting():
    from gtex_explorer.training.comparison import settings_frame
    from test_training_comparison import model
    a, b = model('a'), model('b')
    b['config']['standardize'] = False
    frame = settings_frame([a, b]).set_index('Pengaturan')
    assert frame.loc['Pra-pemrosesan', 'differs']
    assert frame.loc['Pra-pemrosesan', 'B'] == 'log2(TPM+1) tanpa standardisasi'
