import io
import json
import numpy as np
import pandas as pd
import pytest

from test_training_core import prepared_fixture


def trained_fixture():
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector, fit_classifier
    prepared = prepared_fixture()
    prepared.X_validation = None
    prepared.y_validation = None
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
                                     'xgboost': {'n_estimators': 3}}, candidate_count=12)
    result = fit_classifier(prepared, fit_selector(prepared, config), config)
    genes = np.array(prepared.gene_names)[result.selector.selected_indices]
    demo = pd.DataFrame([[2., 3.]], index=['s0'], columns=genes)
    demo.index.name = 'SAMPID'
    return result, config, demo


def test_bundle_roundtrip_preserves_inference_and_checks_integrity(tmp_path):
    from gtex_explorer.model_registry import publish_model, load_saved_model
    from gtex_explorer.analysis.core import audit_and_align_input, predict_tissues, explain_prediction
    result, config, demo = trained_fixture()
    entry = publish_model(result, config, {'classes': ['A', 'B', 'C'], 'run_id': 'run_test'},
                          tmp_path / 'run_test', demo, None)
    preprocessor, classifier = load_saved_model(entry)
    aligned, audit = audit_and_align_input(demo, preprocessor)
    assert audit.ready
    predictions, probabilities = predict_tissues(aligned, preprocessor, classifier)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    explanation = explain_prediction(aligned, 's0', preprocessor, classifier)
    assert set(explanation.gene) == set(demo.columns)
    assert preprocessor['model_id'] == 'run_test'
    assert entry.demo_path.is_file()
    assert entry.manifest_hash
    assert '3 putaran' in entry.display_name
    assert 'run_test' in entry.display_name
    with (entry.bundle_dir / 'xgboost_model.json').open('ab') as handle:
        handle.write(b'bad')
    with pytest.raises(ValueError, match='hash'):
        load_saved_model(entry)


def test_partial_or_invalid_models_are_not_selectable(tmp_path):
    from gtex_explorer.model_registry import publish_model, list_models
    result, config, demo = trained_fixture()
    resource = tmp_path / 'resources'
    state = tmp_path / 'state'
    resource.mkdir()
    (state / 'models' / 'incomplete').mkdir(parents=True)
    assert list_models(resource, state) == []
    with pytest.raises(ValueError):
        publish_model(result, config, {'classes': ['A']}, state / 'models' / 'bad', demo, None)
    assert not (state / 'models' / 'bad').exists()
    publish_model(result, config, {'classes': ['A', 'B', 'C']}, state / 'models' / 'complete', demo, None)
    assert [entry.model_id for entry in list_models(resource, state)] == ['complete']
    bad = state / 'models' / 'malformed'
    bad.mkdir()
    for value in [None, 42, {'schema_version': 1, 'status': 'validated', 'model_id': 12}]:
        (bad / 'bundle_manifest.json').write_text(json.dumps(value))
        assert [entry.model_id for entry in list_models(resource, state)] == ['complete']


def test_bundle_rejects_path_escape_and_changed_manifest_identity(tmp_path):
    from gtex_explorer.model_registry import publish_model, load_saved_model
    result, config, demo = trained_fixture()
    entry = publish_model(result, config, {'classes': ['A', 'B', 'C']}, tmp_path / 'test', demo, None)
    path = entry.bundle_dir / 'bundle_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['files']['../secrets'] = 'hash'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_saved_model(entry)


def test_baseline_export_uses_every_candidate_and_development_example(tmp_path):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data, export_baseline
    from gtex_explorer.model_registry import load_saved_model
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    entry = export_baseline(tmp_path / 'data', tmp_path / 'baseline')
    pre, model = load_saved_model(entry)
    assert entry.model_id == 'original_baseline_5000'
    assert pre['feature_count'] == 6
    assert pre['feature_selection'] == 'baseline'
    assert model.get_booster().num_boosted_rounds() == 300
    assert 'HOLDOUT' not in pd.read_csv(entry.demo_path).SAMPID.tolist()
