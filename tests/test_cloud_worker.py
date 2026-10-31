"""Train a real tiny model, replace local state, then verify saved CV and panels."""
from pathlib import Path

from test_cloud_state import MemoryStore
from test_training_data import checkpoint_sources


def test_training_artifacts_restore_after_instance_replacement(tmp_path, monkeypatch):
    import gtex_explorer.cloud.state as cloud_state
    from gtex_explorer.cloud.state import CloudState
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.worker import run_job
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.model_registry import list_models, load_saved_model
    from gtex_explorer.training.panels import load_run_panel

    root = tmp_path / 'cloud'
    store = MemoryStore()
    durable = CloudState(root, store)
    monkeypatch.setattr(cloud_state, '_instance', durable)
    monkeypatch.setenv('GTEX_CLOUD', '1')
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'web')
    monkeypatch.setenv('GTEX_CLOUD_LEASE', 'test-worker')
    durable.pull()
    data = root / 'datasets' / 'source'
    export_training_data(*checkpoint_sources(tmp_path), data)
    durable.publish()
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
        'evaluate_cv': True, 'cv_folds': 3, 'xgboost': {'n_estimators': 2, 'n_jobs': 1}}, candidate_count=6)
    record = start_job(config, data, root)
    assert record.status == 'queued'
    assert durable.acquire_lease('test-worker', ttl=120)
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'worker')
    assert run_job(record.run_id, data, root) == 0

    restored = CloudState(tmp_path / 'replacement', store)
    restored.pull()
    assert list_gallery(restored.root)[0]['run_id'] == record.run_id
    model = list_models(tmp_path / 'no-originals', restored.root)[0]
    pre, classifier = load_saved_model(model)
    assert pre['feature_count'] == 2
    assert classifier.get_booster().num_boosted_rounds() == 2
    for fold in [1, 2, 3]:
        genes, info = load_run_panel(restored.root, record.run_id, fold)
        assert len(genes) == 2
        assert info['training_samples'] > 0
    from gtex_explorer.training.jobs import resolve_job_data
    import json
    saved = json.loads((restored.root / 'runs' / record.run_id / 'config.json').read_text())
    assert resolve_job_data(saved, restored.root, data) == restored.root / 'datasets' / 'source'
