"""A saved model is not a gallery entry until every result is verified."""
import json

import pytest


def make_run(tmp_path, monkeypatch, *, cv=False):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
        'evaluate_cv': cv, 'cv_folds': 3, 'xgboost': {'n_estimators': 2}}, candidate_count=6)
    job = start_job(config, data, state)
    return data, state, job.run_id


def test_bundle_is_hidden_until_saved_results_verified_and_completed(tmp_path, monkeypatch):
    import gtex_explorer.training.worker as worker
    from gtex_explorer.model_registry import list_models
    from gtex_explorer.training.gallery import list_gallery
    data, state, run_id = make_run(tmp_path, monkeypatch)
    publish = worker.publish_model
    def observe_publication(*args, **kwargs):
        result = publish(*args, **kwargs)
        assert list_gallery(state) == []
        assert list_models(tmp_path / 'resources', state) == []
        return result
    monkeypatch.setattr(worker, 'publish_model', observe_publication)
    assert worker.run_job(run_id, data, state) == 0
    assert [entry['run_id'] for entry in list_gallery(state)] == [run_id]
    receipt = json.loads((state / 'runs' / run_id / 'gallery.json').read_text())
    assert receipt['cv_summary'] is None
    assert receipt['feature_count'] == 2
    assert [entry.model_id for entry in list_models(tmp_path / 'resources', state)] == [run_id]


def test_missing_cv_panel_blocks_gallery_and_prediction_and_retry_reuses_model(tmp_path, monkeypatch):
    import gtex_explorer.training.worker as worker
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.model_registry import list_models
    data, state, run_id = make_run(tmp_path, monkeypatch, cv=True)
    publish = worker.publish_model
    panel_bytes = []
    panel = state / 'runs' / run_id / 'fold_2_panel' / 'genes.csv'
    def damage_panel(*args, **kwargs):
        entry = publish(*args, **kwargs)
        panel_bytes.append(panel.read_bytes())
        panel.unlink()
        return entry
    monkeypatch.setattr(worker, 'publish_model', damage_panel)
    assert worker.run_job(run_id, data, state) == 1
    assert list_gallery(state) == []
    assert list_models(tmp_path / 'resources', state) == []
    panel.write_bytes(panel_bytes[0])
    monkeypatch.setattr(worker, 'fit_classifier', lambda *args: pytest.fail('Retry must reuse saved model'))
    assert worker.run_job(run_id, data, state) == 0
    assert len(list_gallery(state)) == 1
    assert list_gallery(state)[0]['cv_summary']['fold_count'] == 3


def test_completed_legacy_run_is_verified_before_indexing_and_corruption_is_reported(tmp_path, monkeypatch):
    from gtex_explorer.training.worker import run_job
    from gtex_explorer.training.gallery import list_gallery
    data, state, run_id = make_run(tmp_path, monkeypatch, cv=True)
    assert run_job(run_id, data, state) == 0
    receipt = state / 'runs' / run_id / 'gallery.json'
    receipt.unlink()
    assert len(list_gallery(state)) == 1
    assert receipt.exists()
    (state / 'runs' / run_id / 'fold_1_panel' / 'genes.csv').write_text('corrupt')
    issues = []
    assert list_gallery(state, issues=issues) == []
    assert issues and issues[0]['run_id'] == run_id


@pytest.mark.parametrize('status', ['queued', 'running', 'paused', 'cancelled', 'failed'])
def test_unfinished_status_cannot_expose_even_a_valid_receipt(tmp_path, monkeypatch, status):
    from gtex_explorer.training.worker import run_job
    from gtex_explorer.training.jobs import atomic_json
    from gtex_explorer.training.gallery import list_gallery
    data, state, run_id = make_run(tmp_path, monkeypatch)
    assert run_job(run_id, data, state) == 0
    atomic_json(state / 'runs' / run_id / 'status.json', {'status': status})
    assert list_gallery(state) == []


def test_gallery_and_process_history_are_separate_ui_views(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from gtex_explorer.training.worker import run_job
    data, state, ready = make_run(tmp_path, monkeypatch)
    assert run_job(ready, data, state) == 0
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    pending = start_job(validate_training_config({'selector': 'anova', 'top_k': 2,
        'xgboost': {'n_estimators': 2}}, candidate_count=6), data, state).run_id
    script = ('from pathlib import Path\n'
        'from gtex_explorer.ui.training_page import render_training_results_page\n'
        f'render_training_results_page(Path({str(data)!r}), Path({str(state)!r}))')
    app = AppTest.from_string(script, default_timeout=60).run()
    assert not app.exception
    assert app.session_state.training_results_view == 'Gallery model'
    assert app.button(key=f'gallery_select_{ready}')
    assert not any(button.key == f'gallery_select_{pending}' for button in app.button)
    # AppTest does not expose a tab-select action; start a second view with that tab active.
    app = AppTest.from_string(script, default_timeout=60)
    app.session_state.training_results_view = 'Status proses'
    app.session_state.training_selected_run = pending
    app.run()
    assert app.selectbox(key='training_run').value == pending
    assert not app.exception
    assert app.button(key='training_cancel').label == 'Stop'
    assert not any(button.key == 'training_open_prediction' for button in app.button)


@pytest.mark.parametrize('stage', ['saving_results', 'verifying_results'])
def test_final_saving_update_cannot_be_paused_or_stopped_halfway(tmp_path, stage):
    import time
    from gtex_explorer.training.jobs import atomic_json, job_path, process_lock, request_cancel, pause_job
    run_id = 'run_' + 'a' * 32
    root = job_path(run_id, tmp_path)
    atomic_json(root / 'status.json', {'status': 'running', 'stage': stage,
                                     'updated': time.time(), 'error': None})
    with process_lock(tmp_path / 'worker.lock'):
        assert request_cancel(run_id, tmp_path).status == 'running'
        assert not (root / 'cancel.json').exists()
        with pytest.raises(ValueError, match='penyimpanan'):
            pause_job(run_id, tmp_path)


@pytest.mark.parametrize('stage', ['saving_results', 'verifying_results'])
def test_saving_stage_explains_gallery_delay_and_disables_controls_in_ui(tmp_path, stage):
    import time
    from streamlit.testing.v1 import AppTest
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import atomic_json, job_path, process_lock
    root = job_path('run_' + 'b' * 32, tmp_path)
    atomic_json(root / 'config.json', {'config': validate_training_config({'selector':'anova','top_k':2}).to_dict()})
    atomic_json(root / 'status.json', {'status':'running','stage':stage,'updated':time.time(),'error':None})
    atomic_json(root / 'worker.json', {})
    with process_lock(tmp_path / 'worker.lock'):
        app = AppTest.from_string('from pathlib import Path\n'
            'from gtex_explorer.ui.training_page import render_training_results_page\n'
            f'render_training_results_page(Path("unused"), Path({str(tmp_path)!r}))',
            default_timeout=60).run()
        assert not app.exception
        assert app.button(key='training_pause').disabled
        assert app.button(key='training_cancel').disabled
        assert any('disimpan dan diverifikasi' in info.value for info in app.info)
        app.session_state.training_results_view = 'Gallery model'
        app.run()
        assert not any(button.key == 'training_open_prediction' for button in app.button)
        assert any('Belum ada model terverifikasi' in info.value for info in app.info)


def test_cached_verification_skips_rehash_until_a_file_changes(tmp_path, monkeypatch):
    import pytest
    import gtex_explorer.training.data as data
    target = tmp_path / 'artifact.txt'
    target.write_bytes(b'one')
    expected = data.file_hash(target)
    data.verify_files_cached(tmp_path, {'artifact.txt': expected})
    calls = []
    real = data.file_hash
    monkeypatch.setattr(data, 'file_hash', lambda path: calls.append(path) or real(path))
    data.verify_files_cached(tmp_path, {'artifact.txt': expected})
    assert not calls
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        data.verify_files_cached(tmp_path, {'artifact.txt': expected})
    assert calls


def test_cv_run_saves_tissue_scores_and_older_runs_still_verify(tmp_path, monkeypatch):
    from gtex_explorer.training.data import file_hash, stable_hash
    from gtex_explorer.training.gallery import finalize_gallery
    from gtex_explorer.training.worker import run_job
    data, state, run_id = make_run(tmp_path, monkeypatch, cv=True)
    assert run_job(run_id, data, state) == 0
    root, bundle = state / 'runs' / run_id, state / 'models' / run_id
    summary = json.loads((bundle / 'cv_summary.json').read_text(encoding='utf-8'))
    for row in summary['fold_results']:
        assert row['tissue_f1'].keys() == {item['tissue'] for item in summary['tissue_f1']}
        assert sum(row['tissue_f1'].values()) / len(row['tissue_f1']) == pytest.approx(row['macro_f1'])

    # A run from before this field existed has neither the fold scores nor the summary entry.
    for fold in range(1, 4):
        path = root / f'fold_{fold}.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        del payload['result']['tissue_f1']
        payload['checksum'] = stable_hash(payload['result'])
        path.write_text(json.dumps(payload), encoding='utf-8')
    del summary['tissue_f1']
    for row in summary['fold_results']:
        del row['tissue_f1']
    (bundle / 'cv_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    manifest = json.loads((bundle / 'bundle_manifest.json').read_text(encoding='utf-8'))
    if 'cv_summary.json' in manifest['files']:
        manifest['files']['cv_summary.json'] = file_hash(bundle / 'cv_summary.json')
        (bundle / 'bundle_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert 'tissue_f1' not in finalize_gallery(run_id, state)['cv_summary']


def holdout_run(tmp_path, monkeypatch, **extra):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'holdout_percent': 50,
                                       'xgboost': {'n_estimators': 2}, **extra}, candidate_count=6)
    return data, state, start_job(config, data, state).run_id


def test_holdout_run_scores_the_saved_model_on_samples_it_never_saw(tmp_path, monkeypatch):
    from gtex_explorer.model_registry import list_models, load_saved_model
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.training.worker import run_job
    data, state, run_id = holdout_run(tmp_path, monkeypatch)
    assert run_job(run_id, data, state) == 0
    receipt = list_gallery(state)[0]
    summary = receipt['holdout_summary']
    assert receipt['cv_summary'] is None
    assert summary['holdout_percent'] == 50 and summary['training_samples'] and summary['validation_samples']
    assert sum(summary['tissue_f1'].values()) / len(summary['tissue_f1']) == pytest.approx(summary['macro_f1'])
    assert sum(summary['tissue_support'].values()) == summary['validation_samples']
    status = json.loads((state / 'runs' / run_id / 'status.json').read_text(encoding='utf-8'))
    assert status['holdout_summary'] == summary
    entry = [model for model in list_models(tmp_path / 'resources', state) if model.model_id == run_id][0]
    assert load_saved_model(entry)[0]['run_id'] == run_id


def test_resumed_holdout_run_keeps_its_score(tmp_path, monkeypatch):
    import gtex_explorer.training.worker as worker
    data, state, run_id = holdout_run(tmp_path, monkeypatch)
    publish = worker.publish_model
    monkeypatch.setattr(worker, 'publish_model', lambda *args, **kwargs: (_ for _ in ()).throw(OSError('stopped')))
    assert worker.run_job(run_id, data, state) == 1
    assert (state / 'runs' / run_id / 'final_fit').exists()
    monkeypatch.setattr(worker, 'publish_model', publish)
    monkeypatch.setattr(worker, 'fit_classifier', lambda *args: pytest.fail('Resume must reuse the saved fit'))
    assert worker.run_job(run_id, data, state) == 0
    status = json.loads((state / 'runs' / run_id / 'status.json').read_text(encoding='utf-8'))
    assert status['holdout_summary']['macro_f1'] is not None


def test_folds_and_holdout_run_together_without_sharing_samples(tmp_path, monkeypatch):
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.training.worker import run_job
    data, state, run_id = holdout_run(tmp_path, monkeypatch, evaluate_cv=True, cv_folds=2)
    assert run_job(run_id, data, state) == 0
    receipt = list_gallery(state)[0]
    holdout, folds = receipt['holdout_summary'], receipt['cv_summary']
    assert folds['fold_count'] == 2
    for row in folds['fold_results']:
        assert row['training_samples'] + row['validation_samples'] == holdout['training_samples']
