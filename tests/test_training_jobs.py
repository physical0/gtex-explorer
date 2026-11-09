from pathlib import Path
import time
import pytest
from test_training_data import checkpoint_sources


def test_status_write_retries_transient_windows_lock(tmp_path, monkeypatch):
    import os
    import json
    from gtex_explorer.training.jobs import atomic_json
    replace = os.replace
    attempts = []
    def temporarily_locked(source, target):
        attempts.append(1)
        if len(attempts) < 3:
            error = PermissionError('temporary status lock')
            error.winerror = 5
            raise error
        return replace(source, target)
    monkeypatch.setattr(os, 'replace', temporarily_locked)
    atomic_json(tmp_path / 'status.json', {'status': 'completed'})
    assert json.loads((tmp_path / 'status.json').read_text()) == {'status': 'completed'}
    assert not list(tmp_path.glob('*.part'))


def test_status_read_retries_transient_windows_lock(tmp_path, monkeypatch):
    from gtex_explorer.training.jobs import read_job
    run_id = 'run_' + '0' * 32
    status = tmp_path / 'runs' / run_id / 'status.json'
    status.parent.mkdir(parents=True)
    status.write_text('{"status": "completed"}', encoding='utf-8')
    read_text = Path.read_text
    attempts = []
    def temporarily_locked(self, *args, **kwargs):
        if self == status:
            attempts.append(1)
            if len(attempts) < 3:
                # open() reports this lock as a plain errno 13, without a winerror.
                raise PermissionError(13, 'Permission denied')
        return read_text(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', temporarily_locked)
    assert read_job(run_id, tmp_path).status == 'completed'
    assert len(attempts) == 3


def test_status_read_reports_a_lock_that_does_not_clear(tmp_path, monkeypatch):
    from gtex_explorer.training import data
    status = tmp_path / 'status.json'
    status.write_text('{"status": "completed"}', encoding='utf-8')
    attempts = []
    def always_locked(self, *args, **kwargs):
        attempts.append(1)
        raise PermissionError(13, 'Permission denied')
    monkeypatch.setattr(Path, 'read_text', always_locked)
    monkeypatch.setattr(data.time, 'sleep', lambda seconds: None)
    with pytest.raises(PermissionError):
        data.read_json_file(status)
    assert len(attempts) == 6


def test_background_job_publishes_model_and_survives_repeated_status_reads(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job, read_job
    from gtex_explorer.platform.model_registry import list_models, load_saved_model
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
                                     'xgboost': {'n_estimators': 2}}, candidate_count=6)
    record = start_job(config, tmp_path / 'data', tmp_path / 'state')
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        record = read_job(record.run_id, tmp_path / 'state')
        if record.status in {'completed', 'failed', 'cancelled'}:
            break
        time.sleep(.2)
    assert record.status == 'completed', record.error
    models = list_models(tmp_path / 'resources', tmp_path / 'state')
    assert len(models) == 1
    pre, model = load_saved_model(models[0])
    assert pre['feature_count'] == 2
    assert model.get_booster().num_boosted_rounds() == 2
    assert record.details['cv_summary'] is None


def test_run_ids_cannot_escape_state_directory(tmp_path):
    from gtex_explorer.training.jobs import read_job
    with pytest.raises(ValueError):
        read_job('../secret', tmp_path)


def test_cv_job_saves_all_folds_and_cancelled_job_resumes(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job, read_job, request_cancel, resume_job
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'evaluate_cv': True,
                                     'cv_folds': 3, 'xgboost': {'n_estimators': 2}}, candidate_count=6)
    record = start_job(config, tmp_path / 'data', tmp_path / 'state')
    request_cancel(record.run_id, tmp_path / 'state')
    def wait():
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            current = read_job(record.run_id, tmp_path / 'state')
            if current.status in {'completed', 'failed', 'cancelled'}:
                return current
            time.sleep(.2)
        raise AssertionError('Worker timed out')
    assert wait().status == 'cancelled'
    resume_job(record.run_id, tmp_path / 'data', tmp_path / 'state')
    finished = wait()
    assert finished.status == 'completed', finished.error
    assert len(finished.details['cv_summary']['fold_results']) == 3
    assert len(finished.details['cv_summary']['jaccard_pairs']) == 3


def test_active_worker_lock_blocks_duplicate_launch(tmp_path):
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.jobs import process_lock, start_job
    from gtex_explorer.training.config import validate_training_config
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    with process_lock(tmp_path / 'state' / 'worker.lock'):
        with pytest.raises(ValueError, match='already running'):
            start_job(validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=6),
                      tmp_path / 'data', tmp_path / 'state')
