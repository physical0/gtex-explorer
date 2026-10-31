import json
import time
import uuid
from gtex_explorer.training.jobs import atomic_json, job_path, read_job


def test_read_job_recovers_a_dead_worker_without_pid_guessing(tmp_path):
    run_id = 'run_' + uuid.uuid4().hex
    root = job_path(run_id, tmp_path)
    atomic_json(root / 'status.json', {'status': 'running', 'stage': 'final_selector',
                                     'updated': time.time() - 60, 'error': None})
    record = read_job(run_id, tmp_path)
    assert record.status == 'failed'
    assert record.error['type'] == 'WorkerStopped'


def test_recent_queued_job_is_not_marked_dead(tmp_path):
    run_id = 'run_' + uuid.uuid4().hex
    atomic_json(job_path(run_id, tmp_path) / 'status.json',
                {'status': 'queued', 'updated': time.time(), 'error': None})
    assert read_job(run_id, tmp_path).status == 'queued'


def test_final_fit_is_reused_after_cancel_and_publication_status_failure(tmp_path, monkeypatch):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data
    import gtex_explorer.training.jobs as training_jobs
    import gtex_explorer.training.worker as worker
    from gtex_explorer.training.config import validate_training_config
    export_training_data(*checkpoint_sources(tmp_path), tmp_path / 'data')
    # Keep actual job files and training, but execute the worker in this process.
    monkeypatch.setattr(training_jobs, '_launch', lambda *args: None)
    record = training_jobs.start_job(validate_training_config({'selector': 'anova', 'top_k': 2,
                                     'xgboost': {'n_estimators': 2}}, candidate_count=6), tmp_path / 'data', tmp_path / 'state')
    actual_fit = worker.fit_classifier
    fits = []
    def cancelling_fit(*args):
        result = actual_fit(*args)
        fits.append(result)
        atomic_json(job_path(record.run_id, tmp_path / 'state') / 'cancel.json', {'cancel': True})
        return result
    monkeypatch.setattr(worker, 'fit_classifier', cancelling_fit)
    assert worker.run_job(record.run_id, tmp_path / 'data', tmp_path / 'state') == 0
    assert read_job(record.run_id, tmp_path / 'state').status == 'cancelled'
    (job_path(record.run_id, tmp_path / 'state') / 'cancel.json').unlink()
    real_json = worker.atomic_json
    failed = []
    def completion_failure(path, value):
        if value.get('status') == 'completed' and not failed:
            failed.append(True)
            raise OSError('simulated completion status write failure')
        return real_json(path, value)
    monkeypatch.setattr(worker, 'atomic_json', completion_failure)
    assert worker.run_job(record.run_id, tmp_path / 'data', tmp_path / 'state') == 1
    assert worker.run_job(record.run_id, tmp_path / 'data', tmp_path / 'state') == 0
    assert len(fits) == 1
    assert read_job(record.run_id, tmp_path / 'state').status == 'completed'
