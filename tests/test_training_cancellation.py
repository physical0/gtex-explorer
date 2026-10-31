"""Cancellation must interrupt work, retain completed checkpoints and report real progress."""
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


def test_xgboost_can_be_cancelled_between_rounds_before_full_fit():
    from gtex_explorer.training.core import TrainingCancelled, fit_selector, fit_classifier
    from gtex_explorer.training.config import validate_training_config
    from test_training_core import prepared_fixture
    prepared = prepared_fixture()
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
                                      'xgboost': {'n_estimators': 100}}, candidate_count=12)
    selector = fit_selector(prepared, config)
    rounds = []
    def progress(completed):
        rounds.append(completed)
        if completed == 2:
            raise TrainingCancelled()
    with pytest.raises(TrainingCancelled):
        fit_classifier(prepared, selector, config, progress)
    assert rounds == [0, 1, 2]


def test_request_cancel_has_immediate_pending_status_without_changing_other_run(tmp_path):
    from gtex_explorer.training.jobs import atomic_json, job_path, process_lock, request_cancel
    run_id, other = 'run_' + 'a'*32, 'run_' + 'b'*32
    for run in [run_id, other]:
        atomic_json(job_path(run, tmp_path) / 'status.json', {
            'status': 'running', 'stage': 'final_xgboost', 'updated': time.time(),
            'rounds_completed': 7, 'rounds_total': 300, 'error': None})
    with process_lock(tmp_path / 'worker.lock'):
        record = request_cancel(run_id, tmp_path)
    assert record.status == 'cancelling'
    assert record.details['rounds_completed'] == 7
    assert not (job_path(other, tmp_path) / 'cancel.json').exists()


def test_dead_worker_after_cancel_is_cancelled_instead_of_failed(tmp_path):
    from gtex_explorer.training.jobs import atomic_json, job_path, read_job
    run_id = 'run_' + 'c'*32
    root = job_path(run_id, tmp_path)
    atomic_json(root / 'status.json', {'status': 'running', 'stage': 'final_selector',
                                      'updated': 0, 'error': None})
    atomic_json(root / 'cancel.json', {'cancel': True})
    record = read_job(run_id, tmp_path)
    assert record.status == 'cancelled'
    assert record.error is None


def test_stop_interrupts_long_selector_process_and_preserves_finished_files(tmp_path, monkeypatch):
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.config import validate_training_config
    import gtex_explorer.training.jobs as training_jobs
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr(training_jobs, '_launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=6)
    job = training_jobs.start_job(config, data, state)
    root = training_jobs.job_path(job.run_id, state)
    finished = root / 'completed_checkpoint.txt'
    finished.write_text('keep completed work')
    child = "\nimport sys,time\nfrom pathlib import Path\nimport gtex_explorer.training.worker as worker\ndef blocked_selector(*args, **kwargs):\n    Path(sys.argv[4]).write_text('entered selector')\n    time.sleep(60)\n    raise AssertionError('Cancellation did not interrupt the selector')\nworker.fit_selector = blocked_selector\nraise SystemExit(worker.run_job(sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3]),interruptible=True))\n"
    entered = root / 'entered.txt'
    process = subprocess.Popen([sys.executable, '-c', child, job.run_id, str(data), str(state), str(entered)],
                               cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 25
        while not entered.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert entered.exists(), process.poll()
        started = time.monotonic()
        training_jobs.request_cancel(job.run_id, state)
        process.wait(timeout=5)
        assert time.monotonic() - started < 5
        assert process.returncode == 0
        assert training_jobs.read_job(job.run_id, state).status == 'cancelled'
        assert finished.read_text() == 'keep completed work'
        assert not (state / 'models' / job.run_id).exists()
        assert not (root / 'final_fit').exists()
        # Only the test worker was stopped. Its OS lock is released for a new run.
        with training_jobs.process_lock(state / 'worker.lock'):
            pass
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def test_cancelled_classifier_never_publishes_partial_fit(tmp_path, monkeypatch):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    import gtex_explorer.training.jobs as training_jobs
    import gtex_explorer.training.worker as worker
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr(training_jobs, '_launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
                                      'xgboost': {'n_estimators': 100}}, candidate_count=6)
    job = training_jobs.start_job(config, data, state)
    real_fit = worker.fit_classifier
    observed = []
    def cancel_second_round(prepared, selector, config, progress):
        def on_round(completed):
            observed.append(completed)
            if completed == 2:
                training_jobs.request_cancel(job.run_id, state)
            progress(completed)
        return real_fit(prepared, selector, config, on_round)
    monkeypatch.setattr(worker, 'fit_classifier', cancel_second_round)
    assert worker.run_job(job.run_id, data, state) == 0
    record = training_jobs.read_job(job.run_id, state)
    assert record.status == 'cancelled'
    assert observed[-1] == 2
    assert not (state / 'models' / job.run_id).exists()
    assert not (training_jobs.job_path(job.run_id, state) / 'final_fit').exists()
