"""Live pause must preserve the same process and unfinished fit in memory."""
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil
import pytest


def wait_until(check, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.025)
    raise AssertionError('Timed out waiting for worker state')


@pytest.fixture
def live_worker(tmp_path, monkeypatch):
    from test_training_data import checkpoint_sources
    from scripts.export_training_data import export_training_data
    from gtex_explorer.training.config import validate_training_config
    import gtex_explorer.training.jobs as jobs
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr(jobs, '_launch', lambda *args: None)
    children = []

    def launch(stage):
        config = validate_training_config({'selector': 'anova', 'top_k': 2,
            'xgboost': {'n_estimators': 20, 'n_jobs': 1}}, candidate_count=6)
        job = jobs.start_job(config, data, state)
        root = jobs.job_path(job.run_id, state)
        script = "\nimport sys,time\nfrom pathlib import Path\nimport gtex_explorer.training.worker as w\nrun_id,data,state,mode=sys.argv[1:]\nroot=Path(state)/'runs'/run_id\nselector,fit=w.fit_selector,w.fit_classifier\ndef slow_selector(*args,**kwargs):\n    with (root/'selector_calls.txt').open('a') as out: out.write('fit\\n')\n    if mode=='selector':\n        for number in range(60):\n            (root/'heartbeat.txt').write_text(str(number))\n            time.sleep(.05)\n    return selector(*args,**kwargs)\ndef slow_fit(prepared,selected,config,progress):\n    with (root/'classifier_calls.txt').open('a') as out: out.write('fit\\n')\n    def report(number):\n        progress(number)\n        (root/'heartbeat.txt').write_text(str(number))\n        if mode=='xgboost': time.sleep(.08)\n    return fit(prepared,selected,config,report)\nw.fit_selector,w.fit_classifier=slow_selector,slow_fit\nraise SystemExit(w.run_job(run_id,Path(data),Path(state),interruptible=True))\n"
        script_path = root/'training_worker.py'
        script_path.write_text('import sys\nsys.path.insert(0, '+repr(str(Path(__file__).resolve().parents[1]))+')\n'+script)
        process = subprocess.Popen([sys.executable, str(script_path), job.run_id, str(data), str(state), stage],
                                   cwd=Path(__file__).resolve().parents[1],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append((process, root))
        wait_until(lambda: (root/'heartbeat.txt').exists() or process.poll() is not None)
        assert process.poll() is None
        return jobs, job, root, process, state
    yield launch
    for child, root in children:
        if child.poll() is None:
            try:
                identity = json.loads((root/'worker.json').read_text())
                worker = psutil.Process(identity['pid'])
                if worker.create_time() == identity['created']:
                    worker.resume()
                    worker.terminate()
            except (OSError, psutil.Error):
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=5)


@pytest.mark.parametrize('stage', ['selector', 'xgboost'])
def test_live_pause_freezes_progress_and_resumes_same_fit(live_worker, stage):
    jobs, job, root, child, state = live_worker(stage)
    pause = getattr(jobs, 'pause_job', None)
    resume = getattr(jobs, 'unpause_job', None)
    assert callable(pause) and callable(resume), 'Live pause/resume controls are missing'
    identity = json.loads((root/'worker.json').read_text())
    record = pause(job.run_id, state)
    assert record.status == 'paused'
    frozen = (root/'heartbeat.txt').read_bytes()
    time.sleep(.3)
    assert (root/'heartbeat.txt').read_bytes() == frozen
    assert jobs.read_job(job.run_id, state).status == 'paused'
    with pytest.raises(ValueError, match='already running'):
        jobs._assert_idle(state)
    assert resume(job.run_id, state).status in {'running', 'completed'}
    assert json.loads((root/'pause_clock.json').read_text())['seconds'] >= .3
    child.wait(timeout=25)
    assert child.returncode == 0
    assert jobs.read_job(job.run_id, state).status == 'completed'
    assert json.loads((root/'worker.json').read_text()) == identity
    assert (root/'selector_calls.txt').read_text().splitlines() == ['fit']
    assert (root/'classifier_calls.txt').read_text().splitlines() == ['fit']
    assert (root/'heartbeat.txt').read_text() == '20'


def test_end_process_while_paused_does_not_leave_a_suspended_worker(live_worker):
    jobs, job, root, child, state = live_worker('selector')
    pause = getattr(jobs, 'pause_job', None)
    assert callable(pause), 'Live pause is missing'
    pause(job.run_id, state)
    jobs.request_cancel(job.run_id, state)
    child.wait(timeout=5)
    assert jobs.read_job(job.run_id, state).status == 'cancelled'
    assert not (state/'models'/job.run_id).exists()
    assert not (root/'final_fit').exists()


def test_dead_paused_worker_is_recoverable_as_failed(tmp_path):
    from gtex_explorer.training.jobs import atomic_json, job_path, read_job
    run_id = 'run_'+'f'*32
    atomic_json(job_path(run_id,tmp_path)/'status.json',
                {'status':'paused','stage':'final_xgboost','updated':0,'error':None})
    record = read_job(run_id,tmp_path)
    assert record.status == 'failed'
    assert record.error['type'] == 'WorkerStopped'


def test_pause_rejects_controller_or_reused_pid_without_suspending_it(tmp_path):
    import os
    import gtex_explorer.training.jobs as jobs
    run_id='run_'+'d'*32
    root=jobs.job_path(run_id,tmp_path)
    jobs.atomic_json(root/'status.json', {'status':'running','stage':'final_selector',
                     'updated':time.time(),'error':None})
    jobs.atomic_json(root/'worker.json', {'run_id':run_id,'pid':os.getpid(),
                     'created':psutil.Process().create_time(),'pause_supported':True})
    pause=getattr(jobs,'pause_job',None)
    assert callable(pause), 'Live pause is missing'
    with jobs.process_lock(tmp_path/'worker.lock'):
        with pytest.raises(ValueError,match='worker'):
            pause(run_id,tmp_path)
