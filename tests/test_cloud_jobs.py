import json
import time
from unittest.mock import patch

from gtex_explorer.platform.cloud.state import CloudState
from test_cloud_state import MemoryStore, write


def test_remote_pause_and_stop_create_durable_commands_without_touching_local_processes(tmp_path, monkeypatch):
    import gtex_explorer.platform.cloud.state as cloud_state
    from gtex_explorer.training.jobs import pause_job, request_cancel
    store = MemoryStore()
    state = CloudState(tmp_path / 'state', store)
    monkeypatch.setenv('GTEX_CLOUD', '1')
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'web')
    monkeypatch.setattr(cloud_state, '_instance', state)
    run = 'run_' + 'a' * 32
    state.pull()
    write(state.root, f'runs/{run}/status.json', json.dumps({
        'status': 'running', 'stage': 'fitting_xgboost', 'updated': time.time(), 'error': None}).encode())
    state.publish()
    state.acquire_lease('worker')
    with patch('gtex_explorer.training.jobs._verified_worker', side_effect=AssertionError('Must not inspect a remote PID')):
        pause_job(run, state.root)
        assert json.loads((state.root / f'runs/{run}/cloud_control.json').read_text())['action'] == 'pause'
        request_cancel(run, state.root)
        assert json.loads((state.root / f'runs/{run}/cloud_control.json').read_text())['action'] == 'stop'
    restored = CloudState(tmp_path / 'restored', store)
    restored.pull()
    assert json.loads((restored.root / f'runs/{run}/cloud_control.json').read_text())['action'] == 'stop'


def test_stale_worker_is_reported_recoverable_without_pid_probe(tmp_path, monkeypatch):
    import gtex_explorer.platform.cloud.state as cloud_state
    from gtex_explorer.training.jobs import read_job
    store = MemoryStore()
    state = CloudState(tmp_path / 'state', store)
    monkeypatch.setenv('GTEX_CLOUD', '1')
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'web')
    monkeypatch.setattr(cloud_state, '_instance', state)
    run = 'run_' + 'b' * 32
    state.pull()
    write(state.root, f'runs/{run}/status.json', json.dumps({
        'status': 'running', 'stage': 'fitting_xgboost', 'updated': 1, 'error': None}).encode())
    state.publish()
    record = read_job(run, state.root)
    assert record.status == 'failed'
    assert record.error['type'] == 'WorkerStopped'
