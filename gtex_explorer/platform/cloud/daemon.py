"""One fenced training supervisor. Jobs and checkpoints outlive this process."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from gtex_explorer.platform.cloud.state import state, CloudConflict, CloudLeaseLost
from gtex_explorer.training.jobs import atomic_json, pause_job, unpause_job, request_cancel, resolve_job_data

# An idle poll is one small version query; queued jobs start within this delay.
IDLE_POLL = 10
CLEANUP_INTERVAL = 60 * 60


def candidates(manifest):
    return sorted(name for name in manifest['files']
                  if name.endswith('/status.json') and '/runs/run_' in '/' + name)


def main():
    if os.environ.get('GTEX_CLOUD') != '1':
        raise RuntimeError('Cloud worker requires GTEX_CLOUD=1')
    os.environ['GTEX_CLOUD_ROLE'] = 'supervisor'
    os.environ.setdefault('GTEX_CLOUD_PREFIX', 'gtex-production')
    durable = state()
    owner = uuid.uuid4().hex
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    child = None
    active = None
    handled = None
    owns_lease = False
    next_heartbeat = 0.
    next_cleanup = time.monotonic() + 60
    try:
        while not stopped.is_set():
            try:
                if child is None:
                    if time.monotonic() >= next_cleanup:
                        next_cleanup = time.monotonic() + CLEANUP_INTERVAL
                        removed = durable.collect_garbage()
                        if removed:
                            print('Cloud cleanup removed', removed, 'objects', flush=True)
                    manifest, _ = durable._manifest()
                    lease = manifest.get('lease') or {}
                    if lease.get('expires', 0) > time.time() and lease.get('owner') != owner:
                        stopped.wait(IDLE_POLL)
                        continue
                    queued = []
                    for name in candidates(manifest):
                        found = durable.store.get(durable.prefix + '/objects/' + manifest['files'][name])
                        if found is None:
                            raise OSError('Status artifact missing')
                        value = json.loads(found[0])
                        if value['status'] == 'queued':
                            queued.append(name)
                    if not queued:
                        stopped.wait(IDLE_POLL)
                        continue
                    if not durable.acquire_lease(owner, ttl=90):
                        continue
                    owns_lease = True
                    durable.pull()
                    status_path = durable.root / queued[0]
                    active = status_path.parent
                    run_id = active.name
                    workspace = active.parent.parent
                    saved = json.loads((active / 'config.json').read_text(encoding='utf-8'))
                    pending_control = active / 'cloud_control.json'
                    if pending_control.is_file() and json.loads(pending_control.read_text())['action'] == 'stop':
                        atomic_json(active / 'status.json', {'status': 'cancelled', 'stage': 'cancelled',
                                    'updated': time.time(), 'error': None})
                        name = (active / 'status.json').relative_to(durable.root).as_posix()
                        durable.publish(lease_owner=owner, only=[name], owned_paths=[name])
                        durable.release_lease(owner)
                        owns_lease = False
                        active = None
                        continue
                    env = {**os.environ, 'GTEX_CLOUD_ROLE': 'worker', 'GTEX_CLOUD_LEASE': owner}
                    child = subprocess.Popen([sys.executable, '-m', 'gtex_explorer.training.worker',
                                              '--run-id', run_id, '--data', str(resolve_job_data(saved, workspace, saved['data_root'])),
                                              '--state', str(workspace)], env=env)
                    handled = None
                    next_heartbeat = time.monotonic() + 15
                    print('Training worker started:', run_id, flush=True)
                else:
                    if time.monotonic() >= next_heartbeat:
                        if not durable.acquire_lease(owner, ttl=90):
                            raise CloudLeaseLost('Another supervisor owns the worker lease')
                        next_heartbeat = time.monotonic() + 15
                    if child.poll() is not None:
                        value = json.loads((active / 'status.json').read_text(encoding='utf-8'))
                        if value['status'] not in {'completed', 'failed', 'cancelled'}:
                            value.update(status='failed', updated=time.time(), error={
                                'type': 'WorkerStopped', 'message': 'Worker stopped. Continue from checkpoint.'})
                            atomic_json(active / 'status.json', value)
                            name = (active / 'status.json').relative_to(durable.root).as_posix()
                            durable.publish(lease_owner=owner, only=[name], owned_paths=[name])
                        durable.release_lease(owner)
                        owns_lease = False
                        child = None
                        active = None
                        continue
                    manifest, _ = durable._manifest()
                    name = (active / 'cloud_control.json').relative_to(durable.root).as_posix()
                    digest = manifest['files'].get(name)
                    if digest and digest != handled:
                        durable._install(name, digest)
                        control = json.loads((active / 'cloud_control.json').read_text(encoding='utf-8'))
                        workspace = active.parent.parent
                        action = control['action']
                        if action == 'pause':
                            pause_job(active.name, workspace)
                        elif action == 'resume':
                            unpause_job(active.name, workspace)
                        elif action == 'stop':
                            request_cancel(active.name, workspace)
                        else:
                            raise ValueError('Unknown durable training control')
                        relative = active.relative_to(durable.root)
                        names = [(relative / filename).as_posix() for filename in
                                 ['status.json', 'pause_clock.json', 'cancel.json']]
                        durable.publish(lease_owner=owner, only=names, owned_paths=names)
                        handled = digest
                stopped.wait(2)
            except Exception as error:
                print('Cloud supervisor:', type(error).__name__, str(error), flush=True)
                if isinstance(error, CloudLeaseLost) and child is not None:
                    # A fenced-out worker must never continue writing results.
                    child.kill()
                    child.wait()
                    child = None
                    active = None
                    owns_lease = False
                stopped.wait(5)
    finally:
        if child is not None:
            try:
                import psutil
                process = psutil.Process(child.pid)
                process.resume()
                child.terminate()
                child.wait(timeout=10)
            except Exception:
                child.kill()
                child.wait()
            if owns_lease:
                value = json.loads((active / 'status.json').read_text(encoding='utf-8'))
                if value['status'] not in {'completed', 'failed', 'cancelled'}:
                    value.update(status='failed', updated=time.time(), error={
                        'type': 'WorkerStopped', 'message': 'Server restarted. Continue from checkpoint.'})
                    atomic_json(active / 'status.json', value)
                    name = (active / 'status.json').relative_to(durable.root).as_posix()
                    durable.publish(lease_owner=owner, only=[name], owned_paths=[name])
        if owns_lease:
            durable.release_lease(owner)


if __name__ == '__main__':
    main()
