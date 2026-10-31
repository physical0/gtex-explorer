"""Durable state must survive replacement and reject concurrent lost updates."""
import hashlib
import json

import pytest

from gtex_explorer.cloud.state import CloudState, CloudConflict, CloudIntegrityError


class MemoryStore:
    def __init__(self):
        self.objects = {}
        self.counter = 0
        self.fail_artifacts = False

    def get(self, key):
        return self.objects.get(key)

    def put(self, key, data, *, expected=None, create=False):
        old = self.objects.get(key)
        if (create and old is not None) or (expected is not None and (old is None or old[1] != expected)):
            raise CloudConflict('Changed remotely')
        if self.fail_artifacts and '/objects/' in key:
            raise OSError('Storage unavailable')
        self.counter += 1
        etag = str(self.counter)
        self.objects[key] = (data, etag)
        return etag


def write(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_instance_replacement_restores_bytes_and_deletions(tmp_path):
    store = MemoryStore()
    a = CloudState(tmp_path / 'a', store)
    a.pull()
    write(a.root, 'workspaces.json', b'{"name":"Persistent"}')
    a.publish()
    b = CloudState(tmp_path / 'b', store)
    b.pull()
    assert (b.root / 'workspaces.json').read_bytes() == b'{"name":"Persistent"}'
    (b.root / 'workspaces.json').unlink()
    b.publish()
    a.pull()
    assert not (a.root / 'workspaces.json').exists()


def test_independent_changes_merge_but_registry_conflict_is_rejected(tmp_path):
    store = MemoryStore()
    a, b = (CloudState(tmp_path / name, store) for name in ['a', 'b'])
    a.pull()
    b.pull()
    write(a.root, 'workspaces.json', b'A')
    a.publish()
    write(b.root, 'runs/run_test/config.json', b'B')
    b.publish()
    a.pull()
    assert (a.root / 'runs/run_test/config.json').read_bytes() == b'B'
    b.pull()
    write(a.root, 'workspaces.json', b'new-A')
    write(b.root, 'workspaces.json', b'new-B')
    a.publish()
    with pytest.raises(CloudConflict):
        b.publish()
    b.pull()
    assert (b.root / 'workspaces.json').read_bytes() == b'new-A'


def test_failed_upload_cannot_publish_incomplete_state(tmp_path):
    store = MemoryStore()
    a = CloudState(tmp_path / 'a', store)
    a.pull()
    write(a.root, 'workspaces.json', b'old')
    a.publish()
    write(a.root, 'workspaces.json', b'new')
    store.fail_artifacts = True
    with pytest.raises(OSError):
        a.publish()
    store.fail_artifacts = False
    b = CloudState(tmp_path / 'b', store)
    b.pull()
    assert (b.root / 'workspaces.json').read_bytes() == b'old'


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/outside', 'runs/../../outside'])
def test_unsafe_remote_paths_are_rejected(tmp_path, name):
    store = MemoryStore()
    body = b'data'
    digest = hashlib.sha256(body).hexdigest()
    store.put('gtex/state.json', json.dumps({'schema_version': 1, 'files': {name: digest}, 'lease': None}).encode())
    store.put('gtex/objects/' + digest, body)
    with pytest.raises(CloudIntegrityError):
        CloudState(tmp_path / 'state', store).pull()
    assert not (tmp_path / 'outside').exists()


def test_corrupted_artifact_is_not_installed(tmp_path):
    store = MemoryStore()
    a = CloudState(tmp_path / 'a', store)
    a.pull()
    write(a.root, 'workspaces.json', b'original')
    a.publish()
    key = 'gtex/objects/' + hashlib.sha256(b'original').hexdigest()
    store.objects[key] = (b'corrupt', 'broken')
    b = CloudState(tmp_path / 'b', store)
    with pytest.raises(CloudIntegrityError):
        b.pull()
    assert not (b.root / 'workspaces.json').exists()


def test_worker_fencing_prevents_old_worker_committing_after_takeover(tmp_path):
    store = MemoryStore()
    a = CloudState(tmp_path / 'a', store)
    a.pull()
    assert a.acquire_lease('worker-a', now=100, ttl=30)
    assert not a.acquire_lease('worker-b', now=110, ttl=30)
    assert a.acquire_lease('worker-b', now=131, ttl=30)
    write(a.root, 'runs/run_test/status.json', b'completed')
    with pytest.raises(CloudConflict):
        a.publish(lease_owner='worker-a', now=132)
    a.publish(lease_owner='worker-b', now=132)


def test_partial_and_machine_local_files_are_not_backed_up(tmp_path):
    store = MemoryStore()
    state = CloudState(tmp_path / 'state', store)
    state.pull()
    for name in ['worker.lock', 'worker_owner.json', 'runs/run_test/worker.log',
                 'runs/run_test/status.json.part', 'preprocessing_cache/tmp/data.npy',
                 'models/run_test/model.json']:
        write(state.root, name, b'temporary')
    state.publish()
    manifest = json.loads(store.get('gtex/state.json')[0])
    assert manifest['files'] == {}


def test_cloud_read_retries_when_manifest_changes_during_download(monkeypatch):
    from gtex_explorer.cloud.state import NodeBlobStore
    calls = []
    def changing_read(operation, key, path):
        calls.append(key)
        if len(calls) < 3:
            raise CloudConflict('Manifest changed between GET and HEAD')
        path.write_bytes(b'consistent')
        return {'found': True, 'etag': 'strong-etag'}
    store = NodeBlobStore()
    monkeypatch.setattr(store, '_call', changing_read)
    assert store.get('gtex/state.json') == (b'consistent', 'strong-etag')
    assert len(calls) == 3
