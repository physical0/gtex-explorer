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

    def signed_upload_url(self, key):
        return 'https://storage.test/upload/' + key

    def signed_download_url(self, key, filename, expires=3600):
        return 'https://storage.test/download/' + key


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


class FakeSupabase:
    """Just enough PostgREST and Storage behaviour to exercise SupabaseStore."""
    def __init__(self):
        self.rows = {}
        self.objects = {}
        self.calls = []

    def __call__(self, method, path, body=None, headers=None):
        from urllib.parse import unquote, urlsplit, parse_qs
        self.calls.append((method, path))
        url = urlsplit(path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == '/rest/v1/cloud_manifest':
            key = query.get('key', 'eq.')[3:]
            row = self.rows.get(key)
            if method == 'GET':
                fields = query['select'].split(',')
                return 200, json.dumps([{f: row[f] for f in fields}] if row else []).encode()
            value = json.loads(body)
            if method == 'POST':
                if value['key'] in self.rows:
                    return 409, b'{}'
                self.rows[value['key']] = value
                return 201, b''
            if row is None or str(row['version']) != query['version'][3:]:
                return 200, b'[]'
            row.update(version=value['version'], body=value['body'])
            return 200, json.dumps([row]).encode()
        if url.path == '/rest/v1/rpc/stale_objects':
            prefix = json.loads(body)['name_prefix']
            return 200, json.dumps(sorted(k for k in self.objects if k.startswith(prefix))).encode()
        if method == 'DELETE':
            for name in json.loads(body)['prefixes']:
                self.objects.pop(name, None)
            return 200, b'[]'
        name = unquote(url.path.split('/gtex/', 1)[1])
        if method == 'GET':
            return (200, self.objects[name]) if name in self.objects else (400, b'{"error":"not_found","message":"Object not found"}')
        if headers.get('x-upsert') == 'false' and name in self.objects:
            return 400, b'{"error":"Duplicate","message":"The resource already exists"}'
        self.objects[name] = body
        return 200, b'{}'


def supabase_store():
    from gtex_explorer.cloud.state import SupabaseStore
    store = SupabaseStore('https://example.supabase.co', 'test-key')
    fake = FakeSupabase()
    store._request = fake
    return store, fake


def test_unchanged_manifest_is_one_version_query_without_body():
    store, fake = supabase_store()
    store.put('gtex/state.json', b'{"v":1}', create=True)
    assert store.get('gtex/state.json') == (b'{"v":1}', '1')
    fake.calls.clear()
    assert store.get('gtex/state.json') == (b'{"v":1}', '1')
    assert fake.calls == [('GET', '/rest/v1/cloud_manifest?key=eq.gtex%2Fstate.json&select=version')]


def test_manifest_compare_and_swap_rejects_stale_version():
    store, fake = supabase_store()
    first = store.put('gtex/state.json', b'A', create=True)
    store.put('gtex/state.json', b'B', expected=first)
    with pytest.raises(CloudConflict):
        store.put('gtex/state.json', b'C', expected=first)
    other, _ = supabase_store()
    other._request = fake
    assert other.get('gtex/state.json') == (b'B', '2')


def test_objects_above_part_size_round_trip_as_parts(monkeypatch):
    import gtex_explorer.cloud.state as cloud_state
    monkeypatch.setattr(cloud_state, 'PART_SIZE', 4)
    store, fake = supabase_store()
    store.put('gtex/objects/big', b'0123456789', create=True)
    assert sorted(fake.objects) == ['gtex/objects/big', 'gtex/objects/big.part000',
                                    'gtex/objects/big.part001', 'gtex/objects/big.part002']
    with pytest.raises(CloudConflict):
        store.put('gtex/objects/big', b'0123456789', create=True)
    fresh, _ = supabase_store()
    fresh._request = fake
    assert fresh.get('gtex/objects/big') == (b'0123456789', 'immutable')


def test_publish_without_changes_skips_the_network(tmp_path):
    store, fake = supabase_store()
    state = CloudState(tmp_path / 'state', store)
    state.pull()
    write(state.root, 'workspaces.json', b'{}')
    state.publish()
    state.pull()
    fake.calls.clear()
    state.publish()
    assert fake.calls == []


def test_garbage_collection_keeps_live_versions_and_drops_old_ones(tmp_path):
    store, fake = supabase_store()
    state = CloudState(tmp_path / 'state', store)
    state.pull()
    write(state.root, 'workspaces.json', b'old')
    state.publish()
    write(state.root, 'workspaces.json', b'new')
    state.publish()
    fake.objects['incoming/abc/upload'] = b'expired upload'
    assert state.collect_garbage() == 2
    assert sorted(fake.objects) == ['gtex/objects/' + hashlib.sha256(b'new').hexdigest()]
    fresh = CloudState(tmp_path / 'fresh', store)
    fresh.pull()
    assert (fresh.root / 'workspaces.json').read_bytes() == b'new'
