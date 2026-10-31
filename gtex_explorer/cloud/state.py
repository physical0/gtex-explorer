"""Versioned private artifact storage. Local disk is only a disposable working copy."""
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import threading
import time


class CloudConflict(ValueError):
    pass


class CloudIntegrityError(ValueError):
    pass


class CloudLeaseLost(CloudConflict):
    pass


def enabled():
    return os.environ.get('GTEX_CLOUD') == '1'


def remote_client():
    return enabled() and os.environ.get('GTEX_CLOUD_ROLE') not in {'worker', 'supervisor'}


def _safe(root, name):
    parts = PurePosixPath(name)
    if (parts.is_absolute() or '..' in parts.parts or ':' in name or '\\' in name
            or not name or name.startswith('.') or not (root / name).resolve().is_relative_to(root.resolve())):
        raise CloudIntegrityError('Unsafe durable artifact path')
    return root / name


class NodeBlobStore:
    """Use the official JS SDK for streaming files and conditional ETag writes."""
    def _call(self, operation, key, filename, **options):
        args = [os.environ.get('GTEX_NODE', 'node'), str(Path(__file__).with_name('blob.mjs')),
                operation, key, str(filename), json.dumps(options)]
        result = subprocess.run(args, capture_output=True, text=True, timeout=300)
        if result.returncode:
            if result.returncode == 3:
                raise CloudConflict('Data di server berubah. Muat ulang dan ulangi tindakan.')
            raise OSError('Penyimpanan cloud tidak tersedia. Perubahan belum dikonfirmasi.')
        return json.loads(result.stdout)

    def get(self, key):
        with tempfile.TemporaryDirectory(prefix='gtex-blob-') as temporary:
            target = Path(temporary) / 'download'
            for attempt in range(5):
                try:
                    result = self._call('get', key, target)
                    return (target.read_bytes(), result['etag']) if result.get('found') else None
                except CloudConflict:
                    if attempt == 4:
                        raise

    def put(self, key, data, *, expected=None, create=False):
        with tempfile.TemporaryDirectory(prefix='gtex-blob-') as temporary:
            source = Path(temporary) / 'upload'
            source.write_bytes(data)
            return self.put_file(key, source, expected=expected, create=create)

    def put_file(self, key, path, *, expected=None, create=False):
        return self._call('put', key, path, expected=expected, create=create)['etag']


class CloudState:
    def __init__(self, root, store, prefix='gtex'):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.prefix = prefix.rstrip('/')
        self.baseline = {}
        self.loaded = False
        self.lock = threading.RLock()
        self.depth = 0
        self._hash_cache = {}

    def _manifest(self):
        found = self.store.get(self.prefix + '/state.json')
        if found is None:
            return {'schema_version': 1, 'files': {}, 'lease': None}, None
        value = json.loads(found[0])
        if value.get('schema_version') != 1 or not isinstance(value.get('files'), dict):
            raise CloudIntegrityError('Invalid cloud state manifest')
        for name, digest in value['files'].items():
            _safe(self.root, name)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise CloudIntegrityError('Invalid artifact checksum')
        return value, found[1]

    def _replace(self, manifest, etag):
        return self.store.put(self.prefix + '/state.json',
                              json.dumps(manifest, sort_keys=True, allow_nan=False).encode(),
                              expected=etag, create=etag is None)

    def _install(self, name, digest):
        target = _safe(self.root, name)
        found = self.store.get(self.prefix + '/objects/' + digest)
        if found is None or hashlib.sha256(found[0]).hexdigest() != digest:
            raise CloudIntegrityError('Durable artifact checksum mismatch: ' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.cloud.part')
        try:
            temporary.write_bytes(found[0])
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def pull(self):
        with self.lock:
            manifest, _ = self._manifest()
            files = manifest['files']
            # Validate every remote path before touching the local working copy.
            for name in files:
                _safe(self.root, name)
            for name, digest in files.items():
                target = _safe(self.root, name)
                if self.baseline.get(name) != digest or not target.is_file():
                    self._install(name, digest)
            for name in self.baseline.keys() - files.keys():
                _safe(self.root, name).unlink(missing_ok=True)
            self.baseline = dict(files)
            self.loaded = True
            return manifest

    def _durable(self, path):
        relative = path.relative_to(self.root)
        parts = relative.parts
        if path.is_symlink():
            raise CloudIntegrityError('Symlinks cannot be stored as durable artifacts')
        if any(p.startswith('.') or p.endswith(('.part', '.lock', '.log')) for p in parts):
            return False
        if any(p in {'preprocessing_cache', 'ranking_cache', '__pycache__'} for p in parts):
            return False
        if path.name == 'worker_owner.json':
            return False
        if any(p.startswith(('final-fit-', 'publishing-', 'upload-', 'preparing-')) for p in parts):
            return False
        for index, part in enumerate(parts):
            if part in {'models', 'datasets'} and len(parts) > index + 2:
                directory = self.root.joinpath(*parts[:index + 2])
                expected = 'bundle_manifest.json' if part == 'models' else 'manifest.json'
                if not (directory / expected).is_file():
                    return False
        return True

    def _local_files(self):
        files = {}
        for path in self.root.rglob('*'):
            if not path.is_file() or not self._durable(path):
                continue
            name = path.relative_to(self.root).as_posix()
            _safe(self.root, name)
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            previous = self._hash_cache.get(name)
            if previous and previous[0] == signature:
                digest = previous[1]
            else:
                hasher = hashlib.sha256()
                with path.open('rb') as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
                self._hash_cache[name] = (signature, digest)
            files[name] = digest
        return files

    def publish(self, *, lease_owner=None, now=None, owned_paths=(), only=None):
        with self.lock:
            local = self._local_files()
            keys = local.keys() | self.baseline.keys()
            if only is not None:
                keys = keys & set(only)
            changes = {key: local.get(key) for key in keys
                       if local.get(key) != self.baseline.get(key)}
            for key, digest in changes.items():
                if digest is not None:
                    object_key = self.prefix + '/objects/' + digest
                    try:
                        if hasattr(self.store, 'put_file'):
                            self.store.put_file(object_key, _safe(self.root, key), create=True)
                        else:
                            self.store.put(object_key, _safe(self.root, key).read_bytes(), create=True)
                    except CloudConflict:
                        found = self.store.get(object_key)
                        if found is None or hashlib.sha256(found[0]).hexdigest() != digest:
                            raise CloudIntegrityError('Existing durable object failed verification')
            for _ in range(5):
                manifest, etag = self._manifest()
                if lease_owner is not None:
                    lease = manifest.get('lease') or {}
                    if lease.get('owner') != lease_owner or lease.get('expires', 0) <= (time.time() if now is None else now):
                        raise CloudLeaseLost('Worker lease expired; stale worker cannot publish')
                if not changes and etag is not None:
                    return
                for key, digest in changes.items():
                    current = manifest['files'].get(key)
                    if current != self.baseline.get(key) and current != digest and key not in owned_paths:
                        raise CloudConflict('Data di server berubah. Muat ulang sebelum menyimpan: ' + key)
                    if digest is None:
                        manifest['files'].pop(key, None)
                    else:
                        manifest['files'][key] = digest
                try:
                    self._replace(manifest, etag)
                    # Only adopt the files actually present locally. Remote changes
                    # merged above will be downloaded on the next pull.
                    if only is None:
                        self.baseline = local
                    else:
                        for key in keys:
                            if key in local:
                                self.baseline[key] = local[key]
                            else:
                                self.baseline.pop(key, None)
                    self.loaded = True
                    return
                except CloudConflict:
                    continue
            raise CloudConflict('Penyimpanan sedang digunakan. Coba kembali.')

    def acquire_lease(self, owner, *, now=None, ttl=60):
        now = time.time() if now is None else now
        for _ in range(5):
            manifest, etag = self._manifest()
            current = manifest.get('lease') or {}
            if current.get('owner') != owner and current.get('expires', 0) > now:
                return False
            manifest['lease'] = {'owner': owner, 'expires': now + ttl}
            try:
                self._replace(manifest, etag)
                return True
            except CloudConflict:
                continue
        return False

    def release_lease(self, owner):
        for _ in range(5):
            manifest, etag = self._manifest()
            if (manifest.get('lease') or {}).get('owner') != owner:
                return
            manifest['lease'] = None
            try:
                self._replace(manifest, etag)
                return
            except CloudConflict:
                continue

    @contextmanager
    def transaction(self):
        with self.lock:
            outer = self.depth == 0
            if outer:
                self.pull()
            self.depth += 1
            try:
                yield self
            finally:
                self.depth -= 1
                if outer:
                    self.publish()


_instance = None


def state():
    global _instance
    if _instance is None:
        root = Path(os.environ['GTEX_DESKTOP_STATE_DIR'])
        _instance = CloudState(root, NodeBlobStore(), os.environ.get('GTEX_CLOUD_PREFIX', 'gtex'))
    return _instance


def operation(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        if remote_client():
            with state().transaction():
                return function(*args, **kwargs)
        return function(*args, **kwargs)
    return wrapped


def worker_checkpoint():
    if enabled() and os.environ.get('GTEX_CLOUD_ROLE') == 'worker':
        current = state()
        owned = [name for name in current._local_files() if name.endswith('/status.json')]
        current.publish(lease_owner=os.environ['GTEX_CLOUD_LEASE'], owned_paths=owned)
