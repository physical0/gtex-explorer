"""Versioned private artifact storage. Local disk is only a disposable working copy."""
import collections
import contextlib
from contextlib import contextmanager
import functools
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


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


UNAVAILABLE = 'Penyimpanan cloud tidak tersedia. Perubahan belum dikonfirmasi.'
# Supabase Free rejects objects above 50 MB; larger bodies are stored as parts
# behind a small index object so callers still see one key.
PART_SIZE = 45 * 1024 * 1024
PARTS_MAGIC = b'GTEX-PARTS-1\n'


def part_key(key, index):
    return f'{key}.part{index:03d}'


def _file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _deferred(name):
    # Only the training worker reads uploaded matrices, each up to 200 MB. Web
    # instances fetch one through ensure() when a run starts from that dataset.
    parts = PurePosixPath(name).parts
    return remote_client() and parts[-1] == 'raw_tpm.npy' and 'datasets' in parts


class SupabaseStore:
    """Manifests are versioned Postgres rows; every other key is a private Storage object.

    Unchanged manifests cost one tiny version query, and content-addressed
    objects are immutable, so both are cached to keep polling egress low.
    """
    def __init__(self, url=None, key=None, bucket='gtex'):
        self.url = (url or os.environ.get('SUPABASE_URL', '')).rstrip('/')
        self.key = key or os.environ.get('SUPABASE_SECRET_KEY', '')
        self.bucket = bucket
        self._rows = {}
        self._objects = collections.OrderedDict()
        self._object_bytes = 0

    def _request(self, method, path, body=None, headers=None):
        if not self.url or not self.key:
            raise OSError(UNAVAILABLE)
        request = urllib.request.Request(self.url + path, data=body, method=method, headers={
            'apikey': self.key, 'Authorization': 'Bearer ' + self.key, **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()
        except OSError as error:
            raise OSError(UNAVAILABLE) from error

    def _json(self, method, path, value, headers=None):
        return self._request(method, path, json.dumps(value).encode(),
                             {'Content-Type': 'application/json', **(headers or {})})

    @staticmethod
    def _manifest_key(key):
        return key.endswith('/state.json')

    def _object_path(self, key):
        return f'/storage/v1/object/{self.bucket}/' + urllib.parse.quote(key)

    # Manifest rows: version is the ETag, conditional PATCH is the compare-and-swap.
    def _row_filter(self, key):
        return '/rest/v1/cloud_manifest?key=eq.' + urllib.parse.quote(key, safe='')

    def _get_row(self, key):
        cached = self._rows.get(key)
        status, body = self._request('GET', self._row_filter(key) + '&select=version'
                                     + ('' if cached else ',body'))
        if status != 200:
            raise OSError(UNAVAILABLE)
        rows = json.loads(body)
        if not rows:
            self._rows.pop(key, None)
            return None
        version = str(rows[0]['version'])
        if cached and cached[0] == version:
            return cached[1], version
        if cached:
            # Changed remotely: drop the stale copy and fetch the body.
            del self._rows[key]
            return self._get_row(key)
        data = rows[0]['body'].encode()
        self._rows[key] = (version, data)
        return data, version

    def _put_row(self, key, data, expected, create):
        if expected is None and not create:
            current = self._get_row(key)
            if current is None:
                return self._put_row(key, data, None, True)
            expected = current[1]
        text = data.decode()
        if expected is None:
            status, _ = self._json('POST', '/rest/v1/cloud_manifest',
                                   {'key': key, 'version': 1, 'body': text},
                                   {'Prefer': 'return=minimal'})
            version = '1'
        else:
            version = str(int(expected) + 1)
            status, body = self._json('PATCH', self._row_filter(key) + '&version=eq.' + str(int(expected)),
                                      {'version': int(version), 'body': text,
                                       'updated_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                                      {'Prefer': 'return=representation'})
            if status == 200 and not json.loads(body):
                status = 409
        if status == 409:
            raise CloudConflict('Data di server berubah. Muat ulang dan ulangi tindakan.')
        if status not in {200, 201, 204}:
            raise OSError(UNAVAILABLE)
        self._rows[key] = (version, data)
        return version

    # Storage objects.
    def _download(self, key):
        status, body = self._request('GET', f'/storage/v1/object/authenticated/{self.bucket}/'
                                     + urllib.parse.quote(key))
        if status == 200:
            return body
        if status in {400, 404} and b'not' in body.lower() and b'found' in body.lower():
            return None
        raise OSError(UNAVAILABLE)

    def _stream(self, key, write):
        request = urllib.request.Request(
            self.url + f'/storage/v1/object/authenticated/{self.bucket}/' + urllib.parse.quote(key),
            headers={'apikey': self.key, 'Authorization': 'Bearer ' + self.key})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    write(block)
            return True
        except urllib.error.HTTPError as error:
            body = error.read()
            if error.code in {400, 404} and b'not' in body.lower() and b'found' in body.lower():
                return False
            raise OSError(UNAVAILABLE) from error
        except OSError as error:
            raise OSError(UNAVAILABLE) from error

    def download_to(self, key, path):
        """Write an object to disk block by block, so its size never reaches memory."""
        if not self.url or not self.key:
            raise OSError(UNAVAILABLE)
        path = Path(path)
        with path.open('wb') as handle:
            if not self._stream(key, handle.write):
                return False
        if path.stat().st_size > 4096:
            return True
        head = path.read_bytes()
        if head.startswith(PARTS_MAGIC):
            index = json.loads(head[len(PARTS_MAGIC):])
            with path.open('wb') as handle:
                for number in range(index['parts']):
                    if not self._stream(part_key(key, number), handle.write):
                        return False
            if path.stat().st_size != index['size']:
                raise CloudIntegrityError('Incomplete multipart object: ' + key)
        return True

    def _upload(self, key, data, create):
        status, body = self._request('POST', self._object_path(key), data, {
            'Content-Type': 'application/octet-stream', 'x-upsert': 'false' if create else 'true'})
        if status in {400, 409} and (b'Duplicate' in body or b'already exists' in body):
            raise CloudConflict('Object already exists')
        if status != 200:
            raise OSError(UNAVAILABLE)

    def _remember(self, key, data):
        if '/objects/' not in key or len(data) > PART_SIZE:
            return
        self._objects[key] = data
        self._object_bytes += len(data)
        while self._object_bytes > 64 * 1024 * 1024:
            self._object_bytes -= len(self._objects.popitem(last=False)[1])

    def get(self, key):
        if self._manifest_key(key):
            return self._get_row(key)
        if key in self._objects:
            self._objects.move_to_end(key)
            return self._objects[key], 'immutable'
        data = self._download(key)
        if data is None:
            return None
        if data.startswith(PARTS_MAGIC):
            index = json.loads(data[len(PARTS_MAGIC):])
            pieces = [self._download(part_key(key, i)) for i in range(index['parts'])]
            if any(piece is None for piece in pieces):
                return None
            data = b''.join(pieces)
            if len(data) != index['size']:
                raise CloudIntegrityError('Incomplete multipart object: ' + key)
        self._remember(key, data)
        return data, 'immutable'

    def put(self, key, data, *, expected=None, create=False):
        if self._manifest_key(key):
            return self._put_row(key, data, expected, create)
        return self.put_file(key, io.BytesIO(data), expected=expected, create=create)

    def put_file(self, key, path, *, expected=None, create=False):
        if self._manifest_key(key):
            return self._put_row(key, Path(path).read_bytes(), expected, create)
        with (open(path, 'rb') if isinstance(path, (str, os.PathLike)) else contextlib.nullcontext(path)) as handle:
            block, pending = handle.read(PART_SIZE), handle.read(PART_SIZE)
            if not pending:
                self._upload(key, block, create)
                return 'immutable'
            if create and self._download(key) is not None:
                raise CloudConflict('Object already exists')
            # Parts are written first; the index makes the object visible last.
            index = size = 0
            while block:
                self._upload(part_key(key, index), block, False)
                index, size = index + 1, size + len(block)
                block, pending = pending, handle.read(PART_SIZE) if pending else b''
            self._upload(key, PARTS_MAGIC + json.dumps({'parts': index, 'size': size}).encode(), create)
        return 'immutable'

    def delete(self, keys):
        keys = list(keys)
        for start in range(0, len(keys), 100):
            status, _ = self._json('DELETE', f'/storage/v1/object/{self.bucket}', {'prefixes': keys[start:start + 100]})
            if status != 200:
                raise OSError(UNAVAILABLE)
        for key in keys:
            self._objects.pop(key, None)

    def stale(self, prefix, older_than):
        status, body = self._json('POST', '/rest/v1/rpc/stale_objects',
                                  {'name_prefix': prefix, 'older_than': older_than})
        if status != 200:
            raise OSError(UNAVAILABLE)
        return json.loads(body)

    def signed_upload_url(self, key):
        # Targets belong to one browser session; upsert lets a failed upload retry.
        status, body = self._request('POST', f'/storage/v1/object/upload/sign/{self.bucket}/'
                                     + urllib.parse.quote(key), headers={'x-upsert': 'true'})
        if status != 200:
            raise OSError(UNAVAILABLE)
        return self.url + '/storage/v1' + json.loads(body)['url']

    def signed_download_url(self, key, filename, expires=3600):
        status, body = self._json('POST', f'/storage/v1/object/sign/{self.bucket}/' + urllib.parse.quote(key),
                                  {'expiresIn': expires})
        if status != 200:
            raise OSError(UNAVAILABLE)
        return (self.url + '/storage/v1' + json.loads(body)['signedURL']
                + '&' + urllib.parse.urlencode({'download': filename}))


class CloudState:
    def __init__(self, root, store, prefix='gtex'):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.prefix = prefix.rstrip('/')
        self.baseline = {}
        self.loaded = False
        self.remote_exists = False
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
        key = self.prefix + '/objects/' + digest
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.cloud.part')
        try:
            if hasattr(self.store, 'download_to'):
                found = self.store.download_to(key, temporary) and _file_digest(temporary) == digest
            else:
                data = self.store.get(key)
                found = data is not None and hashlib.sha256(data[0]).hexdigest() == digest
                if found:
                    temporary.write_bytes(data[0])
            if not found:
                raise CloudIntegrityError('Durable artifact checksum mismatch: ' + name)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        # Track single installs (e.g. a daemon fetching a stop command) so a
        # later pull removes the file once it is deleted remotely.
        self.baseline[name] = digest

    def pull(self):
        with self.lock:
            manifest, etag = self._manifest()
            self.remote_exists = etag is not None
            files = manifest['files']
            # Validate every remote path before touching the local working copy.
            for name in files:
                _safe(self.root, name)
            baseline = {}
            for name, digest in files.items():
                target = _safe(self.root, name)
                # Left out of the baseline too, or publish would read it as a local deletion.
                if _deferred(name) and not target.is_file():
                    continue
                if self.baseline.get(name) != digest or not target.is_file():
                    self._install(name, digest)
                baseline[name] = digest
            for name in self.baseline.keys() - files.keys():
                _safe(self.root, name).unlink(missing_ok=True)
            self.baseline = baseline
            self.loaded = True
            return manifest

    def ensure(self, directory):
        """Fetch the deferred files of one directory before a web instance reads them."""
        directory = Path(directory).resolve()
        if not directory.is_relative_to(self.root):
            return
        prefix = directory.relative_to(self.root).as_posix() + '/'
        with self.lock:
            manifest, _ = self._manifest()
            for name, digest in manifest['files'].items():
                if name.startswith(prefix) and not _safe(self.root, name).is_file():
                    self._install(name, digest)

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
            # Most reruns change nothing; skip the manifest round trip. Workers
            # still read it so a lost lease fences them out.
            if not changes and lease_owner is None and self.remote_exists:
                return
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
                    self.remote_exists = True
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

    def collect_garbage(self, *, grace='1 hour', transient='1 day'):
        """Delete unreferenced artifact versions and expired uploads/exports."""
        manifest, _ = self._manifest()
        live = set(manifest['files'].values())
        objects = self.prefix + '/objects/'
        stale = [name for name in self.store.stale(objects, grace)
                 if name[len(objects):].split('.part')[0] not in live]
        for prefix in ['incoming/', 'exports/' + self.prefix + '/']:
            stale += self.store.stale(prefix, transient)
        if stale:
            self.store.delete(stale)
        return len(stale)

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

    @contextmanager
    def script(self):
        """Sync around one page run without serializing every session on the instance.

        The lock is held only while pulling and publishing. Writes happen inside
        @operation, which takes its own short transaction, so rendering, parsing and
        prediction in one session no longer delay other sessions.
        """
        self.pull()
        try:
            yield self
        finally:
            self.publish()


_instance = None


def state():
    global _instance
    if _instance is None:
        root = Path(os.environ['GTEX_DESKTOP_STATE_DIR'])
        _instance = CloudState(root, SupabaseStore(), os.environ.get('GTEX_CLOUD_PREFIX', 'gtex'))
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
