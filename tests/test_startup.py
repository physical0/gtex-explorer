"""The first screen must not wait for libraries it does not use."""
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from gtex_explorer.cloud.state import CloudState, CloudIntegrityError
from test_cloud_state import MemoryStore, write

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_chooser_renders_without_the_heavy_libraries(tmp_path):
    # A separate process, because this test run has already imported them.
    code = '''
import json, os, sys
os.environ['GTEX_DESKTOP_STATE_DIR'] = sys.argv[1]
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
with patch('gtex_explorer.ui.workspace_ui.preload_libraries'):
    app = AppTest.from_file('app.py', default_timeout=120).run()
print(json.dumps({'error': bool(app.exception), 'titles': [t.value for t in app.title],
                  'loaded': [m for m in ['xgboost', 'sklearn', 'altair'] if m in sys.modules]}))
'''
    done = subprocess.run([sys.executable, '-c', code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=180)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == {'error': False, 'titles': ['Pilih workspace'], 'loaded': []}


def test_new_instance_downloads_saved_files_in_parallel(tmp_path):
    store = MemoryStore()
    writer = CloudState(tmp_path / 'writer', store)
    for number in range(12):
        write(writer.root, f'runs/run_x/file_{number}.json', str(number).encode())
    writer.publish()

    active = threading.Barrier(2, timeout=5)
    real_get = store.get

    def get(key):
        # Only passes when two downloads are in flight at the same time.
        if '/objects/' in key:
            active.wait()
        return real_get(key)

    store.get = get
    fresh = CloudState(tmp_path / 'fresh', store)
    fresh.pull()
    assert (fresh.root / 'runs/run_x/file_11.json').read_bytes() == b'11'
    store.get = real_get
    # Everything counts as synced afterwards, so nothing is downloaded or published again.
    before = store.counter
    fresh.publish()
    assert store.counter == before


def test_failed_parallel_download_is_reported(tmp_path):
    store = MemoryStore()
    writer = CloudState(tmp_path / 'writer', store)
    write(writer.root, 'a.json', b'a')
    write(writer.root, 'b.json', b'b')
    writer.publish()
    manifest = json.loads(store.objects['gtex/state.json'][0])
    del store.objects['gtex/objects/' + manifest['files']['a.json']]
    with pytest.raises(CloudIntegrityError):
        CloudState(tmp_path / 'fresh', store).pull()
