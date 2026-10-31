import json
import os
from pathlib import Path
import subprocess
import sys


def test_desktop_training_check_runs_real_worker_without_changing_original(tmp_path):
    root = Path(__file__).resolve().parents[1]
    import hashlib
    original = (root / 'model_bundle' / 'xgboost_model.json').read_bytes()
    result = subprocess.run([sys.executable, str(root / 'desktop.py'), '--check-training'],
                            cwd=root, env={**os.environ, 'GTEX_DESKTOP_STATE_DIR': str(tmp_path)},
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['training_worker'] == 'ready'
    assert report['classes'] == 13
    assert report['genes'] == 100
    assert report['boosting_rounds'] == 2
    assert hashlib.sha256((root / 'model_bundle' / 'xgboost_model.json').read_bytes()).digest() == hashlib.sha256(original).digest()
