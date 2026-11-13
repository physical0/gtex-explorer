"""Shared bundled-resource root for source and packaged execution."""

from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = APP_ROOT / "training_data"
