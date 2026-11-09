"""Names and dates that describe a saved training run."""

import json
import time
from pathlib import Path

METHODS = {
    "l1": "Regresi logistik L1",
    "anova": "ANOVA",
    "mutual_information": "Mutual Information",
    "baseline": "Baseline - tanpa seleksi supervised",
}


def run_created(path):
    """Start time saved with the run; older runs fall back to the file time."""
    config = Path(path) / "config.json"
    try:
        created = json.loads(config.read_text(encoding="utf-8")).get("created")
        if isinstance(created, (int, float)):
            return float(created)
    except (OSError, ValueError):
        pass
    try:
        return config.stat().st_mtime
    except OSError:
        return 0.0


def method_description(config):
    """Describe the selection method of a run."""
    text = METHODS.get(config["selector"], config["selector"])
    if config["selector"] == "l1":
        text += f" - C = {config['l1_c']:g}"
    elif config["selector"] == "mutual_information":
        text += f" - n_neighbors = {config['mi_neighbors']}"
    return f"{text} - {config['top_k']:,} gen".replace(",", ".")


def run_label(path):
    """Build the label of a run in selection lists."""
    try:
        saved = json.loads((path / "config.json").read_text(encoding="utf-8"))
        dataset = saved.get("dataset", {})
        filename = dataset.get("source", {}).get("expression_file", "Data notebook")
        stamp = time.strftime("%d/%m %H:%M", time.localtime(run_created(path)))
        return f"{stamp} - {method_description(saved['config'])} - {filename} - {path.name[-6:]}"
    except (OSError, ValueError, KeyError):
        return f"Riwayat tidak dapat dibaca - {path.name[-6:]}"
