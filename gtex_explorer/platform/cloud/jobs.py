"""Web instances submit durable commands. Only the daemon touches worker PIDs."""

import json
import time
import uuid

from gtex_explorer.platform.cloud.state import state


def record(run_id, state_root):
    """Read a training job record from the durable cloud state."""
    from gtex_explorer.training.jobs import JobRecord, job_path

    root = job_path(run_id, state_root)
    value = json.loads((root / "status.json").read_text(encoding="utf-8"))
    manifest, _ = state()._manifest()
    lease = manifest.get("lease") or {}
    if value["status"] in {"running", "paused", "cancelling"} and lease.get("expires", 0) < time.time():
        value = {
            **value,
            "status": "failed",
            "error": {
                "type": "WorkerStopped",
                "message": "Worker server berhenti. Continue from checkpoint menggunakan fold yang sudah tersimpan.",
            },
        }
    control = root / "cloud_control.json"
    if control.exists() and value["status"] in {"running", "paused", "queued"}:
        command = json.loads(control.read_text(encoding="utf-8"))
        if command["action"] == "stop" and command["id"] != value.get("control_id"):
            value = {**value, "status": "cancelling"}
    return JobRecord(run_id, value["status"], value.get("error"), value)


def command(run_id, state_root, action):
    """Queue a stop, pause, or resume command for the cloud worker."""
    from gtex_explorer.training.jobs import FINALIZING_STAGES, atomic_json, job_path

    current = record(run_id, state_root)
    if current.details.get("stage") in FINALIZING_STAGES:
        return current
    if current.status not in {"running", "paused", "queued", "cancelling"}:
        raise ValueError("Proses sudah berakhir. Muat ulang status pelatihan.")
    if action == "pause" and current.status != "running":
        raise ValueError("Pause tersedia ketika worker sedang training.")
    if action == "resume" and current.status != "paused":
        raise ValueError("Resume tersedia ketika worker sedang paused.")
    atomic_json(
        job_path(run_id, state_root) / "cloud_control.json",
        {"id": uuid.uuid4().hex, "action": action, "created": time.time()},
    )
    return record(run_id, state_root)


def assert_idle(state_root):
    """Refuse to continue while any workspace has an active training run."""
    from gtex_explorer.platform.workspaces import coordination_directory, workspace_roots

    for _, directory in workspace_roots(coordination_directory(state_root)):
        for path in (directory / "runs").glob("run_*/status.json"):
            if record(path.parent.name, directory).status in {"queued", "running", "paused", "cancelling"}:
                raise ValueError("Satu proses training masih aktif. Tunggu atau gunakan Stop pada Status proses.")
