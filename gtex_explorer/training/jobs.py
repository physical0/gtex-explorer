"""One local worker, atomic job records and explicit resume/cancel controls."""

import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil

from gtex_explorer.platform.cloud.state import operation, remote_client, state
from gtex_explorer.platform.paths import APP_ROOT
from gtex_explorer.platform.workspaces import coordination_directory, workspace_roots
from gtex_explorer.training.config import TrainingConfig, configuration_fingerprint
from gtex_explorer.training.core import runtime_versions
from gtex_explorer.training.data import (
    atomic_replace_file,
    discard_directory,
    load_training_data,
    read_json_file,
    run_split,
    safe_path,
)

# --- Job records ---


@dataclass
class JobRecord:
    """The saved status of one training run."""

    run_id: str
    status: str
    error: dict | None
    details: dict


FINALIZING_STAGES = {"saving_results", "verifying_results"}
ACTIVE_STATUSES = {"queued", "running", "paused", "cancelling"}
# Data and configuration errors fail the same way on every retry.
DETERMINISTIC_ERRORS = {"SelectorEligibilityError", "ValueError", "TypeError", "KeyError"}


def can_resume(record: "JobRecord") -> bool:
    """Return whether a stopped or failed run can continue."""
    if record.status == "cancelled":
        return True
    return record.status == "failed" and (record.error or {}).get("type") not in DETERMINISTIC_ERRORS


def resolve_job_data(saved, state_root, fallback):
    """Return the data folder that a run was started with."""
    reference = saved.get("data_reference")
    if reference:
        if reference["kind"] == "workspace":
            return safe_path(coordination_directory(state_root), reference["path"])
        if reference["kind"] == "bundled":
            return APP_ROOT / "training_data"
        raise ValueError("Invalid durable dataset reference")
    return Path(saved.get("data_root", fallback)).resolve()


# --- Paths, files, and locks ---


def job_path(run_id: str, state_root: Path) -> Path:
    """Return a run's folder after checking its ID."""
    if not re.fullmatch(r"run_[0-9a-f]{32}", run_id):
        raise ValueError("Invalid generated run ID")
    return Path(state_root) / "runs" / run_id


def atomic_json(path: Path, value: dict):
    """Write a JSON file in one step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, allow_nan=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        atomic_replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def process_lock(path: Path):
    """Hold a lock that also applies across processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        handle.close()  # OS releases ownership even if a worker crashes.


# --- Job status ---


@operation
def has_active_run(state_root: Path) -> bool:
    """Return whether this workspace has a run in progress."""
    for path in (Path(state_root) / "runs").glob("run_*/status.json"):
        try:
            if read_json_file(path).get("status") in ACTIVE_STATUSES:
                return True
        except (OSError, ValueError):
            continue
    return False


@operation
def read_job(run_id: str, state_root: Path) -> JobRecord:
    """Return a run's current record."""
    if remote_client():
        from gtex_explorer.platform.cloud.jobs import record

        return record(run_id, state_root)
    path = job_path(run_id, state_root) / "status.json"
    value = read_json_file(path)
    if value["status"] in {"running", "cancelling", "queued", "paused"} and not (
        value["status"] == "queued" and time.time() - value["updated"] < 30
    ):
        try:
            coordinator = coordination_directory(state_root)
            with process_lock(coordinator / "launch.lock"):
                owner_file = coordinator / "worker_owner.json"
                owner = json.loads(owner_file.read_text(encoding="utf-8")) if owner_file.exists() else None
                unrelated = owner and (
                    owner.get("run_id") != run_id
                    or Path(owner.get("state_root", "")).resolve() != Path(state_root).resolve()
                )
                if unrelated:
                    value = _recover_dead_job(path)
                else:
                    with process_lock(coordinator / "worker.lock"):
                        value = _recover_dead_job(path)
        except OSError:
            pass  # A live worker or launcher still owns its OS lock.
    if value["status"] in {"running", "queued", "cancelling"} and (path.parent / "cancel.json").exists():
        value = {**value, "status": "cancelling"}
    return JobRecord(run_id, value["status"], value.get("error"), value)


def _recover_dead_job(path):
    """Correct the status of a run whose worker has died."""
    value = read_json_file(path)
    if value["status"] in {"running", "cancelling", "queued", "paused"}:
        if (path.parent / "cancel.json").exists():
            value.update(status="cancelled", stage="cancelled", error=None)
        else:
            value.update(
                status="failed",
                error={"type": "WorkerStopped", "message": "Worker stopped. Resume the verified completed stages."},
            )
        atomic_json(path, value)
    return value


# --- Starting jobs ---


def _launch(run_id, data_root, state_root):
    """Start the worker process for a run."""
    if remote_client():
        return  # The daemon consumes the durably queued record after publication.
    command = [sys.executable]
    if getattr(sys, "frozen", False):
        command += ["--training-worker", run_id, "--training-data", str(data_root), "--training-state", str(state_root)]
    else:
        command += [
            "-m",
            "gtex_explorer.training.worker",
            "--run-id",
            run_id,
            "--data",
            str(data_root),
            "--state",
            str(state_root),
        ]
    log_path = job_path(run_id, state_root) / "worker.log"
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.Popen(
            command,
            stdout=log,
            stderr=log,
            cwd=APP_ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )


def _assert_idle(state_root):
    """Refuse to start while another run is active in any workspace."""
    if remote_client():
        from gtex_explorer.platform.cloud.jobs import assert_idle

        return assert_idle(state_root)
    coordinator = coordination_directory(state_root)
    try:
        with process_lock(coordinator / "worker.lock"):
            pass
    except OSError as error:
        raise ValueError("A training worker is already running") from error
    paths = [path for _, root in workspace_roots(coordinator) for path in (root / "runs").glob("run_*/status.json")]
    for path in paths:
        status = read_json_file(path)
        if status["status"] == "queued" and time.time() - status["updated"] < 30:
            raise ValueError("A training worker is starting")
        if status["status"] in {"queued", "running", "cancelling", "paused"}:
            if (path.parent / "cancel.json").exists():
                status.update(status="cancelled", stage="cancelled", error=None)
            else:
                status.update(
                    status="failed",
                    error={"type": "WorkerStopped", "message": "Worker stopped. Resume the verified completed stages."},
                )
            atomic_json(path, status)


@operation
def start_job(config: TrainingConfig, data_root: Path, state_root: Path) -> JobRecord:
    """Save a new run and start it."""
    data_root, state_root = Path(data_root).resolve(), Path(state_root).resolve()
    if remote_client():
        state().ensure(data_root)
    data = load_training_data(data_root)
    _, _, split = run_split(data, config)
    identity = configuration_fingerprint(config, data.fingerprint, split, runtime_versions())
    coordinator = coordination_directory(state_root)
    with process_lock(coordinator / "launch.lock"):
        _assert_idle(state_root)
        run_id = "run_" + uuid.uuid4().hex
        root = job_path(run_id, state_root)
        root.mkdir(parents=True)
        dataset = {
            "sample_count": len(data.sample_ids),
            "donor_count": len(set(data.donor_ids)),
            "class_count": len(data.classes),
            "input_gene_count": len(data.raw_gene_names),
            "candidate_count": data.manifest["candidate_count"],
            "source": data.manifest.get("source", {"kind": "notebook", "expression_file": "Data eksperimen notebook"}),
        }
        portable = {}
        if remote_client():
            if data_root.is_relative_to(coordinator):
                portable["data_reference"] = {
                    "kind": "workspace",
                    "path": data_root.relative_to(coordinator).as_posix(),
                }
            elif data_root == APP_ROOT / "training_data":
                portable["data_reference"] = {"kind": "bundled"}
            else:
                raise ValueError("Cloud training data must belong to a workspace or bundled reference.")
        # File times are reset when a hosted instance restores state, so keep the real start time.
        atomic_json(
            root / "config.json",
            {
                "config": config.to_dict(),
                "identity": identity,
                **portable,
                "created": time.time(),
                "data_root": str(data_root),
                "dataset": dataset,
                "data_fingerprint": data.fingerprint,
                "split_hash": split,
            },
        )
        atomic_json(
            root / "status.json",
            {"status": "queued", "stage": "queued", "updated": time.time(), "error": None, "cv_summary": None},
        )
        try:
            atomic_json(
                coordinator / "worker_owner.json",
                {"run_id": run_id, "state_root": str(state_root), "updated": time.time()},
            )
            _launch(run_id, data_root, state_root)
        except Exception as error:
            atomic_json(
                root / "status.json",
                {
                    "status": "failed",
                    "stage": "launch",
                    "updated": time.time(),
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
            raise
    return read_job(run_id, state_root)


# --- Removing jobs ---


@operation
def delete_run(run_id: str, state_root: Path) -> None:
    """Permanently remove a finished run with its model and gene panels.

    Its dataset stays, because other runs in the workspace may share it.
    """
    root = job_path(run_id, state_root)
    # Outside the lock first, because read_job needs it to recover a dead worker's status.
    read_job(run_id, state_root)
    coordinator = coordination_directory(state_root)
    try:
        with process_lock(coordinator / "launch.lock"):
            if read_job(run_id, state_root).status in ACTIVE_STATUSES:
                raise ValueError("Pelatihan masih berjalan. Hentikan pelatihan sebelum menghapus.")
            # The run goes first: a model without its run is already hidden from the gallery.
            discard_directory(root, coordinator)
            model = Path(state_root) / "models" / run_id
            if model.is_dir():
                discard_directory(model, coordinator)
    except OSError as error:
        raise ValueError("Hasil pelatihan sedang digunakan proses lain. Coba kembali beberapa saat lagi.") from error


# --- Stopping, pausing, and resuming ---


@operation
def request_cancel(run_id: str, state_root: Path) -> JobRecord:
    """Ask a run to stop."""
    if remote_client():
        from gtex_explorer.platform.cloud.jobs import command

        return command(run_id, state_root, "stop")
    with process_lock(job_path(run_id, state_root) / "control.lock"):
        record = read_job(run_id, state_root)
        if record.status == "running" and record.details.get("stage") in FINALIZING_STAGES:
            return record  # Finish the atomic saving/verification update first.
        if record.status in {"queued", "running", "cancelling", "paused"}:
            atomic_json(job_path(run_id, state_root) / "cancel.json", {"cancel": True})
            if record.status == "paused":
                # The cancellation watcher is suspended too. Wake it only after
                # the marker is durable, so the fit cannot publish a partial model.
                _unpause(run_id, state_root, record)
    return read_job(run_id, state_root)


def _verified_worker(run_id: str, state_root: Path):
    """Use PID + creation time + worker command, never a process name lookup."""
    try:
        identity = json.loads((job_path(run_id, state_root) / "worker.json").read_text(encoding="utf-8"))
        process = psutil.Process(identity["pid"])
        if (
            identity["run_id"] != run_id
            or not identity.get("pause_supported")
            or process.pid == os.getpid()
            or process.create_time() != identity["created"]
            or not process.is_running()
        ):
            raise ValueError("Identitas worker tidak sesuai. Muat ulang hasil pelatihan.")
        command = process.cmdline()
        if run_id not in command or not (
            "--training-worker" in command
            or any(
                (Path(argument).name == "training_worker.py" or argument == "gtex_explorer.training.worker")
                for argument in command
            )
        ):
            raise ValueError("Proses bukan worker pelatihan yang diminta.")
        return process
    except (OSError, KeyError, psutil.Error) as error:
        raise ValueError("Worker tidak tersedia untuk jeda. Muat ulang hasil pelatihan.") from error


@operation
def pause_job(run_id: str, state_root: Path) -> JobRecord:
    """Pause a running worker."""
    if remote_client():
        from gtex_explorer.platform.cloud.jobs import command

        return command(run_id, state_root, "pause")
    root = job_path(run_id, state_root)
    with process_lock(root / "control.lock"):
        record = read_job(run_id, state_root)
        if record.status == "paused":
            return record
        if record.status != "running":
            raise ValueError("Hanya worker pelatihan yang sedang berjalan dapat dijeda.")
        if record.details.get("stage") in FINALIZING_STAGES:
            raise ValueError("Tunggu penyimpanan dan verifikasi hasil selesai.")
        process = _verified_worker(run_id, state_root)
        try:
            process.suspend()
            # Read again after suspension so an iteration/status update in flight
            # cannot be mistaken for the state at the earlier button click.
            value = read_json_file(root / "status.json")
            if (
                value["status"] != "running"
                or (root / "cancel.json").exists()
                or value.get("stage") in FINALIZING_STAGES
            ):
                process.resume()
                return read_job(run_id, state_root)
            now = time.time()
            elapsed = value.get("elapsed_seconds", 0) + max(0, now - value["updated"])
            atomic_json(
                root / "status.json",
                {**value, "status": "paused", "paused_at": now, "updated": now, "elapsed_seconds": elapsed},
            )
        except Exception:
            # A failed status write must not leave an invisible suspended worker.
            process.resume()
            raise
    return read_job(run_id, state_root)


def _unpause(run_id, state_root, record):
    """Continue a paused worker and record how long it was paused."""
    root = job_path(run_id, state_root)
    process = _verified_worker(run_id, state_root)
    value = dict(record.details)
    paused_at = value.pop("paused_at")
    clock_path = root / "pause_clock.json"
    previous = json.loads(clock_path.read_text(encoding="utf-8")) if clock_path.exists() else {"seconds": 0}
    try:
        atomic_json(clock_path, {"seconds": previous["seconds"] + max(0, time.time() - paused_at)})
        # Restore status before waking the worker, which becomes its only writer again.
        atomic_json(root / "status.json", {**value, "status": "running", "updated": time.time()})
        process.resume()
    except Exception:
        atomic_json(clock_path, previous)
        atomic_json(root / "status.json", record.details)
        raise


@operation
def unpause_job(run_id: str, state_root: Path) -> JobRecord:
    """Continue a paused run."""
    if remote_client():
        from gtex_explorer.platform.cloud.jobs import command

        return command(run_id, state_root, "resume")
    with process_lock(job_path(run_id, state_root) / "control.lock"):
        record = read_job(run_id, state_root)
        if record.status != "paused":
            raise ValueError("Worker tidak sedang dijeda. Muat ulang hasil pelatihan.")
        _unpause(run_id, state_root, record)
    return read_job(run_id, state_root)


@operation
def resume_job(run_id: str, data_root: Path, state_root: Path) -> JobRecord:
    """Continue a stopped run from its checkpoint."""
    coordinator = coordination_directory(state_root)
    with process_lock(coordinator / "launch.lock"):
        _assert_idle(state_root)
        record = read_job(run_id, state_root)
        if record.status not in {"failed", "cancelled"}:
            raise ValueError("Only stopped runs can resume")
        root = job_path(run_id, state_root)
        # The worker validates full identity before reusing any output.
        (root / "cancel.json").unlink(missing_ok=True)
        if remote_client():
            (root / "cloud_control.json").unlink(missing_ok=True)
        atomic_json(root / "status.json", {**record.details, "status": "queued", "updated": time.time(), "error": None})
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        atomic_json(
            coordinator / "worker_owner.json",
            {"run_id": run_id, "state_root": str(Path(state_root).resolve()), "updated": time.time()},
        )
        _launch(run_id, resolve_job_data(saved, state_root, data_root), Path(state_root).resolve())
    return read_job(run_id, state_root)
