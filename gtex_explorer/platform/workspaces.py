"""Named project roots. Display names never determine filesystem paths."""

import json
import re
import time
import uuid
from pathlib import Path

from gtex_explorer.platform.cloud.state import operation, remote_client, state

# --- Registry ---


def _registry(root):
    """Read the workspace list, or return the default one if it does not exist yet."""
    root = Path(root).resolve()
    path = root / "workspaces.json"
    if not path.exists():
        return {
            "schema_version": 1,
            "last_workspace_id": "default",
            "workspaces": [{"id": "default", "name": "Default", "include_bundled_models": True}],
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("workspaces"), list):
        raise ValueError("Registry workspace tidak valid.")
    ids, names = set(), set()
    for entry in value["workspaces"]:
        identity = entry.get("id", "")
        name = _name(entry.get("name"))
        if identity != "default" and not re.fullmatch(r"ws_[0-9a-f]{32}", identity):
            raise ValueError("ID workspace tidak valid.")
        if identity in ids or name.casefold() in names or type(entry.get("include_bundled_models")) is not bool:
            raise ValueError("Metadata workspace tidak valid.")
        ids.add(identity)
        names.add(name.casefold())
    if "default" not in ids or value.get("last_workspace_id") not in ids:
        raise ValueError("Workspace default tidak ditemukan.")
    return value


def _name(value):
    """Validate and clean a workspace name."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.strip()) > 80
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError("Nama workspace harus berisi 1–80 karakter tanpa baris baru.")
    return value.strip()


def list_workspaces(root):
    """Return every workspace."""
    return _registry(root)["workspaces"]


def workspace_directory(root, workspace_id):
    """Return a workspace's folder."""
    root = Path(root).resolve()
    if workspace_id not in {entry["id"] for entry in list_workspaces(root)}:
        raise ValueError("Workspace tidak ditemukan.")
    path = root if workspace_id == "default" else root / "workspaces" / workspace_id
    if not path.resolve().is_relative_to(root):
        raise ValueError("Lokasi workspace tidak valid.")
    return path


# --- Changing workspaces ---


@operation
def create_workspace(root, name, *, include_bundled_models=True):
    """Create a workspace and return its entry."""
    from gtex_explorer.training.jobs import atomic_json, process_lock

    name = _name(name)
    if type(include_bundled_models) is not bool:
        raise ValueError("Pilihan model bawaan tidak valid.")
    root = Path(root).resolve()
    with process_lock(root / "gtex_explorer.workspaces.lock"):
        registry = _registry(root)
        if name.casefold() in {entry["name"].casefold() for entry in registry["workspaces"]}:
            raise ValueError("Nama workspace sudah digunakan.")
        entry = {
            "id": "ws_" + uuid.uuid4().hex,
            "name": name,
            "include_bundled_models": include_bundled_models,
            "created": time.time(),
        }
        directory = root / "workspaces" / entry["id"]
        directory.mkdir(parents=True)
        atomic_json(directory / "workspace.json", {"schema_version": 1, "id": entry["id"]})
        registry["workspaces"].append(entry)
        atomic_json(root / "workspaces.json", registry)
    return entry


@operation
def rename_workspace(root, workspace_id, name):
    """Change the name of a workspace."""
    from gtex_explorer.training.jobs import atomic_json, process_lock

    name = _name(name)
    root = Path(root).resolve()
    with process_lock(root / "gtex_explorer.workspaces.lock"):
        registry = _registry(root)
        entry = next((x for x in registry["workspaces"] if x["id"] == workspace_id), None)
        if entry is None:
            raise ValueError("Workspace tidak ditemukan.")
        if any(x["id"] != workspace_id and x["name"].casefold() == name.casefold() for x in registry["workspaces"]):
            raise ValueError("Nama workspace sudah digunakan.")
        entry["name"] = name
        atomic_json(root / "workspaces.json", registry)


@operation
def delete_workspace(root, workspace_id):
    """Permanently remove a workspace with its runs, models and datasets."""
    from gtex_explorer.training.data import discard_directory
    from gtex_explorer.training.jobs import ACTIVE_STATUSES, atomic_json, process_lock, read_job

    root = Path(root).resolve()
    if workspace_id == "default":
        raise ValueError("Workspace Default tidak dapat dihapus.")
    directory = workspace_directory(root, workspace_id)
    runs = [path.parent.name for path in (directory / "runs").glob("run_*/status.json")]
    # Outside the locks first, because read_job needs them to recover a dead worker's status.
    for run_id in runs:
        read_job(run_id, directory)
    try:
        with process_lock(root / "launch.lock"), process_lock(root / "gtex_explorer.workspaces.lock"):
            if any(read_job(run_id, directory).status in ACTIVE_STATUSES for run_id in runs):
                raise ValueError("Workspace masih memiliki pelatihan aktif. Hentikan pelatihan sebelum menghapus.")
            registry = _registry(root)
            remaining = [entry for entry in registry["workspaces"] if entry["id"] != workspace_id]
            atomic_json(
                root / "workspaces.json",
                {
                    **registry,
                    "workspaces": remaining,
                    "last_workspace_id": (
                        "default" if registry["last_workspace_id"] == workspace_id else registry["last_workspace_id"]
                    ),
                },
            )
            try:
                discard_directory(directory, root)
            except OSError:
                atomic_json(root / "workspaces.json", registry)
                raise
    except OSError as error:
        raise ValueError("Workspace sedang digunakan proses lain. Coba kembali beberapa saat lagi.") from error
    if remote_client():
        state().forget(directory)


@operation
def remember_workspace(root, workspace_id):
    """Record the workspace to preselect next time."""
    from gtex_explorer.training.jobs import atomic_json, process_lock

    root = Path(root).resolve()
    with process_lock(root / "gtex_explorer.workspaces.lock"):
        registry = _registry(root)
        if workspace_id not in {entry["id"] for entry in registry["workspaces"]}:
            raise ValueError("Workspace tidak ditemukan.")
        registry["last_workspace_id"] = workspace_id
        atomic_json(root / "workspaces.json", registry)


# --- Lookups ---


def last_workspace(root):
    """Return the ID of the last used workspace."""
    return _registry(root)["last_workspace_id"]


def coordination_directory(state_root):
    """Keep one launch/worker lock across all registered project roots."""
    path = Path(state_root).resolve()
    if path.parent.name == "workspaces" and re.fullmatch(r"ws_[0-9a-f]{32}", path.name):
        root = path.parent.parent
        if workspace_directory(root, path.name) != path:
            raise ValueError("Lokasi workspace tidak valid.")
        marker = json.loads((path / "workspace.json").read_text(encoding="utf-8"))
        if marker.get("schema_version") != 1 or marker.get("id") != path.name:
            raise ValueError("Identitas workspace tidak valid.")
        return root
    return path


def workspace_roots(root):
    """Return each workspace with its folder."""
    return [(entry, workspace_directory(root, entry["id"])) for entry in list_workspaces(root)]


def bundled_models_enabled(state_root):
    """Return whether this workspace includes the original models."""
    root = coordination_directory(state_root)
    if root == Path(state_root).resolve():
        return True  # Legacy/default state keeps its existing reference models.
    return next(
        entry["include_bundled_models"] for entry in list_workspaces(root) if entry["id"] == Path(state_root).name
    )
