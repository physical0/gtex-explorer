"""Copy an explicit resource allowlist, never the user's local installation state."""

import json
import shutil
import uuid

from gtex_explorer.platform.paths import APP_ROOT as ROOT


def prepare():
    """Copy only the deployable files into a clean folder."""
    target = ROOT / ".cloud_deploy" / uuid.uuid4().hex
    target.mkdir(parents=True)
    names = [
        "main.py",
        "cloud_app.py",
        "desktop.py",
        "cloud-config.toml",
        "requirements-cloud.txt",
        "Dockerfile.vercel",
        "vercel.json",
        ".dockerignore",
    ]
    for name in names:
        shutil.copy2(ROOT / name, target / name)
    for name in ["gtex_explorer", "scripts", "resources", "data", "assets", "model_bundle", "baseline_bundle"]:
        shutil.copytree(
            ROOT / name,
            target / name,
            ignore=shutil.ignore_patterns("node_modules", "__pycache__", "*.log", "*.part", ".env*"),
        )
    linked = json.loads((ROOT / ".vercel/project.json").read_text(encoding="utf-8"))
    if linked["projectId"] != "prj_tpGnifnPCMb2NBsp14wPOTyqiLbm" or linked["orgId"] != "team_1MGBNiUGaGYz2FQTj0kpPytz":
        raise ValueError("Unexpected deployment destination")
    (target / ".vercel").mkdir()
    (target / ".vercel/project.json").write_text(json.dumps(linked), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(prepare())
