"""Windows desktop entry point. The UI and trained model run entirely locally."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import ProxyHandler, build_opener

from gtex_explorer.platform.paths import APP_ROOT
TITLE = "GTEx Tissue Explorer"


def state_directory() -> Path:
    """Keep runtime files away from packaged, potentially read-only resources."""
    override = os.environ.get("GTEX_DESKTOP_STATE_DIR")
    base = Path(override) if override else Path(os.environ.get("LOCALAPPDATA", str(APP_ROOT))) / "GTExTissueExplorer"
    base.mkdir(parents=True, exist_ok=True)
    return base


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def enforce_offline_network() -> None:
    """Deny external DNS and TCP connections in this Python process only."""
    resolve = socket.getaddrinfo
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def local_resolve(host, *args, **kwargs):
        if host is not None and not is_loopback(str(host)):
            raise OSError("Offline mode: external DNS is disabled.")
        return resolve(host, *args, **kwargs)

    def local_connect(sock, address):
        if isinstance(address, tuple) and not is_loopback(str(address[0])):
            raise OSError("Offline mode: only loopback connections are allowed.")
        return connect(sock, address)

    def local_connect_ex(sock, address):
        if isinstance(address, tuple) and not is_loopback(str(address[0])):
            raise OSError("Offline mode: only loopback connections are allowed.")
        return connect_ex(sock, address)

    socket.getaddrinfo = local_resolve
    socket.socket.connect = local_connect
    socket.socket.connect_ex = local_connect_ex


class OfflineHeaders:
    """Restrict frontend resources and identify our own local server."""
    def __init__(self, app, token: str, port: int):
        self.app = app
        self.token = token
        self.policy = (
            "default-src 'self'; "
            f"connect-src 'self' ws://127.0.0.1:{port}; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; font-src 'self' data:; "
            "worker-src 'self' blob:; frame-src 'self' blob:; "
            "object-src 'none'; base-uri 'self'; form-action 'self'"
        )

    async def __call__(self, scope, receive, send):
        async def local_send(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", []).extend([
                    (b"content-security-policy", self.policy.encode("ascii")),
                    (b"x-gtex-desktop", self.token.encode("ascii")),
                    (b"referrer-policy", b"no-referrer"),
                ])
            await send(message)
        await self.app(scope, receive, local_send)


def check_resources() -> dict:
    """Check packaged artifacts and run real demo inference, without retraining."""
    import streamlit
    from gtex_explorer.inference.model_bundle import MODEL_DIR, load_model_bundle
    from gtex_explorer.inference.audit import audit_and_align_input
    from gtex_explorer.experiments.reference_tables import load_reference_tables
    from gtex_explorer.inference.parsing import parse_expression_file
    from gtex_explorer.inference.prediction import predict_tissues
    if not hasattr(streamlit, "App"):
        raise RuntimeError("Versi desktop memerlukan Streamlit 1.63.0. Instal requirements-desktop.txt.")
    manifest = json.loads((MODEL_DIR / "bundle_manifest.json").read_text(encoding="utf8"))
    model_hash = hashlib.sha256((MODEL_DIR / "xgboost_model.json").read_bytes()).hexdigest()
    if manifest.get("status") != "validated" or model_hash != manifest.get("xgboost_model_sha256"):
        raise ValueError("Bundle model tidak lolos pemeriksaan integritas.")
    load_reference_tables()
    for name in ["brand_dna.png", "anatomy_front.png", "anatomy_left.png", "anatomy_right.png", "anatomy_neutral.png"]:
        if not (APP_ROOT / "assets" / name).is_file():
            raise FileNotFoundError(f"Aset lokal tidak ditemukan: {name}")
    preprocessor, classifier = load_model_bundle()
    with (APP_ROOT / "data" / "demo_tpm.csv").open("rb") as handle:
        matrix = parse_expression_file(handle, "demo_tpm.csv")
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    if not audit.ready:
        raise ValueError(f"Data contoh tidak lolos audit: {audit.message}")
    predictions, probabilities = predict_tissues(aligned, preprocessor, classifier)
    if not probabilities.sum(axis=1).between(0.99999, 1.00001).all():
        raise ValueError("Probabilitas prediksi tidak valid.")
    return {"status": "ready", "offline": True, "classes": len(preprocessor["classes"]),
            "genes": len(preprocessor["selected_genes"]), "demo_samples": len(predictions)}


def serve(port: int, token: str) -> None:
    import streamlit as st
    from starlette.middleware import Middleware
    app = st.App(str(APP_ROOT / "main.py"), middleware=[Middleware(OfflineHeaders, token=token, port=port)])
    app.run(config={
        "global.developmentMode": False,
        "server.address": "127.0.0.1", "server.port": port,
        "server.headless": True, "server.fileWatcherType": "none",
        "server.runOnSave": False, "browser.gatherUsageStats": False,
        "server.enableCORS": True, "server.enableXsrfProtection": True,
        "client.toolbarMode": "minimal", "logger.hideWelcomeMessage": True,
    })


def child_command(port: int, token: str) -> list[str]:
    command = [sys.executable]
    if not getattr(sys, "frozen", False):
        command.append(str(APP_ROOT / "desktop.py"))
    return [*command, "--serve", "--port", str(port), "--token", token]


def wait_for_server(process, url: str, token: str, timeout: float = 60) -> None:
    # Explicitly bypass system proxies: localhost must work without internet.
    opener = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server lokal berhenti saat startup. Periksa desktop.log.")
        try:
            with opener.open(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200 and response.headers.get("X-GTEx-Desktop") == token:
                    return
        except OSError:
            pass
        time.sleep(0.15)
    raise TimeoutError("Server lokal belum siap setelah 60 detik. Periksa desktop.log.")


def stop_server(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def check_local_server() -> None:
    """Verify the actual packaged server without opening a desktop window."""
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]
    token = secrets.token_hex(16)
    url = f"http://127.0.0.1:{port}"
    with (state_directory() / 'gtex_explorer.desktop_runtime.log').open("a", encoding="utf8") as log:
        process = subprocess.Popen(
            child_command(port, token), cwd=str(APP_ROOT), stdout=log, stderr=log,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            wait_for_server(process, url, token)
            with build_opener(ProxyHandler({})).open(url, timeout=5) as response:
                if response.status != 200 or b"<html" not in response.read().lower():
                    raise RuntimeError("Antarmuka lokal tidak tersedia.")
        finally:
            stop_server(process)


def launch_desktop() -> None:
    import webview
    check_resources()
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]
    token = secrets.token_hex(16)
    url = f"http://127.0.0.1:{port}"
    state = state_directory()
    with (state / 'gtex_explorer.desktop_runtime.log').open("a", encoding="utf8") as log:
        process = subprocess.Popen(
            child_command(port, token), cwd=str(APP_ROOT), stdout=log, stderr=log,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            wait_for_server(process, url, token)
            webview.settings["ALLOW_DOWNLOADS"] = True
            webview.settings["ALLOW_FILE_URLS"] = False
            webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
            webview.create_window(TITLE, url, width=1440, height=950,
                                  min_size=(1000, 700), text_select=True, zoomable=True)
            # Force a modern local engine. Never fall back to legacy IE rendering.
            webview.start(gui="edgechromium", private_mode=True,
                          storage_path=str(state / "webview"))
        finally:
            stop_server(process)


def check_training() -> dict:
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job, read_job, request_cancel
    from gtex_explorer.platform.model_registry import list_models, load_saved_model
    from gtex_explorer.inference.parsing import parse_expression_file
    from gtex_explorer.inference.audit import audit_and_align_input
    from gtex_explorer.inference.prediction import explain_prediction, predict_tissues
    import numpy as np
    check_state = state_directory() / 'training-checks' / secrets.token_hex(8)
    record = start_job(validate_training_config({'selector': 'anova', 'top_k': 100,
                       'xgboost': {'n_estimators': 2}}), APP_ROOT / 'training_data', check_state)
    deadline = time.monotonic() + 90
    while record.status in {'queued', 'running', 'cancelling'} and time.monotonic() < deadline:
        time.sleep(.2)
        record = read_job(record.run_id, check_state)
    if record.status != 'completed':
        request_cancel(record.run_id, check_state)
        raise RuntimeError(f'Training worker check failed: {record.error or record.status}')
    entry = next(x for x in list_models(APP_ROOT, check_state) if x.model_id == record.run_id)
    pre, classifier = load_saved_model(entry)
    with entry.demo_path.open('rb') as handle:
        matrix = parse_expression_file(handle, entry.demo_path.name)
    aligned, audit = audit_and_align_input(matrix, pre)
    if not audit.ready:
        raise ValueError(audit.message)
    predictions, probabilities = predict_tissues(aligned, pre, classifier)
    explain_prediction(aligned, str(aligned.index[0]), pre, classifier, boosting_rounds=2)
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Invalid training-check probabilities')
    return {'training_worker': 'ready', 'classes': len(pre['classes']), 'genes': pre['feature_count'],
            'boosting_rounds': 2, 'samples': len(predictions)}


def main() -> int:
    parser = argparse.ArgumentParser(description="GTEx Tissue Explorer offline desktop")
    parser.add_argument("--check", action="store_true", help="Check artifacts and demo inference without opening a window")
    parser.add_argument('--check-training', action='store_true', help='Run a short offline training-worker check')
    parser.add_argument('--training-worker', default='', help=argparse.SUPPRESS)
    parser.add_argument('--training-data', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--training-state', type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=8501, help=argparse.SUPPRESS)
    parser.add_argument("--token", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(APP_ROOT)
    # Streamlit, model loading, and demo inference must also work with external networking denied.
    enforce_offline_network()
    if args.training_worker:
        if args.training_data is None or args.training_state is None:
            parser.error('Training worker requires explicit data and state paths')
        from gtex_explorer.training.worker import run_job
        return run_job(args.training_worker, args.training_data, args.training_state, interruptible=True)
    if args.check_training:
        print(json.dumps(check_training()))
    elif args.serve:
        serve(args.port, args.token)
    elif args.check:
        report = check_resources()
        check_local_server()
        report["local_server"] = "ready"
        print(json.dumps(report, ensure_ascii=False))
    else:
        launch_desktop()
    return 0


