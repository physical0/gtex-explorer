from pathlib import Path
import asyncio
import socket
import sys
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gtex_explorer.platform.desktop_runtime as desktop


@pytest.mark.parametrize('host,expected', [
    ('127.0.0.1', True), ('::1', True), ('localhost', True),
    ('192.168.1.10', False), ('8.8.8.8', False), ('example.com', False),
    ('127.0.0.1.example.com', False),
])
def test_loopback_only(host, expected):
    assert desktop.is_loopback(host) is expected


def test_backend_blocks_external_network(monkeypatch):
    # Register originals so pytest restores them after this test.
    for owner, name in [(socket, 'getaddrinfo'), (socket.socket, 'connect'), (socket.socket, 'connect_ex')]:
        monkeypatch.setattr(owner, name, getattr(owner, name))
    desktop.enforce_offline_network()
    with pytest.raises(OSError, match='external DNS'):
        socket.getaddrinfo('example.com', 443)
    with socket.socket() as sock:
        for operation in [sock.connect, sock.connect_ex]:
            with pytest.raises(OSError, match='loopback'):
                operation(('8.8.8.8', 443))
    assert socket.getaddrinfo('127.0.0.1', 80)


def test_packaged_and_source_child_commands(monkeypatch):
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    assert desktop.child_command(12345, 'token') == [sys.executable, '--serve', '--port', '12345', '--token', 'token']
    monkeypatch.setattr(sys, 'frozen', False)
    assert desktop.child_command(12345, 'token')[1].endswith('desktop.py')


def test_server_shutdown_does_not_stop_other_processes():
    process = Mock()
    process.poll.return_value = None
    desktop.stop_server(process)
    process.terminate.assert_called_once()
    process.wait.assert_called_once_with(timeout=5)
    process.kill.assert_not_called()
    process.reset_mock()
    process.poll.return_value = 0
    desktop.stop_server(process)
    process.terminate.assert_not_called()


def test_readiness_detects_failed_child():
    process = Mock()
    process.poll.return_value = 1
    with pytest.raises(RuntimeError, match='startup'):
        desktop.wait_for_server(process, 'http://127.0.0.1:12345', 'token')


def test_offline_headers_allow_local_assets_and_identify_server():
    messages = []
    async def app(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 200, 'headers': []})
    async def send(message):
        messages.append(message)
    asyncio.run(desktop.OfflineHeaders(app, 'token', 12345)({'type': 'http'}, None, send))
    headers = dict(messages[0]['headers'])
    policy = headers[b'content-security-policy'].decode()
    assert 'ws://127.0.0.1:12345' in policy
    assert 'https:' not in policy
    assert "font-src 'self' data:" in policy
    assert headers[b'x-gtex-desktop'] == b'token'


def test_resources_and_real_demo_inference():
    report = desktop.check_resources()
    assert report['status'] == 'ready'
    assert report['classes'] == 13
    assert report['genes'] == 1000
    assert report['demo_samples'] > 0
