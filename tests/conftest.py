"""Keep all app tests away from the user's actual workspace registry and runs."""
import pytest


@pytest.fixture(autouse=True)
def isolated_application_state(tmp_path, monkeypatch):
    # A test may override this again with its own explicitly prepared state.
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path / 'application-state'))
