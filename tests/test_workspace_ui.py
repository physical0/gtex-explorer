"""Cross-workspace active process navigation from the launcher."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_open_other_workspace_process_requests_full_app_rerun(tmp_path, monkeypatch):
    import streamlit as st
    from gtex_explorer.workspaces import create_workspace, workspace_directory
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    other = create_workspace(tmp_path / 'state', 'Other', include_bundled_models=True)
    data = tmp_path / 'data'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args:None)
    root = workspace_directory(tmp_path / 'state', other['id'])
    job = start_job(validate_training_config({'selector':'anova','top_k':2}, candidate_count=6), data, root)
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path / 'state'))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.sidebar.button
    actual_rerun, scopes = st.rerun, []
    def observed_rerun(*args, **kwargs):
        scopes.append(kwargs.get('scope'))
        return actual_rerun(*args, **kwargs)
    monkeypatch.setattr(st, 'rerun', observed_rerun)
    app.button(key='workspace_active_process').click().run()
    assert not app.exception
    assert 'app' in scopes  # A real fragment-scoped click must refresh app routing too.
    assert app.session_state.workspace_entered is True
    assert app.session_state.active_workspace_id == other['id']
    assert app.selectbox(key='training_run').value == job.run_id
    assert app.session_state.training_results_view == 'Status proses'
