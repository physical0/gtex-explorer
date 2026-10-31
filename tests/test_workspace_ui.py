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


def test_new_session_restores_workspace_page_and_model_from_url(tmp_path, monkeypatch):
    # A hosted reconnect starts a fresh session; the URL carries the user's place.
    from gtex_explorer.workspaces import create_workspace
    other = create_workspace(tmp_path / 'state', 'Other', include_bundled_models=True)
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path / 'state'))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.query_params.update({'ws': other['id'], 'page': 'Kamus Biologi', 'model': 'baseline'})
    app.run()
    assert not app.exception
    assert app.session_state.workspace_entered is True
    assert app.session_state.active_workspace_id == other['id']
    assert app.session_state.page == 'Kamus Biologi'
    assert app.session_state.active_model_id == 'baseline'


def test_url_follows_navigation_and_launcher_clears_it(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path / 'state'))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app.button(key='workspace_open').click().run()
    assert app.query_params['ws'] == ['default'] and app.query_params['page'] == ['Pelatihan Model']
    app.button(key='nav_Kamus Biologi').click().run()
    assert app.query_params['page'] == ['Kamus Biologi']
    app.button(key='workspace_change').click().run()
    assert not app.exception and not dict(app.query_params)
    app.run()
    assert app.session_state.workspace_entered is False  # Leaving is not undone by a rerun.


def test_unknown_workspace_in_url_shows_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path / 'state'))
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.query_params.update({'ws': 'ws_missing', 'page': 'Kamus Biologi'})
    app.run()
    assert not app.exception
    assert app.session_state.workspace_entered is False
    assert app.button(key='workspace_open')
