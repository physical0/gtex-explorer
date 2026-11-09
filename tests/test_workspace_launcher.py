"""The workspace chooser is a separate entry stage before the application."""
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / 'main.py'


def launch(root, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(root))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def open_selected(app):
    app.button(key='workspace_open').click().run()
    assert not app.exception
    return app


def test_startup_excludes_sidebar_and_resource_loading(tmp_path, monkeypatch):
    with patch('gtex_explorer.experiments.reference_tables.load_reference_tables', side_effect=AssertionError('wait for entry')):
        app = launch(tmp_path, monkeypatch)
    assert not app.exception
    assert any(item.value == 'Pilih workspace' for item in app.title)
    assert not app.sidebar.button and not app.sidebar.selectbox and not app.file_uploader
    assert app.button(key='workspace_open')
    assert 'workspace_entered' not in app.session_state or not app.session_state.workspace_entered


def test_selecting_requires_open_and_switching_clears_previous_uploads(tmp_path, monkeypatch):
    from gtex_explorer.platform.workspaces import create_workspace
    other = create_workspace(tmp_path, 'Other', include_bundled_models=False)
    app = open_selected(launch(tmp_path, monkeypatch))
    assert app.title[0].value == 'Latih model'
    before = {upload.key for upload in app.file_uploader}
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.session_state.training_upload_audit = 'clear only after open'
    app.button(key='workspace_change').click().run()
    app.button(key='workspace_select_' + other['id']).click().run()
    assert app.session_state.active_workspace_id == 'default'
    assert app.session_state.training_upload_audit == 'clear only after open'
    open_selected(app)
    assert app.session_state.active_workspace_id == other['id']
    assert 'training_upload_audit' not in app.session_state
    assert app.radio(key='training_source').value == 'Data pengguna'
    assert before.isdisjoint({upload.key for upload in app.file_uploader})
    app.button(key='nav_Prediksi Jaringan').click().run()
    assert any('belum memiliki model' in item.value for item in app.info)


def test_create_selects_without_entering_and_preserves_bundle_policy(tmp_path, monkeypatch):
    from gtex_explorer.platform.workspaces import list_workspaces
    app = launch(tmp_path, monkeypatch)
    app.button(key='workspace_new').click().run()
    app.text_input(key='workspace_new_name').set_value('My own models')
    app.checkbox(key='workspace_include_originals').uncheck()
    app.button(key='workspace_create').click().run()
    assert not app.exception
    selected = app.session_state.workspace_selection
    assert selected != 'default'
    assert not app.session_state.workspace_entered
    assert next(w for w in list_workspaces(tmp_path) if w['id'] == selected)['include_bundled_models'] is False
    open_selected(app)
    app.button(key='nav_Prediksi Jaringan').click().run()
    assert not any(box.key == 'prediction_model' for box in app.selectbox)


def test_many_names_search_and_hidden_selection_keep_open_action(tmp_path, monkeypatch):
    from gtex_explorer.platform.workspaces import create_workspace
    name = ('Analisis ' + 'jaringan ' * 7).strip()
    other = create_workspace(tmp_path, name, include_bundled_models=False)
    for number in range(12):
        create_workspace(tmp_path, f'Project {number:02}')
    app = launch(tmp_path, monkeypatch)
    assert len([b for b in app.button if b.key.startswith('workspace_select_')]) == 14
    app.text_input(key='workspace_search').set_value('ANALISIS').run()
    rows = [b for b in app.button if b.key.startswith('workspace_select_')]
    assert len(rows) == 1 and rows[0].proto.help == name
    rows[0].click().run()
    app.text_input(key='workspace_search').set_value('no match exists').run()
    assert not app.exception
    assert any('Tidak ada workspace' in item.value for item in app.info)
    assert not app.button(key='workspace_open').disabled
    assert app.session_state.workspace_selection == other['id']
    assert any(name in item.value for item in app.markdown)
    open_selected(app)
    assert app.session_state.active_workspace_id == other['id']


def test_rename_under_search_preserves_selection_id_and_policy(tmp_path, monkeypatch):
    from gtex_explorer.platform.workspaces import create_workspace, list_workspaces
    other = create_workspace(tmp_path, 'Before', include_bundled_models=False)
    app = launch(tmp_path, monkeypatch)
    app.text_input(key='workspace_search').set_value('before').run()
    app.button(key='workspace_rename_' + other['id']).click().run()
    app.text_input(key='workspace_rename_name_' + other['id']).set_value('After')
    app.button(key='workspace_save_name').click().run()
    assert not app.exception
    current = next(w for w in list_workspaces(tmp_path) if w['id'] == other['id'])
    assert current['name'] == 'After' and current['include_bundled_models'] is False
    assert app.session_state.workspace_selection == other['id']
    assert not app.session_state.workspace_entered
    assert any('After' in item.value for item in app.markdown)


def test_new_session_preselects_last_workspace_but_waits_for_open(tmp_path, monkeypatch):
    from gtex_explorer.platform.workspaces import create_workspace, remember_workspace
    other = create_workspace(tmp_path, 'Last used')
    remember_workspace(tmp_path, other['id'])
    app = launch(tmp_path, monkeypatch)
    assert app.session_state.workspace_selection == other['id']
    assert not app.sidebar.button
    open_selected(app)
    app.button(key='workspace_change').click().run()
    assert not app.sidebar.button
    open_selected(app)
    assert app.session_state.active_workspace_id == other['id']


def test_compact_switch_action_returns_to_chooser_without_changing_project(tmp_path, monkeypatch):
    app = open_selected(launch(tmp_path, monkeypatch))
    active_id = app.session_state.active_workspace_id
    change = app.button(key='workspace_change')
    assert change.label == 'Ganti'
    assert change.proto.help == 'Kembali ke pemilihan workspace'
    change.click().run()
    assert not app.exception
    assert not app.sidebar.button
    assert app.title[0].value == 'Pilih workspace'
    assert app.session_state.active_workspace_id == active_id
