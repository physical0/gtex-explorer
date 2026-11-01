"""Workspaces, runs and models can be removed permanently, and only when it is safe."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from gtex_explorer.cloud.state import CloudState
from gtex_explorer.training.jobs import delete_run, job_path
from gtex_explorer.workspaces import create_workspace, delete_workspace, list_workspaces, remember_workspace, workspace_directory
from test_black_box_regressions import saved_run
from test_cloud_state import MemoryStore, write

APP = Path(__file__).resolve().parents[1] / 'app.py'
RUN = 'run_' + 'a' * 32


def launch(root, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(root))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_workspace_removal_deletes_files_and_registry_entry(tmp_path):
    keep = create_workspace(tmp_path, 'Keep')
    gone = create_workspace(tmp_path, 'Gone')
    directory = workspace_directory(tmp_path, gone['id'])
    (workspace_directory(tmp_path, keep['id']) / 'marker').write_text('kept')
    saved_run(directory, RUN)
    remember_workspace(tmp_path, gone['id'])
    delete_workspace(tmp_path, gone['id'])
    assert not directory.exists()
    assert [entry['id'] for entry in list_workspaces(tmp_path)] == ['default', keep['id']]
    assert json.loads((tmp_path / 'workspaces.json').read_text())['last_workspace_id'] == 'default'
    assert (workspace_directory(tmp_path, keep['id']) / 'marker').read_text() == 'kept'
    with pytest.raises(ValueError):
        delete_workspace(tmp_path, gone['id'])


def test_default_workspace_cannot_be_removed(tmp_path):
    with pytest.raises(ValueError, match='Default'):
        delete_workspace(tmp_path, 'default')


@pytest.mark.parametrize('status', ['queued', 'running', 'paused', 'cancelling'])
def test_workspace_with_active_training_is_kept(tmp_path, monkeypatch, status):
    entry = create_workspace(tmp_path, 'Busy')
    directory = workspace_directory(tmp_path, entry['id'])
    saved_run(directory, RUN, status=status)
    monkeypatch.setattr('gtex_explorer.training.jobs.read_job',
                        lambda run_id, root: type('Record', (), {'status': status})())
    with pytest.raises(ValueError, match='aktif'):
        delete_workspace(tmp_path, entry['id'])
    assert (directory / 'runs' / RUN / 'status.json').is_file()
    assert entry['id'] in {item['id'] for item in list_workspaces(tmp_path)}


def test_failed_rename_keeps_the_workspace_registered(tmp_path, monkeypatch):
    entry = create_workspace(tmp_path, 'Locked')
    directory = workspace_directory(tmp_path, entry['id'])

    def locked(path, root):
        raise PermissionError('in use')

    monkeypatch.setattr('gtex_explorer.training.data.discard_directory', locked)
    with pytest.raises(ValueError, match='digunakan'):
        delete_workspace(tmp_path, entry['id'])
    assert directory.is_dir()
    assert entry['id'] in {item['id'] for item in list_workspaces(tmp_path)}


def test_run_removal_deletes_run_and_model_but_keeps_dataset_and_other_runs(tmp_path):
    other = 'run_' + 'b' * 32
    saved_run(tmp_path, RUN, status='failed')
    saved_run(tmp_path, other)
    model = tmp_path / 'models' / RUN
    model.mkdir(parents=True)
    (model / 'xgboost_model.json').write_text('{}')
    dataset = tmp_path / 'datasets' / 'abc'
    dataset.mkdir(parents=True)
    (dataset / 'manifest.json').write_text('{}')
    delete_run(RUN, tmp_path)
    assert not job_path(RUN, tmp_path).exists() and not model.exists()
    assert job_path(other, tmp_path).is_dir() and (dataset / 'manifest.json').is_file()


def test_run_removal_refuses_active_training_and_unknown_ids(tmp_path, monkeypatch):
    saved_run(tmp_path, RUN, status='running')
    monkeypatch.setattr('gtex_explorer.training.jobs.read_job',
                        lambda run_id, root: type('Record', (), {'status': 'running'})())
    with pytest.raises(ValueError, match='berjalan'):
        delete_run(RUN, tmp_path)
    assert job_path(RUN, tmp_path).is_dir()
    monkeypatch.undo()
    with pytest.raises(ValueError):
        delete_run('../../outside', tmp_path)


def test_removal_reaches_other_instances_and_files_never_downloaded(tmp_path, monkeypatch):
    store = MemoryStore()
    prefix = 'workspaces/ws_a/datasets/abc/'
    writer = CloudState(tmp_path / 'writer', store)
    write(writer.root, prefix + 'manifest.json', b'{}')
    write(writer.root, prefix + 'raw_tpm.npy', b'matrix')
    write(writer.root, 'keep.json', b'{}')
    writer.publish()
    other = CloudState(tmp_path / 'other', store)
    other.pull()

    monkeypatch.setenv('GTEX_CLOUD', '1')
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'web')
    web = CloudState(tmp_path / 'web', store)
    web.pull()
    assert not (web.root / prefix / 'raw_tpm.npy').exists()
    from gtex_explorer.training.data import discard_directory
    discard_directory(web.root / 'workspaces/ws_a', web.root)
    web.forget(web.root / 'workspaces/ws_a')
    web.publish()
    assert not list(web.root.glob('.trash/*'))
    monkeypatch.delenv('GTEX_CLOUD')
    monkeypatch.delenv('GTEX_CLOUD_ROLE')

    # Nothing of the workspace stays durable, and hidden trash is never uploaded.
    files = json.loads(store.objects['gtex/state.json'][0])['files']
    assert list(files) == ['keep.json']
    other.pull()
    assert not (other.root / prefix / 'raw_tpm.npy').exists()
    assert (other.root / 'keep.json').is_file()


def test_launcher_requires_the_exact_name_and_offers_no_removal_for_default(tmp_path, monkeypatch):
    entry = create_workspace(tmp_path, 'Study A')
    app = launch(tmp_path, monkeypatch)
    assert not any(item.key == 'workspace_delete_default' for item in app.button)
    app.button(key='workspace_delete_' + entry['id']).click().run()
    assert not app.exception
    app.text_input(key='workspace_delete_name_' + entry['id']).set_value('study a')
    app.button(key='workspace_delete_confirm').click().run()
    assert not app.exception and any('tidak sama' in item.value for item in app.error)
    assert entry['id'] in {item['id'] for item in list_workspaces(tmp_path)}
    app.text_input(key='workspace_delete_name_' + entry['id']).set_value('Study A')
    app.button(key='workspace_delete_confirm').click().run()
    assert not app.exception
    assert entry['id'] not in {item['id'] for item in list_workspaces(tmp_path)}
    assert any('telah dihapus' in item.value for item in app.success)
    assert not any(item.key == 'workspace_delete_' + entry['id'] for item in app.button)


def test_session_inside_a_removed_workspace_returns_to_the_launcher(tmp_path, monkeypatch):
    entry = create_workspace(tmp_path, 'Study A')
    app = launch(tmp_path, monkeypatch)
    app.button(key='workspace_select_' + entry['id']).click().run()
    app.button(key='workspace_open').click().run()
    assert app.session_state.workspace_entered
    delete_workspace(tmp_path, entry['id'])
    app.run()
    assert not app.exception
    assert not app.session_state.workspace_entered
    assert any('telah dihapus' in item.value for item in app.success)
    app.button(key='workspace_open').click().run()
    assert not app.exception and app.session_state.active_workspace_id == 'default'


def test_results_page_removes_a_failed_run_only_after_confirmation(tmp_path, monkeypatch):
    from ui_helpers import enter_application
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    saved_run(tmp_path, RUN, status='failed', error={'type': 'ValueError', 'message': 'bad'})
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Hasil Pelatihan').click().run()
    app.button(key='training_delete_run').click().run()
    assert not app.exception
    app.button(key='training_delete_cancel').click().run()
    assert not app.exception and job_path(RUN, tmp_path).is_dir()
    app.button(key='training_delete_run').click().run()
    app.button(key='training_delete_confirm').click().run()
    assert not app.exception and not job_path(RUN, tmp_path).exists()
    assert not any(item.key == 'training_delete_run' for item in app.button)


def test_running_training_offers_no_removal(tmp_path, monkeypatch):
    from ui_helpers import enter_application
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    saved_run(tmp_path, RUN, status='queued')
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Hasil Pelatihan').click().run()
    assert not app.exception
    assert app.button(key='training_cancel').label == 'Stop'
    assert not any(item.key == 'training_delete_run' for item in app.button)
