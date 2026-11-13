"""Project storage, model availability and process coordination boundaries."""
import json
from pathlib import Path

import pytest


def test_default_keeps_legacy_files_in_place_and_new_workspaces_are_independent(tmp_path):
    from gtex_explorer.platform.workspaces import list_workspaces, create_workspace, workspace_directory, rename_workspace
    legacy = tmp_path / 'models' / 'legacy.json'
    legacy.parent.mkdir()
    legacy.write_text('existing')
    assert list_workspaces(tmp_path)[0]['id'] == 'default'
    assert workspace_directory(tmp_path, 'default') == tmp_path.resolve()
    a = create_workspace(tmp_path, 'Study A', include_bundled_models=True)
    b = create_workspace(tmp_path, 'Study B', include_bundled_models=False)
    ar, br = workspace_directory(tmp_path, a['id']), workspace_directory(tmp_path, b['id'])
    assert ar != br and ar.parent == br.parent == tmp_path / 'workspaces'
    (ar / 'models').mkdir()
    (ar / 'models' / 'mine.json').write_text('A only')
    assert not (br / 'models' / 'mine.json').exists()
    rename_workspace(tmp_path, a['id'], 'Renamed study')
    assert workspace_directory(tmp_path, a['id']) == ar
    assert (ar / 'models' / 'mine.json').read_text() == 'A only'
    assert legacy.read_text() == 'existing'


@pytest.mark.parametrize('name', ['', '   ', 'x'*81, 'bad\nname'])
def test_invalid_names_do_not_create_workspace_storage(tmp_path, name):
    from gtex_explorer.platform.workspaces import create_workspace
    with pytest.raises(ValueError):
        create_workspace(tmp_path, name, include_bundled_models=True)
    assert not list((tmp_path / 'workspaces').glob('*'))


def test_unknown_workspace_and_duplicate_name_are_rejected(tmp_path):
    from gtex_explorer.platform.workspaces import create_workspace, workspace_directory
    create_workspace(tmp_path, 'Study A', include_bundled_models=True)
    with pytest.raises(ValueError):
        create_workspace(tmp_path, 'study a', include_bundled_models=True)
    with pytest.raises(ValueError):
        workspace_directory(tmp_path, '../outside')


def test_bundled_model_choice_is_per_workspace_and_does_not_hide_user_models(tmp_path):
    from gtex_explorer.platform.workspaces import create_workspace, workspace_directory
    from gtex_explorer.platform.model_registry import list_models, publish_model
    from test_model_registry import trained_fixture
    resources = Path(__file__).resolve().parents[1]
    with_original = create_workspace(tmp_path, 'With originals', include_bundled_models=True)
    empty = create_workspace(tmp_path, 'Without originals', include_bundled_models=False)
    ar, br = workspace_directory(tmp_path, with_original['id']), workspace_directory(tmp_path, empty['id'])
    assert {'original_l1_c_1_1000', 'original_baseline_5000'} <= {m.model_id for m in list_models(resources, ar)}
    assert list_models(resources, br) == []
    result, config, demo = trained_fixture()
    publish_model(result, config, {'classes':['A','B','C']}, br / 'models' / 'my_model', demo, None)
    assert [m.model_id for m in list_models(resources, br)] == ['my_model']
    assert 'my_model' not in {m.model_id for m in list_models(resources, ar)}


def test_switch_clears_analysis_forms_and_uses_new_uploader_identity(tmp_path):
    from gtex_explorer.platform.workspaces import create_workspace
    from gtex_explorer.ui.workspace import activate_workspace, upload_widget_key
    a = create_workspace(tmp_path, 'A', include_bundled_models=True)
    b = create_workspace(tmp_path, 'B', include_bundled_models=False)
    state = {'page':'Prediksi Jaringan', 'active_workspace_id':a['id'],
             'training_upload_audit':object(), 'training_source':'Data eksperimen notebook',
             'training_selector':'baseline', 'prediction_model':'model_from_a',
             'input_audit':object(), 'probabilities':object(), 'active_model_id':'model_from_a'}
    before = upload_widget_key('upload', state)
    activate_workspace(state, tmp_path, b['id'])
    assert state['active_workspace_id'] == b['id']
    for key in ['training_upload_audit','training_source','training_selector','prediction_model',
                'input_audit','probabilities','active_model_id']:
        assert key not in state
    assert state['page'] == 'Prediksi Jaringan'
    assert upload_widget_key('upload', state) != before
    activate_workspace(state, tmp_path, a['id'])
    assert upload_widget_key('upload', state) != before


def test_training_reservation_blocks_another_workspace_before_worker_start(tmp_path, monkeypatch):
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job, read_job, atomic_json
    from gtex_explorer.platform.workspaces import create_workspace, workspace_directory
    a = create_workspace(tmp_path / 'state', 'A', include_bundled_models=True)
    b = create_workspace(tmp_path / 'state', 'B', include_bundled_models=False)
    ar = workspace_directory(tmp_path / 'state', a['id'])
    br = workspace_directory(tmp_path / 'state', b['id'])
    data = tmp_path / 'data'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args:None)
    config = validate_training_config({'selector':'anova','top_k':2}, candidate_count=6)
    first = start_job(config, data, ar)
    with pytest.raises(ValueError, match='starting|running'):
        start_job(config, data, br)
    assert read_job(first.run_id, ar).status == 'queued'
    atomic_json(ar / 'runs' / first.run_id / 'status.json', {'status':'cancelled','updated':0,'error':None})
    assert start_job(config, data, br).run_id != first.run_id


def test_uploads_with_identical_contents_still_have_separate_files(tmp_path):
    from gtex_explorer.platform.workspaces import create_workspace, workspace_directory
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    from test_training_uploads import uploads
    a = create_workspace(tmp_path, 'A', include_bundled_models=True)
    b = create_workspace(tmp_path, 'B', include_bundled_models=True)
    expression, metadata = uploads()
    audit = audit_uploads(expression, 'mine.csv', metadata, 'labels.csv')
    ar = save_training_upload(audit, workspace_directory(tmp_path, a['id']) / 'datasets', 3)
    br = save_training_upload(audit, workspace_directory(tmp_path, b['id']) / 'datasets', 3)
    assert ar != br and ar.name == br.name
    (ar / 'a-only.txt').write_text('A')
    assert not (br / 'a-only.txt').exists()


def test_unrelated_live_worker_does_not_keep_dead_run_alive(tmp_path):
    import time
    from gtex_explorer.platform.workspaces import create_workspace, workspace_directory
    from gtex_explorer.training.jobs import atomic_json, process_lock, read_job
    a = create_workspace(tmp_path, 'A', include_bundled_models=True)
    b = create_workspace(tmp_path, 'B', include_bundled_models=True)
    ar, br = workspace_directory(tmp_path, a['id']), workspace_directory(tmp_path, b['id'])
    old_id, live_id = 'run_' + 'a'*32, 'run_' + 'b'*32
    for root, identity in [(ar, old_id),(br, live_id)]:
        atomic_json(root / 'runs' / identity / 'status.json',
                    {'status':'running','updated':time.time()-60,'error':None})
    atomic_json(tmp_path / 'worker_owner.json', {'state_root':str(br),'run_id':live_id})
    with process_lock(tmp_path / 'worker.lock'):
        assert read_job(old_id, ar).status == 'failed'
        assert read_job(live_id, br).status == 'running'


def test_invalid_launcher_target_cannot_change_active_workspace(tmp_path, monkeypatch):
    import gtex_explorer.ui.workspace as workspace_ui
    import pytest
    from gtex_explorer.platform.workspaces import create_workspace
    entry = create_workspace(tmp_path, 'Before', include_bundled_models=False)
    class State(dict):
        __getattr__ = dict.__getitem__
    state = State(active_workspace_id=entry['id'], workspace_entered=False,
                  training_selected_run='keep_this_run')
    monkeypatch.setattr(workspace_ui.st, 'session_state', state)
    with pytest.raises(ValueError, match='Workspace tidak ditemukan'):
        workspace_ui._enter_workspace(tmp_path, 'invalid')
    assert state['active_workspace_id'] == entry['id']
    assert state['workspace_entered'] is False
    assert state['training_selected_run'] == 'keep_this_run'


def hold_registry(root, seconds):
    """Another session holding the workspace list's lock for `seconds`; returns its thread once it holds it."""
    import threading
    import time
    from gtex_explorer.platform.workspaces import REGISTRY_LOCK
    from gtex_explorer.training.jobs import process_lock
    held = threading.Event()

    def hold():
        with process_lock(Path(root) / REGISTRY_LOCK):
            held.set()
            time.sleep(seconds)

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait()
    return thread


def test_opening_a_workspace_waits_for_another_session_writing_the_list(tmp_path):
    # Two sessions writing the list at once used to fail at once with PermissionError.
    from gtex_explorer.platform.workspaces import create_workspace, last_workspace, remember_workspace
    entry = create_workspace(tmp_path, 'Study A')
    other = hold_registry(tmp_path, 0.4)
    remember_workspace(tmp_path, entry['id'])
    other.join()
    assert last_workspace(tmp_path) == entry['id']


def test_a_list_kept_busy_past_the_wait_still_opens_the_workspace_but_blocks_changes(tmp_path, monkeypatch):
    import gtex_explorer.platform.workspaces as workspaces
    entry = workspaces.create_workspace(tmp_path, 'Study A')
    monkeypatch.setattr(workspaces, 'REGISTRY_WAIT', 0.2)
    other = hold_registry(tmp_path, 1.0)
    try:
        # Saving "last used" is skipped rather than failing the workspace that is being opened.
        workspaces.remember_workspace(tmp_path, entry['id'])
        assert workspaces.last_workspace(tmp_path) == 'default'
        with pytest.raises(ValueError, match='sedang digunakan proses lain'):
            workspaces.create_workspace(tmp_path, 'Study B')
        with pytest.raises(ValueError, match='sedang digunakan proses lain'):
            workspaces.rename_workspace(tmp_path, entry['id'], 'Study C')
    finally:
        other.join()
    assert [item['name'] for item in workspaces.list_workspaces(tmp_path)] == ['Default', 'Study A']


def test_training_locks_still_give_up_at_once(tmp_path):
    import time
    from gtex_explorer.training.jobs import LockBusy, process_lock
    other = hold_registry(tmp_path, 0.5)  # any lock file behaves the same
    from gtex_explorer.platform.workspaces import REGISTRY_LOCK
    started = time.monotonic()
    with pytest.raises(LockBusy):
        with process_lock(tmp_path / REGISTRY_LOCK):
            pass
    assert time.monotonic() - started < 0.2
    other.join()
