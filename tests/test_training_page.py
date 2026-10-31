from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

APP = Path(__file__).resolve().parents[1] / 'app.py'


def test_training_controls_are_method_specific_and_cv_is_optional(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Pelatihan Model').click().run()
    assert not app.exception
    assert app.radio(key='training_source').value == 'Data pengguna'
    assert app.button(key='training_start').disabled
    assert any(item.key == 'nav_Hasil Pelatihan' for item in app.button)
    assert any(item.key == 'nav_Panel Gen Pelatihan' for item in app.button)
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    assert not app.exception
    assert app.checkbox(key='training_cv').value is False
    assert app.selectbox(key='training_folds').disabled
    app.selectbox(key='training_selector').set_value('baseline').run()
    assert app.selectbox(key='training_preset').value == 'Custom'
    assert app.number_input(key='training_top_k').disabled
    assert app.number_input(key='training_top_k').value == 5000
    app.selectbox(key='training_selector').set_value('mutual_information').run()
    assert app.number_input(key='training_mi_neighbors').value == 3
    assert not any(item.key == 'training_l1_c' for item in app.number_input)
    app.checkbox(key='training_cv').check().run()
    assert not app.selectbox(key='training_folds').disabled
    app.number_input(key='training_top_k').set_value(None).run()
    assert not app.exception
    assert app.button(key='training_start').disabled


def test_training_workspace_results_and_panels_do_not_show_notebook_panels(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Hasil Pelatihan').click().run()
    assert not app.exception
    assert any('Belum ada' in item.value for item in app.info)
    app.button(key='nav_Panel Gen Pelatihan').click().run()
    assert not app.exception
    assert not app.dataframe


def test_blank_candidate_count_blocks_training_with_recovery_message(tmp_path, monkeypatch):
    from test_training_uploads import uploads
    from gtex_explorer.training.uploads import audit_uploads
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    expression, metadata = uploads()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.session_state.training_upload_audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    app.button(key='nav_Pelatihan Model').click().run()
    assert not app.exception
    app.number_input(key='training_candidates').set_value(None).run()
    assert not app.exception
    assert app.button(key='training_start').disabled
    assert any('Isi jumlah gen kandidat' in item.value for item in app.error)


def test_model_swap_discards_audit_predictions_and_rounds(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='load_local_demo').click().run()
    app.button(key='run_prediction').click().run()
    assert 'predictions' in app.session_state
    shared_input = app.session_state['input_matrix_before_imputation'].copy()
    sample = app.session_state['active_sample']
    app.selectbox(key='prediction_model').set_value('original_baseline_5000').run()
    assert not app.exception
    assert 'predictions' not in app.session_state
    pd.testing.assert_frame_equal(app.session_state['input_matrix_before_imputation'], shared_input)
    assert app.session_state['active_sample'] == sample
    assert app.session_state['input_filename'] == 'demo_tpm_shared.csv'
    assert app.session_state['input_audit'].matched_gene_count == 5000
    app.button(key='run_prediction').click().run()
    assert app.session_state['probabilities'].shape == (len(shared_input), 13)


def test_cancelled_run_shows_requested_rounds_without_claiming_completion(tmp_path, monkeypatch):
    from test_black_box_regressions import saved_run
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    saved_run(tmp_path, 'run_'+'d'*32, status='cancelled')
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Hasil Pelatihan').click().run()
    assert not app.exception
    assert any('Target XGBoost: 300 putaran per fit' in c.value for c in app.caption)
    assert any('Pelatihan dihentikan' in item.value for item in app.info)
    assert not any(item.key == 'training_cancel' for item in app.button)


def test_stop_has_immediate_feedback_and_displays_actual_round_progress(tmp_path, monkeypatch):
    import time
    from test_black_box_regressions import saved_run
    from gtex_explorer.training.jobs import atomic_json, process_lock
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    run_id='run_'+'e'*32
    saved_run(tmp_path,run_id,status='running')
    atomic_json(tmp_path/'runs'/run_id/'status.json', {'status':'running','stage':'final_xgboost',
                'rounds_completed':60,'rounds_total':300,'updated':time.time(),'error':None})
    with process_lock(tmp_path/'worker.lock'):
        app = AppTest.from_file(str(APP),default_timeout=60).run()
        app = enter_application(app)
        app.button(key='nav_Hasil Pelatihan').click().run()
        assert not app.exception
        assert app.get('progress')[0].proto.value == 20
        assert app.button(key='training_cancel').label == 'Stop'
        app.button(key='training_cancel').click().run()
        assert not app.exception
        assert app.button(key='training_cancel').disabled
        assert any('Permintaan berhenti diterima' in item.value for item in app.info)


def test_paused_training_has_live_resume_and_separate_end_button(tmp_path, monkeypatch):
    import time
    from test_black_box_regressions import saved_run
    from gtex_explorer.training.jobs import atomic_json, process_lock
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR',str(tmp_path))
    run_id='run_'+'b'*32
    saved_run(tmp_path,run_id,status='paused')
    atomic_json(tmp_path/'runs'/run_id/'status.json', {'status':'paused','stage':'final_xgboost',
                'rounds_completed':60,'rounds_total':300,'updated':time.time(),
                'paused_at':time.time(),'elapsed_seconds':12,'error':None})
    with process_lock(tmp_path/'worker.lock'):
        app=AppTest.from_file(str(APP),default_timeout=60).run()
        app = enter_application(app)
        app.button(key='nav_Hasil Pelatihan').click().run()
        assert not app.exception
        assert app.button(key='training_unpause').label == 'Resume'
        assert app.button(key='training_cancel').label == 'Stop'
        assert not any(item.key=='training_resume' for item in app.button)
        assert any('memori' in item.value for item in app.info)
        assert any('12 detik' in item.value for item in app.caption)


def test_run_time_uses_saved_start_not_restored_file_time(tmp_path):
    # Hosted instances rewrite files on restore, so file times are not the run's age.
    import json, os, time
    from gtex_explorer.ui.training_page import run_created
    run = tmp_path / 'run_x'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'created': 1_000_000.0}), encoding='utf-8')
    os.utime(run / 'config.json', (time.time(), time.time()))
    assert run_created(run) == 1_000_000.0
    (run / 'config.json').write_text(json.dumps({}), encoding='utf-8')
    assert abs(run_created(run) - (run / 'config.json').stat().st_mtime) < 1e-6
