"""Acceptance checks for the approved three-page training workflow."""
import json
import time
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from test_training_page import APP


def setup_app(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Pelatihan Model').click().run()
    return app


def test_explicit_steps_keep_data_and_configuration_on_back_and_next(tmp_path, monkeypatch):
    from test_training_uploads import uploads
    from gtex_explorer.training.uploads import audit_uploads
    app = setup_app(tmp_path, monkeypatch)
    assert any(t.label == '3. Tinjau dan latih' for t in app.tabs)
    assert app.button(key='training_to_configuration').disabled
    expression, metadata = uploads()
    app.session_state.training_upload_audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    app.run()
    app.button(key='training_to_configuration').click().run()
    assert app.session_state.training_step == '2. Konfigurasi'
    app.selectbox(key='training_selector').set_value('anova').run()
    app.number_input(key='training_top_k').set_value(2).run()
    app.number_input(key='training_xgb_n_estimators').set_value(7).run()
    app.button(key='training_to_review').click().run()
    assert app.session_state.training_step == '3. Tinjau dan latih'
    assert not app.button(key='training_start').disabled
    assert app.tabs[2].button(key='training_start').label == 'Latih dan simpan model'
    app.button(key='training_review_back').click().run()
    assert app.session_state.training_step == '2. Konfigurasi'
    assert app.number_input(key='training_top_k').value == 2
    assert app.number_input(key='training_xgb_n_estimators').value == 7
    app.button(key='training_back_to_data').click().run()
    assert app.session_state.training_step == '1. Data training'
    assert app.session_state.training_upload_audit.source['expression_file'] == 'own.csv'


def test_review_explains_blocking_configuration_beside_training_action(tmp_path, monkeypatch):
    app = setup_app(tmp_path, monkeypatch)
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.number_input(key='training_top_k').set_value(None).run()
    assert app.button(key='training_start').disabled
    assert any('jumlah gen' in error.value.lower() for error in app.tabs[2].error)


@pytest.mark.parametrize('status,key,label,icon', [
    ('running','training_pause','Pause',':material/pause:'),
    ('paused','training_unpause','Resume',':material/play_arrow:'),
])
def test_process_controls_keep_pause_resume_and_full_stop(tmp_path, monkeypatch, status, key, label, icon):
    from test_black_box_regressions import saved_run
    from gtex_explorer.training.jobs import atomic_json, process_lock
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    run_id='run_'+'a'*32
    saved_run(tmp_path,run_id,status=status)
    atomic_json(tmp_path/'runs'/run_id/'status.json', {'status':status,'stage':'final_xgboost',
        'rounds_completed':60,'rounds_total':300,'updated':time.time(),
        'paused_at':time.time(),'elapsed_seconds':12,'error':None})
    with process_lock(tmp_path/'worker.lock'):
        app=AppTest.from_file(str(APP),default_timeout=60).run()
        app = enter_application(app)
        app.button(key='nav_Hasil Pelatihan').click().run()
        assert not app.exception
        assert app.button(key=key).label == label
        assert app.button(key=key).proto.icon == icon
        assert app.button(key='training_cancel').label == 'Stop'
        assert app.button(key='training_cancel').proto.icon == ':material/stop:'
        assert any(h.value=='Status proses' for h in app.subheader)


@pytest.mark.parametrize('method,score_label', [
    ('anova','Skor F (ANOVA)'),('mutual_information','Skor MI (nat)'),
    ('l1','Koefisien absolut maksimum'),
])
def test_panel_preserves_tabs_diagnostics_scientific_values_and_fold_context(tmp_path, monkeypatch, method, score_label):
    from gtex_explorer.training.jobs import atomic_json
    from gtex_explorer.training.config import validate_training_config
    import gtex_explorer.ui.pages.training_genes as training_gene_page
    run_id='run_'+'c'*32
    root=tmp_path/'runs'/run_id
    atomic_json(root/'config.json',{'config':validate_training_config({'selector':method,'top_k':2,
        'evaluate_cv':True,'cv_folds':2}).to_dict(), 'dataset':{'sample_count':100,'class_count':13,
        'donor_count':100,'source':{'expression_file':'own.csv'}}})
    atomic_json(root/'status.json',{'status':'completed','updated':0,'cv_summary':{
        'mean_jaccard':.25,'jaccard_pairs':[{'fold_a':1,'fold_b':2,'jaccard':.25}]}})
    atomic_json(root/'fold_1.json',{})
    raw=pd.DataFrame({'gene':['GENE_A','GENE_B'],'p_value':[1.16e-46,.002],
        'selection_rank':[1,2],'selection_score':[1904.1204,.01],'nonzero_class_count':[2,1]})
    info={'selector':method,'candidate_count':5000,'requested_k':2,
          'training_samples':49,'validation_samples':51,'converged':True,
          'nonzero_gene_count':1000,'sparsity':.8,'iterations':30,'n_neighbors':3}
    monkeypatch.setattr(training_gene_page,'load_run_panel',lambda *args:(raw,info))
    # This is a display-format fixture. Publication is covered with real artifacts separately.
    monkeypatch.setattr(training_gene_page,'choose_run',lambda *args, **kwargs:run_id)
    app=AppTest.from_string('import streamlit as st\nfrom pathlib import Path\n'
        'from gtex_explorer.ui.pages.training_genes import render_training_gene_page\n'
        f'render_training_gene_page(Path({str(tmp_path)!r}))',default_timeout=60).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Daftar gen','Diagnostik metode','Stabilitas antar-fold']
    frames=[x.value for x in app.dataframe]
    genes=next(frame for frame in frames if 'ID gen' in frame.columns)
    assert score_label in genes.columns
    assert list(genes.columns)[:2]==['Peringkat','ID gen']
    assert genes.loc[0,'p-value'] == 1.16e-46
    table=next(item for item in app.dataframe if 'ID gen' in item.value.columns)
    assert json.loads(table.proto.columns)['p-value']['type_config']['format'] == '%.2e'
    assert raw.loc[0,'p_value'] == 1.16e-46  # Display must not mutate the export data.
    assert any('49' in m.value and 'training' in m.value for m in app.markdown)
    assert any('51' in m.value and 'validation' in m.value for m in app.markdown)
    pairs=next(frame for frame in frames if 'Jaccard' in frame.columns)
    assert pairs.loc[0,'Jaccard']=='25,00%'  # Same decimal comma as the results page.
    assert any(m.value == '25,00%' for m in app.metric)
    assert app.button(key='training_panel_back').label=='Kembali ke hasil'
    assert any('Solver konvergen' in str(frame) for frame in frames)


def test_upload_change_clears_the_stored_audit_before_files_can_be_removed():
    app=AppTest.from_string("\nimport streamlit as st\nimport gtex_explorer.ui.pages.training as training_page\nif not st.session_state.get('initialized'):\n    st.session_state.initialized=True\n    st.session_state.training_upload_audit={'old':'dataset'}\n    st.session_state.training_upload_signature='old-file-signature'\n    st.session_state.training_data_ready=True\nst.button('Change uploaded file', key='change_file',\n          on_click=getattr(training_page,'_invalidate_training_upload',lambda:None))\n",default_timeout=60).run()
    app.button(key='change_file').click().run()
    assert 'training_upload_audit' not in app.session_state
    assert 'training_upload_signature' not in app.session_state
    assert app.session_state.training_data_ready is False


def test_deterministic_failures_do_not_offer_checkpoint_resume():
    from gtex_explorer.training.jobs import JobRecord, can_resume
    def record(status, error=None):
        return JobRecord('run_x', status, error, {})
    assert can_resume(record('cancelled'))
    assert can_resume(record('failed', {'type': 'WorkerStopped'}))
    assert not can_resume(record('failed', {'type': 'SelectorEligibilityError'}))
    assert not can_resume(record('completed'))


def test_training_form_values_survive_visiting_another_page(tmp_path, monkeypatch):
    from ui_helpers import enter_application
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run(), 'Pelatihan Model')
    app.number_input(key='training_xgb_n_estimators').set_value(500).run()
    app.button(key='nav_Kamus Biologi').click().run()
    app.button(key='nav_Pelatihan Model').click().run()
    assert not app.exception
    assert app.number_input(key='training_xgb_n_estimators').value == 500
