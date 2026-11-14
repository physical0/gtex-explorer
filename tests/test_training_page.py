from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

APP = Path(__file__).resolve().parents[1] / 'main.py'


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
    assert app.slider(key='training_folds').disabled and app.slider(key='training_folds').value == 5
    assert not any(item.key == 'training_preset' for item in app.selectbox)
    app.selectbox(key='training_selector').set_value('baseline').run()
    assert app.number_input(key='training_top_k').disabled
    assert app.number_input(key='training_top_k').value == 5000
    app.selectbox(key='training_selector').set_value('mutual_information').run()
    assert app.number_input(key='training_mi_neighbors').value == 3
    assert not any(item.key == 'training_l1_c' for item in app.number_input)
    assert app.slider(key='training_holdout').value == 20 and not app.slider(key='training_holdout').disabled
    captions = lambda: ' '.join(item.value for item in app.caption)
    assert 'Data development:' in captions() and 'Data holdout eksperimen notebook tetap tidak dipakai' in captions()
    app.slider(key='training_holdout').set_value(0).run()
    assert not app.exception
    assert 'Dengan 0%' in captions() and 'Data development:' not in captions()
    app.slider(key='training_holdout').set_value(20).run()
    app.checkbox(key='training_cv').check().run()
    assert not app.slider(key='training_holdout').disabled and not app.slider(key='training_folds').disabled
    assert '645 sampel data development dibagi menjadi 5 fold' in captions() and 'Data holdout tidak ikut' in captions()
    app.slider(key='training_folds').set_value(4).run()
    assert not app.exception and '(75%) untuk pelatihan' in captions() and '(25%) untuk validasi' in captions()
    app.slider(key='training_holdout').set_value(0).run()
    assert not app.exception and '807 sampel data development dibagi menjadi 4 fold' in captions()
    # A fold that would miss a tissue is refused instead of training on it.
    app.slider(key='training_folds').set_value(10).run()
    assert not app.exception and any('Pilih jumlah fold lain' in item.value for item in app.error)
    app.slider(key='training_holdout').set_value(20).run()
    assert any('Pilih jumlah fold lain' in item.value for item in app.error) and 'Data development: 645 sampel' in captions()
    assert app.button(key='training_start').disabled
    app.slider(key='training_holdout').set_value(20).run()
    app.slider(key='training_folds').set_value(5).run()
    headings = [item.value for item in app.subheader]
    assert headings.index('Pembagian data utama') < headings.index('Seleksi fitur') < headings.index('Evaluasi tambahan')
    assert not app.slider(key='training_folds').disabled
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


def test_default_gene_count_follows_the_data_until_the_user_types_one(tmp_path, monkeypatch):
    from test_training_uploads import uploads
    from gtex_explorer.training.uploads import audit_uploads
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    expression, metadata = uploads()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.session_state.training_upload_audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    app.button(key='nav_Pelatihan Model').click().run()
    method = lambda: app.selectbox(key='training_selector')
    top_k = lambda: app.number_input(key='training_top_k').value
    # The form starts on L1 with C = 1, as the removed preset did.
    assert method().value == 'l1' and app.number_input(key='training_l1_c').value == 1.0
    # A small upload: every gene is used, by default and for the baseline.
    assert top_k() == 3
    method().set_value('baseline').run()
    assert not app.exception and top_k() == 3
    method().set_value('l1').run()
    assert top_k() == 3
    # Larger data: the default refits instead of keeping the small number.
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    assert not app.exception and top_k() == 1000
    method().set_value('baseline').run()
    assert top_k() == 5000
    method().set_value('anova').run()
    assert top_k() == 1000
    # A number the user types is theirs to keep, and is only lowered when the data has fewer genes.
    app.number_input(key='training_top_k').set_value(250).run()
    method().set_value('mutual_information').run()
    assert top_k() == 250
    app.radio(key='training_source').set_value('Data pengguna').run()
    assert top_k() == 3
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    assert top_k() == 3


def test_missing_notebook_data_after_an_older_upload_shows_only_the_message(tmp_path, monkeypatch):
    from test_training_uploads import uploads
    from gtex_explorer.training.uploads import audit_uploads
    import gtex_explorer.ui.pages.training as training_page
    def missing(_):
        raise FileNotFoundError('training_data')
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr(training_page, 'cached_data', missing)
    expression, metadata = uploads()
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.session_state.training_upload_audit = audit_uploads(expression, 'own.csv', metadata, 'labels.csv')
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    assert not app.exception
    assert any('Data eksperimen notebook tidak tersedia' in item.value for item in app.error)
    assert app.button(key='training_start').disabled


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
    from gtex_explorer.training.run_labels import run_created
    run = tmp_path / 'run_x'
    run.mkdir()
    (run / 'config.json').write_text(json.dumps({'created': 1_000_000.0}), encoding='utf-8')
    os.utime(run / 'config.json', (time.time(), time.time()))
    assert run_created(run) == 1_000_000.0
    (run / 'config.json').write_text(json.dumps({}), encoding='utf-8')
    assert abs(run_created(run) - (run / 'config.json').stat().st_mtime) < 1e-6


def test_finished_runs_do_not_count_as_active(tmp_path):
    import json
    from gtex_explorer.training.jobs import has_active_run
    assert not has_active_run(tmp_path)
    status = tmp_path / 'runs' / ('run_' + 'c' * 32) / 'status.json'
    status.parent.mkdir(parents=True)
    for value, active in [('completed', False), ('failed', False), ('cancelled', False),
                          ('queued', True), ('running', True), ('paused', True), ('cancelling', True)]:
        status.write_text(json.dumps({'status': value}), encoding='utf-8')
        assert has_active_run(tmp_path) is active


def test_metadata_column_choices_explain_what_each_column_is_for(tmp_path, monkeypatch):
    import io
    from unittest.mock import patch

    class Upload(io.BytesIO):
        def __init__(self, content, name):
            super().__init__(content)
            self.name, self.size, self.file_id = name, len(content), name

    expression = Upload(b'SAMPID,ENSG1,ENSG2\nS1,1,2\nS2,3,4\n', 'tpm.csv')
    metadata = Upload(b'SAMPID,SUBJID,SMTSD\nS1,D1,Lung\nS2,D2,Liver\n', 'meta.csv')
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run(), 'Pelatihan Model')
    with patch('streamlit.file_uploader', side_effect=lambda label, **_: expression if 'ekspresi' in label else metadata):
        app.run()
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    for phrase in ['diambil dari judul kolom pada berkas metadata', 'ID Sampel menghubungkan metadata', 'ID harus sama persis',
                   'donor yang sama tidak dipisah antara data pelatihan dan validasi.', 'dipelajari dan diprediksi model',
                   'Periksa kembali kolom donor dan jaringan']:
        assert phrase in captions
    assert 'agar skor tidak terlalu tinggi' not in captions


def test_steps_follow_the_pipeline_and_the_review_uses_the_form_labels(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.button(key='training_to_configuration').click().run()
    configuration = list(app.tabs[1])
    position = lambda match: next(i for i, item in enumerate(configuration) if match(item))
    heading = lambda text: position(lambda item: item.type == 'subheader' and item.value == text)
    candidates = position(lambda item: getattr(item, 'key', None) == 'training_candidates')
    method = position(lambda item: getattr(item, 'key', None) == 'training_selector')
    # Candidate genes belong to preprocessing; the method and its gene count to feature selection.
    missing = position(lambda item: getattr(item, 'key', None) == 'training_missing')
    technical = position(lambda item: getattr(item, 'key', None) == 'training_technical_filter')
    # Pra-pemrosesan follows the pipeline: empty values, the technical filter, then the candidate count.
    assert heading('Pembagian data utama') < heading('Pra-pemrosesan') < missing < technical < candidates
    assert candidates < heading('Seleksi fitur') < method
    assert method < position(lambda item: getattr(item, 'key', None) == 'training_l1_c') < heading('Evaluasi tambahan')
    # The notebook data is named once in the data summary.
    assert not any(item.value.startswith('Data eksperimen notebook') for item in app.tabs[1].caption)

    app.button(key='training_to_review').click().run()
    assert not app.exception
    review = app.tabs[2]
    assert [item.value for item in review.subheader] == [
        'Tinjau data dan konfigurasi', 'Data dan pra-pemrosesan', 'Seleksi fitur dan model']
    data_table, model_table = (table.value for table in review.table)
    assert list(data_table.index) == ['Data development', 'Data holdout', 'Pembagian', 'Nilai kosong', 'Filter teknis',
                                      'Gen kandidat', 'Pra-pemrosesan']
    assert data_table.loc['Nilai kosong'].iloc[0] == 'Isi dengan median'
    assert data_table.loc['Filter teknis'].iloc[0] == 'Aktif'
    assert list(model_table.index) == ['Metode seleksi fitur', 'Top-k gen', 'C (regularisasi)', 'Evaluasi tambahan',
                                       'Classifier', 'Putaran boosting']
    assert review.expander[0].label == 'Parameter XGBoost dan opsi selector'
    parameters = review.expander[0].dataframe[0].value
    assert 'Learning rate' in parameters['Parameter'].tolist() and 'Bobot kelas' in parameters['Parameter'].tolist()
    # No internal setting names leak into the review.
    assert not {'selector', 'top_k', 'log2_transform', 'holdout_percent', 'learning_rate'} & set(parameters['Parameter'])
    app.button(key='training_review_back').click().run()
    app.selectbox(key='training_selector').set_value('anova').run()
    app.button(key='training_to_review').click().run()
    assert app.tabs[2].expander[0].label == 'Parameter XGBoost'
    assert 'C (regularisasi)' not in app.tabs[2].table[1].value.index



def test_process_guide_explains_every_setting_on_the_form(tmp_path, monkeypatch):
    import html
    from gtex_explorer.display import METHODS
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    guide = html.unescape(next(item.value for item in app.markdown if 'class="training-guide"' in item.value))
    # Streamlit's test harness lists an expander with an icon among its status elements.
    section = next(item for item in app.status if item.label == 'Cara kerja pelatihan')
    assert not section.proto.expanded
    # A guide, not a status: optional stages say what runs them instead of being marked skipped.
    assert 'Dilewati' not in guide and guide.count('>Opsional<') == 2
    for heading in ['Langkah penggunaan', 'Alur proses pelatihan', 'Pengaturan dan pengaruhnya', '>Pra-pemrosesan<']:
        assert heading in guide
    # Where pra-pemrosesan happens, with and without CV, and what the technical filter keeps; each said once.
    from gtex_explorer.training.data import TECHNICAL_FILTER_RULE
    assert guide.count('Pada setiap fold, pra-pemrosesan') == 1
    assert guide.count('proses pra-pemrosesan dilakukan satu kali pada data development') == 1
    assert guide.count(TECHNICAL_FILTER_RULE) == 1 and 'tetap untuk data notebook' not in guide
    assert 'Cross-validation aktif' not in guide  # no second telling of the run next to the flow
    # Every control of the Konfigurasi step, for every method, is explained under its own label.
    labels = set(METHODS.values())
    for method in ['l1', 'mutual_information', 'anova', 'baseline']:
        app.selectbox(key='training_selector').set_value(method).run()
        step = app.tabs[1]
        for kind in ['number_input', 'checkbox', 'slider', 'selectbox', 'expander']:
            labels |= {item.label for item in getattr(step, kind)}
    assert {'Bobot kelas', 'n_neighbors', 'Thread CPU', 'Jumlah fold', 'Porsi data holdout'} <= labels and len(labels) > 20
    missing = sorted(label for label in labels if label not in guide)
    assert not missing, missing


def test_guide_step_references_name_the_step_they_mean():
    import re
    from gtex_explorer.ui.components import training_guide as guide
    texts = [guide.PREPROCESSING_LEAD] + [text for _, items in guide.SETTINGS for _, text in items]
    titles = [title for title, _, _ in guide.FLOW]
    references = [match for text in texts for match in re.finditer(r'tahap (\d+)(?: \(([^)]+)\))?', text)]
    assert references
    for match in references:
        # Every "tahap N" carries the name of flow step N, so a renamed or renumbered step cannot go stale.
        assert match.group(2) == titles[int(match.group(1)) - 1], match.group(0)
