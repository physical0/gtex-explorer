import re
import time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.ui.components.training_guide import FLOW
from gtex_explorer.ui.components.training_progress import progress_html, run_progress

APP = Path(__file__).resolve().parents[1] / 'main.py'
CV_HOLDOUT = {'evaluate_cv': True, 'cv_folds': 5, 'holdout_percent': 20}
GUIDE_STAGES = [title for title, _, _ in FLOW]


def states(progress):
    return {step.title: step.state for step in progress.steps}


def test_stages_use_the_guide_names_and_only_the_ones_the_run_uses():
    every = run_progress(CV_HOLDOUT, 'queued', {'stage': 'queued'})
    assert [step.title for step in every.steps] == GUIDE_STAGES
    plain = run_progress({'evaluate_cv': False, 'cv_folds': 5}, 'queued', {'stage': 'queued'})
    assert [step.title for step in plain.steps] == [
        'Inisialisasi pelatihan', 'Pembagian data utama', 'Pelatihan model final', 'Penyimpanan dan verifikasi']
    assert states(plain)['Pelatihan model final'] == 'waiting'
    assert next(step.meta for step in plain.steps if step.title == 'Pelatihan model final') == 'Seluruh sampel'


@pytest.mark.parametrize('details,stage,title,substep,text', [
    ({'stage': 'splitting_data'}, 'Pembagian data utama', 'Pembagian data utama', None, 'data holdout'),
    ({'stage': 'preparing_fold', 'fold': 2}, 'Cross-validation', 'Cross-validation · fold 2 dari 5', 'Pra-pemrosesan',
     'data validation fold 2'),
    ({'stage': 'fitting_selector', 'fold': 2}, 'Cross-validation', 'Cross-validation · fold 2 dari 5', 'Seleksi fitur',
     'data training fold 2'),
    ({'stage': 'fitting_xgboost', 'fold': 2, 'rounds_completed': 60, 'rounds_total': 300}, 'Cross-validation',
     'Cross-validation · fold 2 dari 5', 'XGBoost', None),
    # Once every round is done the fit scores its validation fold.
    ({'stage': 'fitting_xgboost', 'fold': 2, 'rounds_completed': 300, 'rounds_total': 300}, 'Cross-validation',
     'Cross-validation · fold 2 dari 5', 'Evaluasi', None),
    ({'stage': 'final_preprocessing'}, 'Pelatihan model final', 'Pelatihan model final', 'Pra-pemrosesan',
     'data development'),
    ({'stage': 'final_xgboost', 'rounds_completed': 60, 'rounds_total': 300}, 'Pelatihan model final',
     'Pelatihan model final', 'XGBoost', None),
    ({'stage': 'final_xgboost', 'rounds_completed': 300, 'rounds_total': 300}, 'Evaluasi holdout', 'Evaluasi holdout',
     None, 'data holdout'),
    ({'stage': 'verifying_results'}, 'Penyimpanan dan verifikasi', 'Penyimpanan dan verifikasi', 'Verifikasi', None),
])
def test_each_reported_stage_names_its_stage_step_and_data(details, stage, title, substep, text):
    progress = run_progress(CV_HOLDOUT, 'running', details)
    found = states(progress)
    assert found[stage] == 'current'
    position = GUIDE_STAGES.index(stage)
    assert all(found[name] == 'done' for name in GUIDE_STAGES[:position])
    assert all(found[name] == 'waiting' for name in GUIDE_STAGES[position + 1:])
    assert progress.title == title
    assert [name for name, state in progress.substeps if state == 'current'] == ([substep] if substep else [])
    if text:
        assert text in progress.text
    assert progress.label == 'Berjalan'


def test_fold_progress_and_rounds_say_which_fit_they_count():
    progress = run_progress(CV_HOLDOUT, 'running', {'stage': 'fitting_xgboost', 'fold': 3})
    metas = {step.title: step.meta for step in progress.steps}
    assert metas['Cross-validation'] == 'Fold 3 dari 5'
    assert progress.rounds_label == 'Putaran XGBoost fold 3'
    final = run_progress(CV_HOLDOUT, 'running', {'stage': 'final_xgboost'})
    assert {step.title: step.meta for step in final.steps}['Cross-validation'] == '5 fold selesai'
    assert final.rounds_label == 'Putaran XGBoost model final'


def test_without_holdout_the_final_fit_finishes_without_an_evaluation_stage():
    progress = run_progress({'evaluate_cv': False, 'cv_folds': 5}, 'running',
                            {'stage': 'final_xgboost', 'rounds_completed': 300, 'rounds_total': 300})
    assert states(progress)['Pelatihan model final'] == 'current'
    assert all(state == 'done' for _, state in progress.substeps)
    assert progress.text == 'Dilatih pada seluruh sampel.'


@pytest.mark.parametrize('status,details,label,tone', [
    ('paused', {'stage': 'fitting_xgboost', 'fold': 2}, 'Dijeda', 'paused'),
    ('cancelled', {'stage': 'cancelled', 'cancelled_stage': 'fitting_selector', 'fold': 2}, 'Dihentikan', 'stopped'),
    ('failed', {'stage': 'failed', 'failed_stage': 'preparing_fold', 'fold': 2}, 'Gagal', 'failed'),
    # A worker that died keeps its last stage.
    ('failed', {'stage': 'fitting_xgboost', 'fold': 2}, 'Gagal', 'failed'),
])
def test_a_stopped_run_shows_the_stage_it_stopped_at(status, details, label, tone):
    progress = run_progress(CV_HOLDOUT, status, details)
    assert states(progress)['Cross-validation'] == 'current'
    assert progress.title == 'Cross-validation · fold 2 dari 5'
    assert (progress.label, progress.tone) == (label, tone)


def test_completed_run_marks_every_stage_done():
    progress = run_progress(CV_HOLDOUT, 'completed', {'stage': 'completed', 'rounds_completed': 300,
                                                       'rounds_total': 300})
    assert set(states(progress).values()) == {'done'}
    assert progress.title == 'Semua tahap selesai'
    assert not progress.substeps


def test_markup_marks_the_current_stage_for_screen_readers():
    markup = progress_html(run_progress(CV_HOLDOUT, 'running', {'stage': 'fitting_selector', 'fold': 2}))
    assert markup.count('aria-current="step"') == 1
    assert 'Cross-validation<span class="run-hidden"> (tahap saat ini)</span>' in markup
    assert 'Seleksi fitur<span class="run-hidden"> (tahap saat ini)</span>' in markup


def test_status_tab_shows_the_running_fold_and_what_its_rounds_count(tmp_path, monkeypatch):
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import atomic_json, process_lock
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    run_id = 'run_' + 'f' * 32
    root = tmp_path / 'runs' / run_id
    config = validate_training_config({'selector': 'anova', 'top_k': 500, 'evaluate_cv': True, 'cv_folds': 5,
                                       'holdout_percent': 20}).to_dict()
    atomic_json(root / 'config.json', {'config': config})
    atomic_json(root / 'status.json', {'status': 'running', 'stage': 'fitting_xgboost', 'fold': 2,
                                       'rounds_completed': 150, 'rounds_total': 300, 'updated': time.time(),
                                       'elapsed_seconds': 1234, 'error': None})
    with process_lock(tmp_path / 'worker.lock'):
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        app = enter_application(app)
        app.button(key='nav_Hasil Pelatihan').click().run()
        assert not app.exception
        markup = next(item.value for item in app.markdown if 'class="run-progress' in item.value)
        assert 'Cross-validation · fold 2 dari 5' in markup
        assert 'Dilatih pada data training fold 2, lalu dievaluasi pada data validation fold 2.' in markup
        bar = app.get('progress')[0].proto
        assert bar.value == 50
        assert bar.text == 'Putaran XGBoost fold 2: 150 dari 300'
        # A running run keeps counting from its last update, so only the thousands format is fixed.
        assert any(re.search(r'Waktu proses: 1\.2\d\d detik', item.value) for item in app.caption)


def test_worker_reports_each_stage_and_remembers_where_a_failure_stopped(tmp_path, monkeypatch):
    import json
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.config import validate_training_config
    import gtex_explorer.training.jobs as training_jobs
    import gtex_explorer.training.worker as worker
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr(training_jobs, '_launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'xgboost': {'n_estimators': 5}},
                                      candidate_count=6)
    job = training_jobs.start_job(config, data, state)
    stages = []
    real_write = worker.atomic_json

    def record_stage(path, value):
        if Path(path).name == 'status.json' and (not stages or stages[-1] != value.get('stage')):
            stages.append(value.get('stage'))
        return real_write(path, value)

    def broken_selector(*args, **kwargs):
        raise ValueError('selector broke')

    monkeypatch.setattr(worker, 'atomic_json', record_stage)
    monkeypatch.setattr(worker, 'fit_selector', broken_selector)
    assert worker.run_job(job.run_id, data, state) == 1
    assert stages == ['splitting_data', 'final_preprocessing', 'final_selector', 'failed']
    saved = json.loads((training_jobs.job_path(job.run_id, state) / 'status.json').read_text(encoding='utf-8'))
    assert saved['failed_stage'] == 'final_selector'
    progress = run_progress(config.to_dict(), 'failed', saved)
    assert progress.title == 'Pelatihan model final'
    assert [name for name, state in progress.substeps if state == 'current'] == ['Seleksi fitur']


def test_run_picker_leads_each_run_with_its_status(tmp_path, monkeypatch):
    from test_black_box_regressions import saved_run
    from gtex_explorer.training.jobs import process_lock
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    runs = {'run_' + letter * 32: status for letter, status in
            [('1', 'completed'), ('2', 'failed'), ('3', 'cancelled'), ('4', 'running')]}
    for run_id, status in runs.items():
        saved_run(tmp_path, run_id, status=status)
    with process_lock(tmp_path / 'worker.lock'):
        app = AppTest.from_file(str(APP), default_timeout=60).run()
        app = enter_application(app)
        app.button(key='nav_Hasil Pelatihan').click().run()
        assert not app.exception
        picker = app.selectbox(key='training_run')
        labels = {run_id: next(label for label in picker.options if label.endswith(run_id[-6:])) for run_id in runs}
        assert labels['run_' + '1' * 32].startswith('✓ Selesai · ')
        assert labels['run_' + '2' * 32].startswith('✕ Gagal · ')
        assert labels['run_' + '3' * 32].startswith('■ Dihentikan · ')
        assert labels['run_' + '4' * 32].startswith('● Berjalan · ')


def test_dead_worker_shows_as_failed_in_the_picker(tmp_path, monkeypatch):
    # Without a worker holding its lock, a run left as running has stopped.
    from test_black_box_regressions import saved_run
    from gtex_explorer.ui.components.training_runs import _run_status
    run_id = 'run_' + '5' * 32
    saved_run(tmp_path, run_id, status='running')
    assert _run_status(tmp_path / 'runs' / run_id, tmp_path) == 'failed'


def test_run_label_without_status_is_unchanged(tmp_path):
    from test_black_box_regressions import saved_run
    from gtex_explorer.training.run_labels import run_label
    run_id = 'run_' + '6' * 32
    saved_run(tmp_path, run_id)
    plain = run_label(tmp_path / 'runs' / run_id)
    assert plain.endswith('666666') and not plain.startswith('✓')
    assert run_label(tmp_path / 'runs' / run_id, 'completed') == '✓ Selesai · ' + plain
