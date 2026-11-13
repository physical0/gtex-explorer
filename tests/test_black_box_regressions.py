"""Regressions observed in the 25 October UI acceptance run."""
import io
import json
import time

import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from test_training_page import APP


def saved_run(state, run_id, *, status='completed', error=None):
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import atomic_json
    root = state / 'runs' / run_id
    atomic_json(root / 'config.json', {'config': validate_training_config({'selector': 'anova', 'top_k': 500}).to_dict()})
    atomic_json(root / 'status.json', {'status': status, 'stage': status, 'updated': time.time(),
                                     'error': error, 'cv_summary': None})


def test_new_training_run_stays_selected_and_cancel_targets_that_run(tmp_path, monkeypatch):
    from gtex_explorer.training.jobs import atomic_json
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    # Only the OS process launch is suppressed. Job creation and cancellation are real.
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    older = 'run_' + '1' * 32
    saved_run(tmp_path, older)
    previous = 'run_' + '3' * 32
    saved_run(tmp_path, previous)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Hasil Pelatihan').click().run()
    app.selectbox(key='training_run').set_value(older).run()
    assert app.selectbox(key='training_run').value == older
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.button(key='training_start').click().run()
    assert not app.exception
    new_id = next(path.name for path in (tmp_path / 'runs').iterdir() if path.name not in {older, previous})
    assert app.selectbox(key='training_run').value == new_id
    app.run()
    assert app.selectbox(key='training_run').value == new_id
    cancel = app.button(key='training_cancel')
    cancel.click().run()
    assert (tmp_path / 'runs' / new_id / 'cancel.json').is_file()
    assert not (tmp_path / 'runs' / older / 'cancel.json').exists()
    # A worker completion/full-page refresh must not revert the selected run.
    atomic_json(tmp_path / 'runs' / new_id / 'status.json', {
        'status': 'completed', 'stage': 'completed', 'updated': time.time(), 'error': None, 'cv_summary': None})
    app.run()
    assert not app.exception
    assert app.selectbox(key='training_run').value == new_id
    app.selectbox(key='training_run').set_value(older).run()
    app.button(key='nav_Ringkasan').click().run()
    app.button(key='nav_Hasil Pelatihan').click().run()
    assert app.selectbox(key='training_run').value == older


def test_invalid_restored_round_count_cannot_break_prediction_page(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_string("\nimport streamlit as st\nfrom gtex_explorer.inference.model_bundle import load_model_bundle\nfrom gtex_explorer.ui.pages.prediction import _render_prediction_panel\nst.session_state['prediction_boosting_rounds'] = 999\npre, model = load_model_bundle()\n_render_prediction_panel(pre, model)\n", default_timeout=60).run()
    assert not app.exception
    assert app.selectbox(key='prediction_boosting_rounds').value == 300
    assert app.selectbox(key='prediction_boosting_rounds').proto.HasField('default')
    assert any('300 dari 300' in c.value for c in app.caption)


def test_empty_expression_file_explains_how_to_fix_input():
    from gtex_explorer.inference.parsing import parse_expression_file
    with pytest.raises(ValueError, match='kosong') as error:
        parse_expression_file(io.BytesIO(b''), 'empty.csv')
    assert 'kolom' in str(error.value).lower()


def test_blank_gene_count_has_a_readable_field_name(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Pelatihan Model').click().run()
    app.number_input(key='training_top_k').set_value(None).run()
    assert app.button(key='training_start').disabled
    errors = ' '.join(item.value.lower() for item in app.error)
    # Named by its label on the form.
    assert 'top-k gen' in errors and 'isi' in errors


def test_l1_failure_explains_support_and_recovery_action(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    saved_run(tmp_path, 'run_' + '2' * 32, status='failed', error={
        'type': 'SelectorEligibilityError',
        'message': 'L1 has 0 nonzero genes, fewer than top-k 5000. Lower k or adjust C.'})
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Hasil Pelatihan').click().run()
    errors = ' '.join(item.value.lower() for item in app.error)
    assert '0 gen' in errors and '5.000' in errors
    assert 'kurangi' in errors and 'c' in errors


def test_corrupt_saved_model_is_explained_and_cannot_be_selected(tmp_path, monkeypatch):
    from gtex_explorer.platform.model_registry import publish_model
    from test_model_registry import trained_fixture
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    result, config, demo = trained_fixture()
    entry = publish_model(result, config, {'classes': ['A', 'B', 'C']},
                          tmp_path / 'models' / 'run_corrupt', demo, None)
    with (entry.bundle_dir / 'xgboost_model.json').open('ab') as handle:
        handle.write(b' ')
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    assert not app.exception
    assert 'run_corrupt' not in app.selectbox(key='prediction_model').options
    notices = ' '.join(item.value for item in [*app.warning, *app.error, *app.text])
    assert 'run_corrupt' in notices
    assert 'integritas' in notices.lower() or 'hash' in notices.lower()
    app.button(key='load_local_demo').click().run()
    app.button(key='run_prediction').click().run()
    assert app.session_state['predictions'].shape[0] == len(app.session_state['audited_tpm'])


def test_corrupt_original_manifest_does_not_crash_registry(tmp_path):
    from gtex_explorer.platform.model_registry import list_models
    original = tmp_path / 'model_bundle'
    original.mkdir()
    for name in ['preprocessor.joblib', 'xgboost_model.json']:
        (original / name).write_bytes(b'bad')
    (original / 'bundle_manifest.json').write_text('{')
    failure = None
    try:
        models = list_models(tmp_path, tmp_path / 'state')
    except (ValueError, KeyError, TypeError) as error:
        failure = error
    assert failure is None, f'An invalid model manifest crashed the registry: {failure}'
    assert models == []
