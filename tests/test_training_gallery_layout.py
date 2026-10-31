"""Acceptance checks for the selected master-detail gallery design."""
import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def gallery_app(tmp_path, monkeypatch):
    from test_training_gallery import make_run
    from gtex_explorer.training.worker import run_job
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    data, state, first = make_run(tmp_path, monkeypatch, cv=True)
    assert run_job(first, data, state) == 0
    second = start_job(validate_training_config({'selector': 'baseline',
        'xgboost': {'n_estimators': 2}}, candidate_count=6), data, state).run_id
    assert run_job(second, data, state) == 0
    app = AppTest.from_string('from pathlib import Path\n'
        'from gtex_explorer.ui.training_page import render_training_results_page\n'
        f'render_training_results_page(Path({str(data)!r}), Path({str(state)!r}))',
        default_timeout=60).run()
    assert not app.exception
    return app, first, second


def test_gallery_has_searchable_list_and_no_completed_process_bar(gallery_app):
    app, first, second = gallery_app
    assert [tab.label for tab in app.tabs] == ['Gallery model', 'Status proses']
    assert app.session_state.training_results_view == 'Gallery model'
    assert app.text_input(key='training_gallery_search').label == 'Cari model atau dataset'
    assert {button.key for button in app.button} >= {f'gallery_select_{first}', f'gallery_select_{second}'}
    assert not app.get('progress')
    assert not any(heading.value == 'Status proses' for heading in app.subheader)


def test_filtering_selects_real_results_and_clear_restores_the_list(gallery_app):
    app, first, second = gallery_app
    app.text_input(key='training_gallery_search').set_value('anova').run()
    assert app.button(key=f'gallery_select_{first}')
    assert not any(button.key == f'gallery_select_{second}' for button in app.button)
    assert app.session_state.training_gallery_run == first
    assert any('Macro-F1' in str(frame.value) for frame in app.dataframe)
    app.text_input(key='training_gallery_search').set_value('BaSeLiNe').run()
    assert app.session_state.training_gallery_run == second
    assert any('Tanpa CV' in text.value for text in app.caption)
    assert not any('Macro-F1' in str(frame.value) for frame in app.dataframe)
    app.text_input(key='training_gallery_search').set_value('notebook').run()
    assert {button.key for button in app.button} >= {f'gallery_select_{first}', f'gallery_select_{second}'}
    app.text_input(key='training_gallery_search').set_value('').run()
    app.button(key=f'gallery_select_{first}').click().run()
    assert app.session_state.training_gallery_run == first
    assert app.session_state.training_selected_run == first


def test_no_matches_does_not_leave_a_stale_prediction_action(gallery_app):
    app, first, second = gallery_app
    app.text_input(key='training_gallery_search').set_value('no-such-dataset').run()
    assert not app.exception
    assert any('Tidak ada model' in item.value for item in app.info)
    assert not any(button.key == 'training_open_prediction' for button in app.button)
    assert not app.dataframe


def test_model_actions_precede_metrics_and_keep_correct_run_on_navigation(gallery_app):
    app, first, second = gallery_app
    app.button(key=f'gallery_select_{first}').click().run()
    tree = list(app)
    prediction_position = next(i for i, item in enumerate(tree) if getattr(item, 'key', None) == 'training_open_prediction')
    metric_position = next(i for i, item in enumerate(tree) if getattr(item, 'value', None) == 'Evaluasi CV' and item.type == 'subheader')
    assert prediction_position < metric_position
    app.button(key='training_open_prediction').click().run()
    assert app.session_state.prediction_model == first
    assert app.session_state.page == 'Prediksi Jaringan'
    app.button(key='training_open_genes').click().run()
    assert app.session_state.training_panel_run == first
    assert app.session_state.page == 'Panel Gen Pelatihan'
