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
        'from gtex_explorer.ui.pages.training import render_training_results_page\n'
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


def test_removing_a_gallery_model_deletes_its_files_and_selects_the_next(gallery_app, tmp_path):
    app, first, second = gallery_app
    assert len(list(tmp_path.rglob(first))) == 2  # The run and its model.
    app.button(key=f'gallery_select_{first}').click().run()
    assert app.button(key='training_delete_run').label == 'Hapus model'
    app.button(key='training_delete_run').click().run()
    assert not app.exception
    assert app.button(key=f'gallery_select_{first}')  # Nothing is removed before confirming.
    app.button(key='training_delete_confirm').click().run()
    assert not app.exception
    assert not list(tmp_path.rglob(first))
    assert not any(button.key == f'gallery_select_{first}' for button in app.button)
    assert app.session_state.training_gallery_run == second
    assert list(tmp_path.rglob(second))


def test_gallery_shows_macro_f1_and_scores_for_each_tissue(gallery_app):
    app, first, _ = gallery_app
    app.button(key=f'gallery_select_{first}').click().run()
    assert any(item.label == 'Rerata Macro-F1' for item in app.metric)
    assert 'F1-score per jaringan' in [item.value for item in app.subheader]
    assert not [item for item in app.caption if 'Latih ulang model' in item.value]


def test_gallery_explains_when_an_older_model_has_no_tissue_scores():
    app = AppTest.from_string('''
from gtex_explorer.ui.pages.training import _show_cv_results
row = {'fold': 1, 'macro_f1': .9, 'macro_precision': .9, 'macro_recall': .9, 'balanced_accuracy': .9, 'accuracy': .9, 'selected_genes': ['a']}
summary = {'fold_count': 2, 'fold_results': [row, {**row, 'fold': 2}], 'mean_jaccard': None, 'jaccard_pairs': [],
           **{prefix + key: .9 if prefix == 'mean_' else .01 for prefix in ['mean_', 'std_']
              for key in ['macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'accuracy']}}
_show_cv_results(summary, {'cv_folds': 2})
''', default_timeout=30).run()
    assert not app.exception
    assert any(item.label == 'Rerata Macro-F1' for item in app.metric)
    assert any('Latih ulang model' in item.value for item in app.caption)


def test_gallery_shows_holdout_scores_for_a_run_without_folds(tmp_path, monkeypatch):
    from test_training_gallery import holdout_run
    from gtex_explorer.training.worker import run_job
    data, state, run_id = holdout_run(tmp_path, monkeypatch)
    assert run_job(run_id, data, state) == 0
    app = AppTest.from_string('from pathlib import Path\n'
        'from gtex_explorer.ui.pages.training import render_training_results_page\n'
        f'render_training_results_page(Path({str(data)!r}), Path({str(state)!r}))', default_timeout=60).run()
    assert not app.exception
    assert 'Evaluasi holdout' in [item.value for item in app.subheader]
    assert 'F1-score per jaringan' in [item.value for item in app.subheader]
    assert any(item.label == 'Macro-F1' for item in app.metric)
    assert any('Holdout 50%' in item.value for item in app.markdown)


def test_gallery_shows_holdout_and_fold_results_together(tmp_path, monkeypatch):
    from test_training_gallery import holdout_run
    from gtex_explorer.training.worker import run_job
    data, state, run_id = holdout_run(tmp_path, monkeypatch, evaluate_cv=True, cv_folds=2)
    assert run_job(run_id, data, state) == 0
    app = AppTest.from_string('from pathlib import Path\n'
        'from gtex_explorer.ui.pages.training import render_training_results_page\n'
        f'render_training_results_page(Path({str(data)!r}), Path({str(state)!r}))', default_timeout=60).run()
    assert not app.exception
    headings = [item.value for item in app.subheader]
    assert headings.index('Evaluasi holdout') < headings.index('Evaluasi CV') and headings.count('F1-score per jaringan') == 2
    assert any('Holdout 50% dan 2 fold' in item.value for item in app.markdown)
