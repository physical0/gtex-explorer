"""Run names: given when training starts or later, kept apart from what the gallery verifies, shown everywhere."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from test_training_gallery import make_run
from ui_helpers import enter_application

APP = Path(__file__).resolve().parents[1] / 'main.py'


@pytest.fixture
def named_model(tmp_path, monkeypatch):
    """A finished, verified model whose run is named after it was saved."""
    from gtex_explorer.training.jobs import rename_run
    from gtex_explorer.training.worker import run_job
    data, state, run_id = make_run(tmp_path, monkeypatch)
    assert run_job(run_id, data, state) == 0
    rename_run(run_id, state, '  Percobaan *ANOVA*  ')
    return data, state, run_id


def test_names_are_trimmed_limited_to_one_line_and_unique_ignoring_case(tmp_path, monkeypatch):
    from gtex_explorer.training.jobs import clean_run_name, rename_run
    from gtex_explorer.training.run_labels import run_name
    assert clean_run_name('  Uji 1  ') == 'Uji 1'
    assert clean_run_name('   ') is None and clean_run_name(None) is None
    for bad in ['dua\nbaris', 'x' * 81]:
        with pytest.raises(ValueError, match='Nama pelatihan'):
            clean_run_name(bad)
    _, state, first = make_run(tmp_path, monkeypatch)
    assert rename_run(first, state, 'Uji ANOVA') == 'Uji ANOVA'
    assert run_name(state / 'runs' / first) == 'Uji ANOVA'
    # Renaming a run to its own name, in another case, is allowed.
    assert rename_run(first, state, 'uji anova') == 'uji anova'
    second = 'run_' + 'e' * 32
    (state / 'runs' / second).mkdir()
    (state / 'runs' / second / 'config.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='sudah digunakan'):
        rename_run(second, state, 'UJI ANOVA')
    # An empty name clears it.
    assert rename_run(first, state, '') is None
    assert run_name(state / 'runs' / first) is None and not (state / 'runs' / first / 'name.json').exists()


def test_a_name_given_at_start_is_saved_beside_the_settings_not_in_them(tmp_path, monkeypatch):
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.run_labels import run_name
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'xgboost': {'n_estimators': 2}},
                                      candidate_count=6)
    job = start_job(config, data, state, name='Uji awal')
    root = state / 'runs' / job.run_id
    assert run_name(root) == 'Uji awal'
    assert 'Uji awal' not in (root / 'config.json').read_text(encoding='utf-8')
    with pytest.raises(ValueError, match='sudah digunakan'):
        start_job(config, data, state, name='uji AWAL')


def test_renaming_a_verified_model_keeps_it_in_the_gallery_and_names_it_everywhere(named_model):
    from gtex_explorer.platform.model_registry import list_models
    from gtex_explorer.platform.paths import APP_ROOT
    from gtex_explorer.training.comparison import gallery_models
    from gtex_explorer.training.gallery import list_gallery
    from gtex_explorer.training.run_labels import run_label
    from gtex_explorer.ui.pages.training_comparison import _option_label
    _, state, run_id = named_model
    issues = []
    assert [entry['run_id'] for entry in list_gallery(state, issues=issues)] == [run_id] and not issues
    assert run_label(state / 'runs' / run_id, 'completed').startswith('✓ Selesai · Percobaan *ANOVA* · ')
    model = gallery_models(state)[0]
    assert model['name'] == 'Percobaan *ANOVA*'
    assert _option_label(model).startswith('Percobaan *ANOVA* · ANOVA')
    entry = next(entry for entry in list_models(APP_ROOT, state) if entry.model_id == run_id)
    assert entry.display_name.startswith('Percobaan *ANOVA* · ANOVA')


def test_hasil_pelatihan_leads_with_the_name_and_renames_from_a_dialog(named_model, monkeypatch):
    _, state, run_id = named_model
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(state))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Hasil Pelatihan').click().run()
    assert not app.exception
    # The gallery lists and heads the model by its name, shown as typed rather than as Markdown.
    assert any(r'Percobaan \*ANOVA\*' in item.value for item in app.subheader)
    assert r'**Percobaan \*ANOVA\***' in app.button(key='gallery_select_' + run_id).label
    app.text_input(key='training_gallery_search').set_value('percobaan').run()
    assert any(item.key == 'gallery_select_' + run_id for item in app.button)
    app.button(key='training_rename_gallery').click().run()
    assert not app.exception
    field = app.text_input(key='training_rename_name_' + run_id)
    assert field.value == 'Percobaan *ANOVA*'
    field.set_value('Model utama')
    next(button for button in app.button if button.label == 'Simpan nama').click().run()
    assert not app.exception
    assert json.loads((state / 'runs' / run_id / 'name.json').read_text(encoding='utf-8')) == {'name': 'Model utama'}
    # The search still says "percobaan", so the renamed model is filtered out until it is cleared.
    assert not any(item.key == 'gallery_select_' + run_id for item in app.button)
    app.text_input(key='training_gallery_search').set_value('').run()
    assert any(item.value == 'Model utama' for item in app.subheader)


def test_review_step_passes_the_name_to_the_new_run(tmp_path, monkeypatch):
    from gtex_explorer.training.run_labels import run_name
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.text_input(key='training_run_name').set_value('  Uji notebook  ').run()
    app.button(key='training_start').click().run()
    assert not app.exception
    runs = list((tmp_path / 'runs').iterdir())
    assert len(runs) == 1 and run_name(runs[0]) == 'Uji notebook'
    assert app.selectbox(key='training_run').value == runs[0].name
