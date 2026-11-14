"""Pra-pemrosesan choices for the candidate genes: how empty values are handled, and the technical filter."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from gtex_explorer.training.config import configuration_fingerprint, validate_training_config
from ui_helpers import enter_application

APP = Path(__file__).resolve().parents[1] / 'main.py'


def test_default_choices_keep_saved_configs_and_identities_unchanged():
    base = validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=6)
    explicit = validate_training_config({'selector': 'anova', 'top_k': 2, 'missing_values': 'median',
                                         'technical_filter': True}, candidate_count=6)
    # Runs saved before these choices existed have neither key and must keep their identity.
    assert 'missing_values' not in base.to_dict() and 'technical_filter' not in base.to_dict()
    assert base.to_dict() == explicit.to_dict()
    fingerprint = lambda config: configuration_fingerprint(config, 'data', 'split', {'numpy': '1'})
    assert fingerprint(base) == fingerprint(explicit)
    dropped = validate_training_config({'selector': 'anova', 'top_k': 2, 'missing_values': 'drop'}, candidate_count=6)
    unfiltered = validate_training_config({'selector': 'anova', 'top_k': 2, 'technical_filter': False},
                                          candidate_count=6)
    assert dropped.to_dict()['missing_values'] == 'drop' and unfiltered.to_dict()['technical_filter'] is False
    assert len({fingerprint(base), fingerprint(dropped), fingerprint(unfiltered)}) == 3
    assert validate_training_config(dropped.to_dict(), candidate_count=6) == dropped
    assert base.transforms() == {'log2': True, 'standardize': True, 'missing': 'median', 'technical_filter': True}
    with pytest.raises(ValueError, match='missing_values'):
        validate_training_config({'missing_values': 'mean'}, candidate_count=6)
    with pytest.raises(ValueError, match='technical_filter'):
        validate_training_config({'technical_filter': 'no'}, candidate_count=6)


def test_empty_values_are_filled_with_the_median_or_their_genes_left_out():
    from gtex_explorer.training.data import _handle_missing
    train = pd.DataFrame({'gap': [1.0, np.nan, 3.0], 'full': [2.0, 4.0, 6.0]})
    filled, medians = _handle_missing(train, 'median')
    assert list(filled.columns) == ['gap', 'full'] and filled.loc[1, 'gap'] == 2.0 and medians['gap'] == 2.0
    dropped, none = _handle_missing(train, 'drop')
    assert list(dropped.columns) == ['full'] and none is None


def test_without_the_filter_only_constant_genes_are_left_out():
    from gtex_explorer.training.data import _candidate_mask
    train = pd.DataFrame({'expressed': [5.0, 9.0, 7.0, 6.0, 8.0], 'rare': [0.0, 0.0, 0.0, 0.0, 0.4],
                          'constant': [3.0] * 5})
    assert _candidate_mask(train, True).to_dict() == {'expressed': True, 'rare': False, 'constant': False}
    assert _candidate_mask(train, False).to_dict() == {'expressed': True, 'rare': True, 'constant': False}


def test_a_run_trains_with_the_choices_and_names_them(tmp_path, monkeypatch):
    import json
    from scripts.export_training_data import export_training_data
    from test_training_data import checkpoint_sources
    import gtex_explorer.training.jobs as training_jobs
    from gtex_explorer.training.data import load_training_data, usable_candidate_genes
    from gtex_explorer.training.run_labels import method_description
    from gtex_explorer.training.worker import run_job
    data, state = tmp_path / 'data', tmp_path / 'state'
    export_training_data(*checkpoint_sources(tmp_path), data)
    monkeypatch.setattr(training_jobs, '_launch', lambda *args: None)
    loaded = load_training_data(data)
    # Without the filter at least as many genes can become candidates.
    assert usable_candidate_genes(loaded, technical_filter=False) >= usable_candidate_genes(loaded)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'evaluate_cv': True, 'cv_folds': 3,
                                       'missing_values': 'drop', 'technical_filter': False,
                                       'xgboost': {'n_estimators': 2}}, candidate_count=6)
    job = training_jobs.start_job(config, data, state)
    assert run_job(job.run_id, data, state) == 0
    assert training_jobs.read_job(job.run_id, state).status == 'completed'
    saved = json.loads((state / 'runs' / job.run_id / 'config.json').read_text(encoding='utf-8'))['config']
    assert saved['missing_values'] == 'drop' and saved['technical_filter'] is False
    assert method_description(saved) == 'ANOVA - 2 gen - gen bernilai kosong dibuang - tanpa filter teknis'


def test_form_offers_both_choices_and_the_review_shows_them(tmp_path, monkeypatch):
    monkeypatch.setenv('GTEX_DESKTOP_STATE_DIR', str(tmp_path))
    app = enter_application(AppTest.from_file(str(APP), default_timeout=60).run())
    app.button(key='nav_Pelatihan Model').click().run()
    app.radio(key='training_source').set_value('Data eksperimen notebook').run()
    app.button(key='training_to_configuration').click().run()
    assert app.selectbox(key='training_missing').value == 'median'
    assert app.checkbox(key='training_technical_filter').value is True
    app.selectbox(key='training_missing').set_value('drop').run()
    app.checkbox(key='training_technical_filter').uncheck().run()
    assert not app.exception
    app.button(key='training_to_review').click().run()
    data_table = app.tabs[2].table[0].value
    assert data_table.loc['Nilai kosong'].iloc[0] == 'Buang gen yang memiliki nilai kosong'
    assert data_table.loc['Filter teknis'].iloc[0] == 'Nonaktif'
