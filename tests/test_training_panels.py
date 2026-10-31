import json
import numpy as np
import pytest
from test_training_uploads import uploads


def test_user_run_retains_dataset_and_real_final_and_fold_panels(tmp_path, monkeypatch):
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.worker import run_job
    from gtex_explorer.training.panels import load_run_panel
    expression, metadata = uploads()
    import io
    import pandas as pd
    original = pd.read_csv(io.BytesIO(expression), index_col=0)
    extra = original + .25
    extra.columns = [f's{i}' for i in range(7, 13)]
    expression = pd.concat([original, extra], axis=1).to_csv().encode()
    metadata = pd.DataFrame({'SAMPID': [f's{i}' for i in range(1, 13)],
                             'SUBJID': [f'd{i//2}' for i in range(12)],
                             'SMTSD': ['A', 'B'] * 6}).to_csv(index=False).encode()
    root = save_training_upload(audit_uploads(expression, 'mine.csv', metadata, 'labels.csv'), tmp_path / 'datasets', 3)
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    config = validate_training_config({'selector': 'anova', 'top_k': 2, 'evaluate_cv': True,
                                      'cv_folds': 2, 'xgboost': {'n_estimators': 2}}, candidate_count=3)
    state = tmp_path / 'state'
    job = start_job(config, root, state)
    saved = json.loads((state / 'runs' / job.run_id / 'config.json').read_text())
    assert saved['data_root'] == str(root.resolve())
    assert saved['dataset']['sample_count'] == 12
    assert saved['dataset']['source']['expression_file'] == 'mine.csv'
    assert run_job(job.run_id, root, state) == 0
    final, diagnostics = load_run_panel(state, job.run_id, 'final')
    fold, fold_info = load_run_panel(state, job.run_id, 1)
    assert len(final) == len(fold) == 2
    assert {'selection_score', 'p_value', 'gene'}.issubset(fold.columns)
    assert set(final.gene) <= {'OTHER_A', 'OTHER_B', 'OTHER_C'}
    assert fold_info['training_samples'] + fold_info['validation_samples'] == 12
    assert diagnostics['candidate_count'] == 3
    # Corrupt completed panels must be rejected, never quietly replaced/refitted.
    panel_file = state / 'runs' / job.run_id / 'fold_1_panel' / 'genes.csv'
    panel_file.write_text('corrupt')
    with pytest.raises(ValueError, match='hash'):
        load_run_panel(state, job.run_id, 1)


def test_resume_uses_saved_dataset_not_current_form_selection(tmp_path, monkeypatch):
    from gtex_explorer.training.jobs import atomic_json, resume_job
    root = tmp_path / 'state' / 'runs' / ('run_' + 'a' * 32)
    root.mkdir(parents=True)
    original = tmp_path / 'dataset_original'
    atomic_json(root / 'config.json', {'data_root': str(original)})
    atomic_json(root / 'status.json', {'status': 'cancelled', 'updated': 0, 'error': None})
    calls = []
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: calls.append(args))
    resume_job(root.name, tmp_path / 'some_other_dataset', tmp_path / 'state')
    assert calls[0][1] == original.resolve()
