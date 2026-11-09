"""Unusual training data shapes, through the same upload route a user takes."""
import io
import json

import numpy as np
import pandas as pd
import pytest

from gtex_explorer.ui.shell.error_messages import user_error
from gtex_explorer.training.data import usable_candidate_genes


def upload(tissue_counts, *, genes=40, donors=None):
    rng = np.random.default_rng(1)
    tissues = np.repeat(np.arange(len(tissue_counts)), tissue_counts)
    samples = [f'S{i:04d}' for i in range(len(tissues))]
    values = rng.lognormal(3, .5, (len(tissues), genes))
    for row, tissue in enumerate(tissues):
        values[row, tissue * 3:tissue * 3 + 3] *= 6
    matrix = pd.DataFrame(values, index=pd.Index(samples, name='SAMPID'), columns=[f'ENSG{i:011d}' for i in range(genes)])
    meta = pd.DataFrame({'SAMPID': samples, 'SMTSD': [f'Tissue {t}' for t in tissues],
                         'SUBJID': donors or [f'D{i:04d}' for i in range(len(tissues))]})
    return matrix.to_csv().encode(), meta.to_csv(index=False).encode()


def train(tmp_path, monkeypatch, files, settings, candidates=None):
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.jobs import start_job
    from gtex_explorer.training.uploads import audit_uploads, save_training_upload
    from gtex_explorer.training.worker import run_job
    monkeypatch.setattr('gtex_explorer.training.jobs._launch', lambda *args: None)
    audit = audit_uploads(files[0], 'tpm.csv', files[1], 'meta.csv')
    count = candidates or len(audit.gene_names)
    root = save_training_upload(audit, tmp_path / 'datasets', count)
    config = validate_training_config({'selector': 'anova', 'top_k': min(count, 5), 'xgboost': {'n_estimators': 2, 'n_jobs': 1}, **settings},
                                      candidate_count=count)
    job = start_job(config, root, tmp_path)
    assert run_job(job.run_id, root, tmp_path) in (0, 1)
    return json.loads((tmp_path / 'runs' / job.run_id / 'status.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('tissue_counts,donors', [
    ([20, 20, 20], None),                                   # balanced
    ([60, 60, 6], None),                                    # one small tissue
    ([12] * 4, [f'D{d}' for _ in range(4) for d in range(12)]),  # every donor gives every tissue
    ([18, 18], [f'R{i // 3}' for i in range(36)]),          # repeated samples from a donor
])
def test_holdout_and_folds_finish_on_workable_shapes(tmp_path, monkeypatch, tissue_counts, donors):
    status = train(tmp_path, monkeypatch, upload(tissue_counts, donors=donors), {'holdout_percent': 25, 'evaluate_cv': True, 'cv_folds': 2})
    assert status['status'] == 'completed'
    holdout = status['holdout_summary']
    assert holdout['training_samples'] + holdout['validation_samples'] == sum(tissue_counts)
    assert sum(holdout['tissue_support'].values()) == holdout['validation_samples']
    for row in status['cv_summary']['fold_results']:
        assert row['training_samples'] + row['validation_samples'] == holdout['training_samples']


@pytest.mark.parametrize('tissue_counts,donors,settings,expected', [
    ([10, 10, 1], None, {'holdout_percent': 20}, 'Tissue 2'),                       # tissue with one sample
    ([6, 6], ['Only'] * 12, {'holdout_percent': 20}, 'Tissue 0'),                   # a single donor
    ([6, 6], ['Only'] * 12, {'evaluate_cv': True, 'cv_folds': 3}, 'jumlah fold'),   # folds from a single donor
    ([2, 2], None, {'evaluate_cv': True, 'cv_folds': 3}, 'jumlah fold'),            # more folds than samples per tissue
])
def test_unworkable_shapes_are_refused_before_training_in_plain_language(tmp_path, monkeypatch, tissue_counts, donors, settings, expected):
    with pytest.raises(ValueError) as error:
        train(tmp_path, monkeypatch, upload(tissue_counts, donors=donors), settings)
    message = user_error(error.value)
    assert expected in message
    assert not any(word in message for word in ['n_splits', 'is empty', 'cannot'])


def test_too_few_samples_after_holdout_explains_what_to_change(tmp_path, monkeypatch):
    status = train(tmp_path, monkeypatch, upload([2, 2]), {'holdout_percent': 20})
    assert status['status'] == 'failed'
    assert 'porsi data holdout' in user_error(status['error']['message'])


def inactive_gene_upload():
    tpm, meta = upload([20, 20, 20])
    matrix = pd.read_csv(io.BytesIO(tpm), index_col=0)
    matrix.iloc[:, 30:35] = 0
    matrix.iloc[:, 35:] = np.nan
    return matrix.to_csv().encode(), meta


def usable_on_form(files, *args):
    from gtex_explorer.training.uploads import audit_uploads
    from gtex_explorer.ui.pages.training import _upload_preview
    return usable_candidate_genes(_upload_preview(audit_uploads(files[0], 'tpm.csv', files[1], 'meta.csv')), *args)


@pytest.mark.parametrize('settings,arguments', [
    ({}, ()), ({'holdout_percent': 20}, (20,)), ({'holdout_percent': 20, 'evaluate_cv': True, 'cv_folds': 2}, (20, 2))])
def test_the_form_limit_is_exactly_what_training_accepts(tmp_path, monkeypatch, settings, arguments):
    files = inactive_gene_upload()
    limit = usable_on_form(files, *arguments)
    assert limit == 30
    (tmp_path / 'allowed').mkdir()
    assert train(tmp_path / 'allowed', monkeypatch, files, settings, candidates=limit)['status'] == 'completed'
    (tmp_path / 'above').mkdir()
    status = train(tmp_path / 'above', monkeypatch, files, settings, candidates=limit + 1)
    assert status['status'] == 'failed' and 'Kurangi Gen kandidat' in status['error']['message']


def test_the_form_limit_accounts_for_a_gene_active_only_outside_the_training_part():
    tpm, meta = upload([20, 20, 20])
    matrix = pd.read_csv(io.BytesIO(tpm), index_col=0)
    files = matrix.to_csv().encode(), meta
    from gtex_explorer.training.uploads import audit_uploads
    from gtex_explorer.training.data import make_holdout_assignment
    from gtex_explorer.ui.pages.training import _upload_preview
    split = make_holdout_assignment(_upload_preview(audit_uploads(*[files[0], 'tpm.csv', files[1], 'meta.csv'])), 50)
    # This gene is measurable only among the samples set aside, so it cannot be learned from.
    matrix.iloc[:, 0] = 0.
    matrix.iloc[split.validation_indices, 0] = 50.
    files = matrix.to_csv().encode(), meta
    assert usable_on_form(files) == 40
    assert usable_on_form(files, 50) == 39


def test_refused_splits_do_not_break_the_form_limit():
    assert usable_on_form(upload([6, 6], donors=['Only'] * 12), 20, 3) == 40
