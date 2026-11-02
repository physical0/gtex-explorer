import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score


def prepared_fixture():
    from gtex_explorer.training.data import PreparedData
    random = np.random.default_rng(42)
    labels = np.repeat(np.arange(3), 20)
    X = random.normal(size=(60, 12)).astype('float32')
    X[:, 0] += labels * 5
    X[:, 1] += (labels == 1) * 6
    return PreparedData(X[:48], labels[:48], tuple(f'ENSG{i:011d}' for i in range(12)),
                        tuple(f's{i}' for i in range(48)), tuple(f'd{i}' for i in range(48)), 'fixture',
                        X[48:], labels[48:], tuple(f's{i}' for i in range(48, 60)),
                        tuple(f'd{i}' for i in range(48, 60)), np.ones(12), np.zeros(12), np.ones(12))


@pytest.mark.parametrize('method', ['anova', 'mutual_information', 'l1', 'baseline'])
def test_selectors_use_training_data_and_actual_method_scores(tmp_path, method):
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector
    data = prepared_fixture()
    config = validate_training_config({'selector': method, 'top_k': 2}, candidate_count=12)
    result = fit_selector(data, config, tmp_path)
    assert len(result.selected_indices) == (12 if method == 'baseline' else 2)
    assert len(set(result.selected_indices)) == len(result.selected_indices)
    if method == 'anova':
        from sklearn.feature_selection import f_classif
        expected, p = f_classif(data.X_train, data.y_train)
        np.testing.assert_allclose(result.gene_diagnostics.selection_score, expected)
        np.testing.assert_allclose(result.gene_diagnostics.p_value, p)
    if method == 'l1':
        assert result.fit_diagnostics['converged']
        assert result.fit_diagnostics['nonzero_gene_count'] >= 2
        assert result.gene_diagnostics.iloc[result.selected_indices].selection_score.gt(1e-12).all()
    # Validation values and labels must not affect the selector.
    data.X_validation[:] = 1000000
    data.y_validation[:] = 0
    rerun = fit_selector(data, config, tmp_path / 'independent')
    np.testing.assert_array_equal(result.selected_indices, rerun.selected_indices)


def test_l1_rejects_zero_support_and_nonconvergence():
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector, SelectorEligibilityError
    data = prepared_fixture()
    with pytest.raises(SelectorEligibilityError, match='nonzero') as failure:
        fit_selector(data, validate_training_config({'l1_c': .0001, 'top_k': 2}, candidate_count=12))
    assert failure.value.diagnostics['nonzero_gene_count'] == 0
    with pytest.raises(SelectorEligibilityError, match='converge'):
        fit_selector(data, validate_training_config({'l1_c': 1000, 'top_k': 1,
                     'l1_max_iter': 100, 'l1_tol': .000001}, candidate_count=12))


def test_anova_rejects_training_partition_without_residual_degrees_of_freedom():
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector
    data = prepared_fixture()
    data.X_train = data.X_train[:3]
    data.y_train = np.arange(3)
    with pytest.raises(ValueError, match='ANOVA.*sampel training'):
        fit_selector(data, validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=12))


def test_mutual_information_runs_without_child_processes_and_preserves_scores(monkeypatch):
    # Frozen Windows executables cannot use loky's child-process entry point.
    from joblib._parallel_backends import LokyBackend
    from sklearn.feature_selection import mutual_info_classif
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector

    data = prepared_fixture()
    expected = mutual_info_classif(data.X_train, data.y_train, discrete_features=False,
                                  n_neighbors=3, random_state=42, n_jobs=1)

    def unavailable_child_process(*args, **kwargs):
        raise RuntimeError('Process workers unavailable in frozen application')

    monkeypatch.setattr(LokyBackend, 'configure', unavailable_child_process)
    config = validate_training_config({'selector': 'mutual_information', 'top_k': 3,
                                      'xgboost': {'n_jobs': 2}}, candidate_count=12)
    result = fit_selector(data, config)
    np.testing.assert_array_equal(result.gene_diagnostics.selection_score, expected)
    expected_order = np.lexsort((np.arange(len(expected)), -expected))[:3]
    np.testing.assert_array_equal(result.selected_indices, expected_order)


def test_ranking_reuse_does_not_reuse_classifier_or_top_k(tmp_path):
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector
    data = prepared_fixture()
    first = fit_selector(data, validate_training_config({'selector': 'anova', 'top_k': 2}, candidate_count=12), tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    second = fit_selector(data, validate_training_config({'selector': 'anova', 'top_k': 3,
                          'xgboost': {'max_depth': 2}}, candidate_count=12), tmp_path)
    assert before == sorted(p.name for p in tmp_path.iterdir())
    np.testing.assert_array_equal(first.selected_indices, second.selected_indices[:2])


def test_classifier_metrics_are_computed_from_validation_predictions():
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector, fit_classifier
    data = prepared_fixture()
    config = validate_training_config({'selector': 'anova', 'top_k': 2,
                                     'xgboost': {'n_estimators': 3}}, candidate_count=12)
    result = fit_classifier(data, fit_selector(data, config), config)
    pred = result.validation_predictions
    assert result.validation_metrics['macro_f1'] == f1_score(data.y_validation, pred, labels=[0, 1, 2], average='macro', zero_division=0)
    assert result.validation_metrics['macro_precision'] == precision_score(data.y_validation, pred, labels=[0, 1, 2], average='macro', zero_division=0)
    assert result.validation_metrics['macro_recall'] == recall_score(data.y_validation, pred, labels=[0, 1, 2], average='macro', zero_division=0)


def test_cv_requires_complete_folds_and_jaccard_is_not_baseline_stability():
    from gtex_explorer.training.core import aggregate_cv
    rows = [{'fold': i, 'macro_f1': x, 'macro_precision': x, 'macro_recall': x,
             'balanced_accuracy': x, 'accuracy': x} for i, x in enumerate([.6, .8, 1.], 1)]
    summary = aggregate_cv(rows, [{'a', 'b'}, {'b', 'c'}, {'b', 'd'}], expected_folds=3, supervised_selector=True)
    assert summary['mean_macro_f1'] == pytest.approx(.8)
    assert summary['std_macro_f1'] == pytest.approx(.2)
    assert len(summary['jaccard_pairs']) == 3
    assert summary['mean_jaccard'] == pytest.approx(1/3)
    baseline = aggregate_cv(rows, [], expected_folds=3, supervised_selector=False)
    assert baseline['mean_jaccard'] is None
    with pytest.raises(ValueError):
        aggregate_cv(rows[:2], [], expected_folds=3, supervised_selector=False)


def test_tissue_scores_average_back_to_macro_f1():
    from gtex_explorer.training.core import fit_selector, fit_classifier
    from gtex_explorer.training.config import validate_training_config
    data = prepared_fixture()
    config = validate_training_config({'selector': 'anova', 'top_k': 3, 'xgboost': {'n_estimators': 2}}, candidate_count=6)
    metrics = fit_classifier(data, fit_selector(data, config), config).validation_metrics
    assert len(metrics['tissue_f1']) == 3
    assert sum(metrics['tissue_f1']) / 3 == pytest.approx(metrics['macro_f1'])


def tissue_rows(scores):
    return [{'fold': i, 'macro_f1': .5, 'macro_precision': .5, 'macro_recall': .5, 'balanced_accuracy': .5,
             'accuracy': .5, **({'tissue_f1': row} if row else {})} for i, row in enumerate(scores, 1)]


def test_cv_summary_reports_each_tissue_across_folds():
    from gtex_explorer.training.core import aggregate_cv
    summary = aggregate_cv(tissue_rows([{'Liver': .6, 'Heart': 1.}, {'Liver': .8, 'Heart': 1.}, {'Liver': 1., 'Heart': 1.}]),
                           [], expected_folds=3, supervised_selector=False)
    assert summary['tissue_f1'] == [{'tissue': 'Liver', 'mean_f1': pytest.approx(.8), 'std_f1': pytest.approx(.2)},
                                    {'tissue': 'Heart', 'mean_f1': 1., 'std_f1': 0.}]
    with pytest.raises(ValueError):
        aggregate_cv(tissue_rows([{'Liver': .6}, {'Heart': 1.}, {'Liver': 1.}]), [], expected_folds=3, supervised_selector=False)
    with pytest.raises(ValueError):
        aggregate_cv(tissue_rows([{'Liver': 1.2}, {'Liver': .8}, {'Liver': 1.}]), [], expected_folds=3, supervised_selector=False)


def test_cv_summary_without_tissue_scores_is_unchanged_for_older_runs():
    from gtex_explorer.training.core import aggregate_cv
    old = aggregate_cv(tissue_rows([None, None, None]), [], expected_folds=3, supervised_selector=False)
    assert 'tissue_f1' not in old
    mixed = aggregate_cv(tissue_rows([{'Liver': 1.}, None, None]), [], expected_folds=3, supervised_selector=False)
    assert 'tissue_f1' not in mixed
