"""Recompute exported metrics independently of the notebook/app helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gtex_explorer.analysis.core import load_reference_tables


def test_cv_counts_fold_identity_parameters_and_aggregation() -> None:
    tables = load_reference_tables()
    folds = tables['cv_folds']
    summary = tables['cv_summary'].set_index('configuration_id').sort_index()
    assert len(folds) == 80 and folds.configuration_id.nunique() == 16
    assert not folds.duplicated(['configuration_id', 'fold']).any()
    assert folds.groupby('configuration_id').size().eq(5).all()
    assert all(set(group.fold) == {1, 2, 3, 4, 5} for _, group in folds.groupby('configuration_id'))
    assert folds.groupby('fold').fold_identity_sha256.nunique().eq(1).all()
    assert folds.xgboost_parameters_sha256.nunique() == 1
    baseline = folds.loc[folds.experiment_type.eq('baseline')]
    assert len(baseline) == 5 and baseline.genes.eq(5000).all()
    assert not baseline.uses_feature_selection.any()
    for metric in ['macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'accuracy']:
        recomputed = folds.groupby('configuration_id')[metric].agg(['mean', 'std']).sort_index()
        np.testing.assert_allclose(summary[f'mean_{metric}'], recomputed['mean'], atol=1e-12)
        np.testing.assert_allclose(summary[f'std_{metric}'], recomputed['std'], atol=1e-12)


def test_final_holdout_metrics_match_saved_predictions_and_overview() -> None:
    tables = load_reference_tables()
    predictions = tables['final_holdout_predictions']
    assert len(predictions) == 193 and predictions.SUBJID.nunique() == 124
    assert not predictions.SAMPID.duplicated().any()
    metrics = {
        'macro_f1': lambda y, p: f1_score(y, p, average='macro', zero_division=0),
        'macro_precision': lambda y, p: precision_score(y, p, average='macro', zero_division=0),
        'macro_recall': lambda y, p: recall_score(y, p, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score,
        'accuracy': accuracy_score,
    }
    for kind, prediction_column in [('feature_selection', 'selected_model_prediction'), ('baseline', 'all_5000_prediction')]:
        rows = tables['final_holdout'].loc[tables['final_holdout'].experiment_type.eq(kind)]
        assert len(rows) == 1
        for metric, calculate in metrics.items():
            value = calculate(predictions.true_tissue, predictions[prediction_column])
            assert np.isclose(rows.iloc[0][metric], value, atol=1e-12)
            if kind == 'feature_selection':
                assert np.isclose(tables['final_test'].iloc[0][metric], value, atol=1e-12)
