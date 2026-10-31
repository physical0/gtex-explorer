"""Training-only selector fits and XGBoost evaluation, independent of Streamlit."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
from itertools import combinations
import json
import os
from pathlib import Path
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from gtex_explorer.training.config import TrainingConfig
from gtex_explorer.training.data import PreparedData, file_hash, stable_hash, verify_files, atomic_publish_directory


@dataclass
class SelectorFit:
    ranked_indices: np.ndarray
    selected_indices: np.ndarray
    gene_diagnostics: pd.DataFrame
    fit_diagnostics: dict
    selector_seconds: float


@dataclass
class FitResult:
    classifier: XGBClassifier
    prepared: PreparedData
    selector: SelectorFit
    validation_metrics: dict | None
    validation_predictions: np.ndarray | None
    classifier_seconds: float


class SelectorEligibilityError(ValueError):
    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class TrainingCancelled(RuntimeError):
    """An interrupted fit must never be evaluated or published as a full model."""


class _RoundProgress(TrainingCallback):
    def __init__(self, report):
        self.report = report

    def before_training(self, model):
        self.report(0)
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.report(epoch + 1)
        return False


def runtime_versions() -> dict:
    return {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost']}


def fit_selector(prepared: PreparedData, config: TrainingConfig,
                 ranking_cache: Path | None = None) -> SelectorFit:
    count = prepared.X_train.shape[1]
    if not 1 <= config.top_k <= count:
        raise ValueError('Requested genes exceed candidate count')
    if config.selector == 'anova' and len(prepared.y_train) <= len(np.unique(prepared.y_train)):
        raise ValueError('ANOVA memerlukan jumlah sampel training lebih besar daripada jumlah kelas. '
                         'Tambahkan sampel/donor atau pilih pembagian fold lain.')
    key = stable_hash({'partition': prepared.fingerprint, 'genes': list(prepared.gene_names),
                       'settings': config.selector_settings(), 'versions': runtime_versions()})
    target = Path(ranking_cache) / key if ranking_cache else None
    if target and target.is_dir():
        meta = json.loads((target / 'ranking.json').read_text())
        if meta['key'] != key:
            raise ValueError('Ranking cache identity mismatch')
        verify_files(target, meta['files'])
        frame = pd.read_csv(target / 'genes.csv', float_precision='round_trip')
        if frame.gene.tolist() != list(prepared.gene_names):
            raise ValueError('Ranking gene axis mismatch')
        info = meta['diagnostics']
        elapsed = meta['seconds']
    else:
        started = time.monotonic()
        frame = pd.DataFrame({'gene': prepared.gene_names})
        info = {'selector': config.selector, 'candidate_count': count}
        if config.selector == 'baseline':
            scores = np.zeros(count)
        elif config.selector == 'anova':
            scores, p_values = f_classif(prepared.X_train, prepared.y_train)
            frame['p_value'] = p_values
            info['pvalues_are_donor_aware'] = False
        elif config.selector == 'mutual_information':
            # Threads avoid loky's incompatible subprocess entry point in frozen Windows builds.
            # The same seeded estimator and gene ranking are used in source and desktop runs.
            with parallel_backend('threading'):
                scores = mutual_info_classif(prepared.X_train, prepared.y_train, discrete_features=False,
                                            n_neighbors=config.mi_neighbors, random_state=42,
                                            n_jobs=config.xgboost_parameters['n_jobs'])
            info['n_neighbors'] = config.mi_neighbors
            info['positive_score_gene_count'] = int(np.sum(scores > 0))
        else:
            estimator = LogisticRegression(penalty='l1', solver='saga', C=config.l1_c,
                                           class_weight=config.l1_class_weight,
                                           max_iter=config.l1_max_iter, tol=config.l1_tol,
                                           random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                estimator.fit(prepared.X_train, prepared.y_train)
            scores = np.max(np.abs(estimator.coef_), axis=0)
            frame['nonzero_class_count'] = (np.abs(estimator.coef_) > 1e-12).sum(axis=0)
            info.update(converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught),
                        iterations=int(estimator.n_iter_.max()), l1_c=config.l1_c,
                        nonzero_gene_count=int(np.sum(scores > 1e-12)),
                        sparsity=1 - float(np.mean(scores > 1e-12)))
        scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf,
                               posinf=np.inf, neginf=-np.inf)
        order = np.lexsort((np.arange(count), -scores))
        rank = np.empty(count, dtype=int)
        rank[order] = np.arange(1, count + 1)
        frame['selection_rank'] = rank
        if config.selector != 'baseline':
            frame['selection_score'] = scores
        elapsed = time.monotonic() - started
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='ranking-', dir=target.parent) as directory:
                root = Path(directory) / 'data'
                root.mkdir()
                frame.to_csv(root / 'genes.csv', index=False)
                meta = {'key': key, 'diagnostics': info, 'seconds': elapsed,
                        'files': {'genes.csv': file_hash(root / 'genes.csv')}}
                (root / 'ranking.json').write_text(json.dumps(meta, allow_nan=False), encoding='utf-8')
                atomic_publish_directory(root, target)
    info = {**info, 'requested_k': config.top_k}
    if config.selector == 'l1':
        if not info['converged']:
            raise SelectorEligibilityError('L1 did not converge. Increase iterations or adjust tolerance.', info)
        if info['nonzero_gene_count'] < config.top_k:
            raise SelectorEligibilityError(f'L1 has {info["nonzero_gene_count"]} nonzero genes, fewer than top-k {config.top_k}. Lower k or adjust C.', info)
    order = np.argsort(frame.selection_rank.to_numpy(), kind='stable')
    return SelectorFit(order, order[:config.top_k], frame, info, elapsed)


def fit_classifier(prepared: PreparedData, selector: SelectorFit, config: TrainingConfig,
                   progress=None) -> FitResult:
    classes = np.unique(prepared.y_train)
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError('Training labels must contain the full encoded class mapping')
    if len(classes) < 2:
        raise ValueError('At least two tissue classes are required')
    parameters = dict(config.xgboost_parameters)
    if progress is not None:
        parameters['callbacks'] = [_RoundProgress(progress)]
    classifier = XGBClassifier(num_class=len(classes), **parameters)
    started = time.monotonic()
    classifier.fit(prepared.X_train[:, selector.selected_indices], prepared.y_train)
    elapsed = time.monotonic() - started
    predictions, metrics = None, None
    if prepared.X_validation is not None:
        # Explicit argmax also handles two-class fixtures with multi:softprob.
        probabilities = classifier.predict_proba(prepared.X_validation[:, selector.selected_indices])
        predictions = np.asarray(probabilities).argmax(axis=1).astype(int)
        precision = precision_score(prepared.y_validation, predictions, labels=classes, average='macro', zero_division=0)
        recall = recall_score(prepared.y_validation, predictions, labels=classes, average='macro', zero_division=0)
        metrics = {'macro_f1': float(f1_score(prepared.y_validation, predictions, labels=classes, average='macro', zero_division=0)),
                   'macro_precision': float(precision), 'macro_recall': float(recall),
                   'balanced_accuracy': float(recall),
                   'accuracy': float(accuracy_score(prepared.y_validation, predictions))}
    return FitResult(classifier, prepared, selector, metrics, predictions, elapsed)


def aggregate_cv(results: list[dict], selected_sets: list[set[str]], *,
                 expected_folds: int, supervised_selector: bool) -> dict:
    if len(results) != expected_folds or {r['fold'] for r in results} != set(range(1, expected_folds + 1)):
        raise ValueError('CV requires all distinct successful folds')
    summary = {'fold_count': expected_folds, 'fold_results': sorted(results, key=lambda r: r['fold']),
               'mean_jaccard': None, 'jaccard_pairs': []}
    for metric in ['macro_f1', 'macro_precision', 'macro_recall', 'balanced_accuracy', 'accuracy']:
        values = np.array([row[metric] for row in results], dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Invalid CV metrics')
        summary['mean_' + metric] = float(values.mean())
        summary['std_' + metric] = float(values.std(ddof=1))
    if supervised_selector:
        if len(selected_sets) != expected_folds or any(not genes for genes in selected_sets):
            raise ValueError('Missing selected gene sets')
        for a, b in combinations(range(expected_folds), 2):
            overlap = len(selected_sets[a] & selected_sets[b]) / len(selected_sets[a] | selected_sets[b])
            summary['jaccard_pairs'].append({'fold_a': a + 1, 'fold_b': b + 1, 'jaccard': overlap})
        summary['mean_jaccard'] = float(np.mean([p['jaccard'] for p in summary['jaccard_pairs']]))
    return summary
