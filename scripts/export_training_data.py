"""One-time export from explicitly trusted local notebook checkpoints only."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from gtex_explorer.training.data import file_hash, load_training_data, unique_ids, atomic_publish_directory


def export_training_data(cache_path: Path, preprocessing_path: Path,
                         folds_path: Path, output_dir: Path) -> dict:
    # Joblib may execute code: these are project-owned checkpoint paths, not uploads.
    import joblib
    cache = joblib.load(cache_path)
    pre = joblib.load(preprocessing_path, mmap_mode='r')
    folds = joblib.load(folds_path, mmap_mode='r')
    fingerprint = cache['cache_metadata']['experiment_fingerprint']
    if any(p['experiment_fingerprint'] != fingerprint for p in [pre, folds]):
        raise ValueError('Checkpoint fingerprint mismatch')
    values, final, fold_values = cache['variables'], pre['variables'], folds['variables']
    raw = fold_values['X_development_raw']
    log = final['X_development_log']
    if list(raw.index) != list(log.index) or not log.equals(values['X_development_log']):
        raise ValueError('Development axes/values mismatch')
    samples = unique_ids(list(raw.index))
    unique_ids(list(raw.columns), genes=True)
    if set(samples) & set(values['X_test_log'].index):
        raise ValueError('Holdout overlap')
    meta = values['metadata'].set_index(values['SAMPLE_COLUMN'])
    if meta.index.has_duplicates:
        raise ValueError('Duplicate sample metadata')
    tissues = meta.loc[list(samples), values['TISSUE_COLUMN']].astype(str)
    classes = sorted(tissues.unique())
    labels = np.array([classes.index(t) for t in tissues], dtype=np.int64)
    donors = meta.loc[list(samples), values['DONOR_COLUMN']].astype(str).tolist()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='training-export-', dir=output_dir.parent) as temporary:
        root = Path(temporary) / 'payload'
        root.mkdir()
        def save(name, array):
            np.save(root / name, np.asarray(array), allow_pickle=False)
            return name
        manifest = {'schema_version': 1, 'source_fingerprint': fingerprint,
                    'sample_ids': list(samples), 'donor_ids': donors, 'classes': classes,
                    'holdout_included': False, 'candidate_count': len(log.columns),
                    'preprocessing_version': 'notebook-v4-training-local-1',
                    'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'xgboost']},
                    'median_provenance': 'inverse log2 of development-imputed TPM, then gene median',
                    'sources': {p.name: file_hash(p) for p in [cache_path, preprocessing_path, folds_path]},
                    'cached_folds': []}
        save('labels.npy', labels)
        save('raw_tpm.npy', raw.to_numpy(dtype=np.float32))
        (root / 'raw_genes.json').write_text(json.dumps(list(raw.columns)), encoding='utf-8')
        scaler = final['final_scaler']
        final_log = log.to_numpy(dtype=np.float64)
        manifest['final'] = {'genes': list(log.columns), 'train_samples': list(samples),
                             'X_train': save('final_scaled.npy', final['X_development_scaled']),
                             'training_log': save('final_log.npy', final_log),
                             'raw_tpm_medians': save('medians.npy', np.median(np.exp2(final_log) - 1, axis=0)),
                             'scaler_mean': save('mean.npy', scaler.mean_),
                             'scaler_scale': save('scale.npy', scaler.scale_)}
        for fold in fold_values['prepared_cv_folds']:
            number = int(fold['fold'])
            spec = {'fold': number, 'genes': list(map(str, fold['gene_names'])),
                    'train_samples': list(map(str, fold['train_sample_ids'])),
                    'validation_samples': list(map(str, fold['validation_sample_ids'])),
                    'X_train': save(f'fold_{number}_train.npy', fold['X_train']),
                    'X_validation': save(f'fold_{number}_validation.npy', fold['X_validation'])}
            for side, key in [('train_samples', 'y_train'), ('validation_samples', 'y_validation')]:
                expected = [labels[samples.index(s)] for s in spec[side]]
                if not np.array_equal(expected, fold[key]):
                    raise ValueError('Fold label mapping mismatch')
            manifest['cached_folds'].append(spec)
        manifest['files'] = {p.name: file_hash(p) for p in root.iterdir() if p.is_file()}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        loaded = load_training_data(root)
        report = {'samples': len(loaded.sample_ids), 'donors': len(set(loaded.donor_ids)),
                  'classes': len(loaded.classes), 'candidate_genes': len(log.columns),
                  'cached_folds': len(manifest['cached_folds']), 'holdout_included': False,
                  'data_fingerprint': loaded.fingerprint}
        # Windows cannot rename directories with active memory-mapped handles.
        del loaded
        import gc
        gc.collect()
        atomic_publish_directory(root, output_dir)
    return report


def export_baseline(data_root: Path, output_dir: Path):
    import pandas as pd
    from gtex_explorer.training.config import validate_training_config
    from gtex_explorer.training.core import fit_selector, fit_classifier
    from gtex_explorer.training.data import load_final_prepared
    from gtex_explorer.platform.model_registry import publish_model
    data = load_training_data(data_root)
    prepared = load_final_prepared(data)
    config = validate_training_config({'selector': 'baseline'}, candidate_count=len(prepared.gene_names))
    selector = fit_selector(prepared, config)
    result = fit_classifier(prepared, selector, config)
    demo = pd.DataFrame(np.exp2(prepared.training_log[:3]) - 1,
                        index=prepared.train_sample_ids[:3], columns=prepared.gene_names)
    return publish_model(result, config, {'classes': list(data.classes),
                         'model_id': 'original_baseline_5000',
                         'display_name': 'Baseline XGBoost - seluruh gen kandidat',
                         'data_fingerprint': data.fingerprint}, output_dir, demo, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--preprocessing', type=Path, required=True)
    parser.add_argument('--folds', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_training_data(args.cache, args.preprocessing, args.folds, args.output), indent=2))


if __name__ == '__main__':
    main()
