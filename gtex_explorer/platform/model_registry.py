"""Validated, non-pickle local model bundles."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
from gtex_explorer.training.config import TrainingConfig
from gtex_explorer.training.core import FitResult, runtime_versions
from gtex_explorer.training.data import cached_file_hash, file_hash, stable_hash, unique_ids, verify_files, atomic_publish_directory


@dataclass(frozen=True)
class ModelEntry:
    model_id: str
    display_name: str
    bundle_dir: Path
    manifest_hash: str
    demo_path: Path


def _entry(root: Path) -> ModelEntry:
    path = root / 'bundle_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or manifest.get('status') != 'validated':
        raise ValueError('Unsupported or incomplete saved model')
    model_id = manifest['model_id']
    if not isinstance(model_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', model_id):
        raise ValueError('Invalid model ID')
    if not isinstance(manifest.get('display_name'), str) or not isinstance(manifest.get('files'), dict):
        raise ValueError('Invalid model manifest fields')
    if any(not isinstance(name, str) or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
           for name, digest in manifest['files'].items()):
        raise ValueError('Invalid model file hash mapping')
    if not isinstance(manifest.get('versions'), dict) or not isinstance(manifest['versions'].get('xgboost'), str):
        raise ValueError('Invalid model versions')
    for key in ['feature_count', 'class_count', 'boosting_rounds']:
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError('Invalid numeric model manifest fields')
    display_name = manifest['display_name']
    if not model_id.startswith('original_'):
        display_name += f" - {manifest['boosting_rounds']} putaran - {model_id[-8:]}"
    return ModelEntry(model_id, display_name, root, file_hash(path), root / 'demo_tpm.csv')


def load_saved_model(entry: ModelEntry) -> tuple[dict, XGBClassifier]:
    root = entry.bundle_dir
    _entry(root)  # Validate the shape of the untrusted manifest before indexing it.
    if file_hash(root / 'bundle_manifest.json') != entry.manifest_hash:
        raise ValueError('Model manifest hash changed')
    manifest = json.loads((root / 'bundle_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or manifest.get('status') != 'validated' or manifest.get('model_id') != entry.model_id:
        raise ValueError('Invalid saved model identity/schema')
    verify_files(root, manifest['files'])
    required = {'preprocessor.json', 'preprocessing.npz', 'xgboost_model.json', 'demo_tpm.csv', 'selected_genes.csv', 'diagnostics.json'}
    if not required.issubset(manifest['files']):
        raise ValueError('Incomplete saved model files')
    if manifest['versions']['xgboost'].split('.')[0] != runtime_versions()['xgboost'].split('.')[0]:
        raise ValueError('Incompatible XGBoost model version')
    pre = json.loads((root / 'preprocessor.json').read_text(encoding='utf-8'))
    if not isinstance(pre, dict) or not isinstance(pre.get('selected_genes'), list) or not isinstance(pre.get('classes'), list):
        raise ValueError('Invalid preprocessing metadata')
    if any(not isinstance(value, str) for value in pre['selected_genes'] + pre['classes']):
        raise ValueError('Invalid model identifier axes')
    genes = unique_ids(pre['selected_genes'], genes=True)
    classes = unique_ids(pre['classes'])
    if not 1 <= len(genes) <= 5000 or len(genes) != pre['feature_count'] or len(genes) != manifest['feature_count']:
        raise ValueError('Model gene count mismatch')
    if len(classes) != manifest['class_count'] or pre['model_id'] != entry.model_id:
        raise ValueError('Model identity/class mismatch')
    with np.load(root / 'preprocessing.npz', allow_pickle=False) as arrays:
        if set(arrays.files) != {'raw_tpm_medians', 'scaler_mean', 'scaler_scale'}:
            raise ValueError('Invalid preprocessing arrays')
        for key in arrays.files:
            values = arrays[key]
            if values.dtype.kind not in 'fi' or values.shape != (len(genes),) or not np.isfinite(values).all():
                raise ValueError('Invalid preprocessing constants')
            pre[key] = values.copy()
    if (pre['raw_tpm_medians'] < 0).any() or (pre['scaler_scale'] <= 0).any():
        raise ValueError('Invalid TPM medians/scaling')
    classifier = XGBClassifier()
    classifier.load_model(root / 'xgboost_model.json')
    booster = classifier.get_booster()
    if booster.num_features() != len(genes) or classifier.n_classes_ != len(classes) or booster.num_boosted_rounds() != manifest['boosting_rounds']:
        raise ValueError('Model binary does not match manifest')
    return pre, classifier


def publish_model(result: FitResult, config: TrainingConfig, provenance: dict,
                  output_dir: Path, demo_tpm: pd.DataFrame, cv_summary: dict | None,
                  holdout_summary: dict | None = None) -> ModelEntry:
    from gtex_explorer.inference.audit import audit_and_align_input
    from gtex_explorer.inference.prediction import predict_tissues
    prepared = result.prepared
    if prepared.X_validation is not None and not holdout_summary:
        raise ValueError('A fold model cannot be published')
    if bool(config.holdout_percent) != bool(holdout_summary):
        raise ValueError('Hold-out score does not match the configuration')
    if config.evaluate_cv and (not cv_summary or cv_summary.get('fold_count') != config.cv_folds):
        raise ValueError('Complete CV required')
    if not config.evaluate_cv and cv_summary is not None:
        raise ValueError('Training-only cannot claim CV scores')
    classes = unique_ids(provenance['classes'])
    if len(classes) != result.classifier.n_classes_:
        raise ValueError('Class mapping mismatch')
    indices = result.selector.selected_indices
    genes = unique_ids([prepared.gene_names[i] for i in indices], genes=True)
    if len(genes) != config.top_k:
        raise ValueError('Selected gene count mismatch')
    if config.selector == 'l1' and (not result.selector.fit_diagnostics.get('converged') or result.selector.fit_diagnostics['nonzero_gene_count'] < len(genes)):
        raise ValueError('Ineligible L1 fit')
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError('Refusing to overwrite model')
    model_id = provenance.get('model_id', output_dir.name)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', model_id):
        raise ValueError('Invalid model ID')
    pre = {'bundle_schema_version': 2, 'model_id': model_id, 'run_id': provenance.get('run_id', ''),
           'selector_id': config.selector, 'feature_selection': config.selector,
           'l1_c': config.l1_c if config.selector == 'l1' else None, 'feature_count': len(genes),
           'selected_genes': list(genes), 'classes': list(classes),
           'xgboost_parameters': dict(config.xgboost_parameters)}
    constants = {}
    for key in ['raw_tpm_medians', 'scaler_mean', 'scaler_scale']:
        value = getattr(prepared, key)
        if value is None:
            raise ValueError('Final preprocessing constants unavailable')
        constants[key] = np.asarray(value)[indices]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='publishing-', dir=output_dir.parent) as temporary:
        root = Path(temporary) / 'bundle'
        root.mkdir()
        result.classifier.save_model(root / 'xgboost_model.json')
        np.savez(root / 'preprocessing.npz', **constants)
        (root / 'preprocessor.json').write_text(json.dumps(pre, allow_nan=False), encoding='utf-8')
        result.selector.gene_diagnostics.iloc[indices].to_csv(root / 'selected_genes.csv', index=False)
        demo_tpm.to_csv(root / 'demo_tpm.csv', index_label='SAMPID')
        (root / 'diagnostics.json').write_text(json.dumps(result.selector.fit_diagnostics, allow_nan=False), encoding='utf-8')
        if cv_summary:
            (root / 'cv_summary.json').write_text(json.dumps(cv_summary, allow_nan=False), encoding='utf-8')
        if holdout_summary:
            (root / 'holdout_summary.json').write_text(json.dumps(holdout_summary, allow_nan=False), encoding='utf-8')
        manifest = {'schema_version': 1, 'status': 'validated', 'model_id': model_id,
                    'display_name': provenance.get('display_name', f'{config.selector} - {len(genes):,} gen'),
                    'provenance': provenance, 'config': config.to_dict(), 'versions': runtime_versions(),
                    'class_count': len(classes), 'feature_count': len(genes),
                    'boosting_rounds': config.xgboost_parameters['n_estimators'],
                    'demo_partition': 'development_training_donors',
                    'files': {p.name: file_hash(p) for p in root.iterdir()}}
        (root / 'bundle_manifest.json').write_text(json.dumps(manifest, allow_nan=False), encoding='utf-8')
        pending = _entry(root)
        loaded, classifier = load_saved_model(pending)
        aligned, audit = audit_and_align_input(pd.read_csv(pending.demo_path, index_col=0), loaded)
        if not audit.ready:
            raise ValueError(f'Invalid model example: {audit.message}')
        _, probabilities = predict_tissues(aligned, loaded, classifier)
        if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6):
            raise ValueError('Invalid model probabilities')
        atomic_publish_directory(root, output_dir)
    return _entry(output_dir)


_validated_bundles = {}


def _validate_once(entry: ModelEntry) -> None:
    """Fully load a bundle only when its files change. The loaded model is not kept."""
    root = entry.bundle_dir
    signature = (entry.manifest_hash, tuple(sorted(
        (path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in root.iterdir() if path.is_file())))
    if _validated_bundles.get(str(root)) != signature:
        load_saved_model(entry)
        _validated_bundles[str(root)] = signature


def list_models(resource_root: Path, state_root: Path, *, issues: list | None = None) -> list[ModelEntry]:
    from gtex_explorer.platform.workspaces import bundled_models_enabled
    include_originals = bundled_models_enabled(state_root)
    entries = []
    original = Path(resource_root) / 'model_bundle'
    names = ['bundle_manifest.json', 'preprocessor.joblib', 'xgboost_model.json']
    if include_originals and original.is_dir():
        try:
            if not all((original / name).is_file() for name in names):
                raise ValueError('Berkas model asli belum lengkap. Pulihkan folder model_bundle dari paket aplikasi.')
            meta = json.loads((original / names[0]).read_text(encoding='utf-8'))
            if not isinstance(meta, dict) or meta.get('status') != 'validated':
                raise ValueError('Manifest model asli tidak valid. Pulihkan folder model_bundle dari paket aplikasi.')
            if cached_file_hash(original / names[2]) != meta.get('xgboost_model_sha256'):
                raise ValueError('Original model hash mismatch')
            entries.append(ModelEntry('original_l1_c_1_1000', 'Original - L1, C = 1 - 1.000 gen', original,
                                      stable_hash({name: cached_file_hash(original / name) for name in names}),
                                      Path(resource_root) / 'data' / 'demo_tpm.csv'))
        except (ValueError, KeyError, OSError, TypeError) as error:
            if issues is not None:
                issues.append({'model': 'Model asli', 'reason': str(error)})
    paths = [Path(resource_root) / 'baseline_bundle'] if include_originals else []
    model_root = Path(state_root) / 'models'
    if model_root.is_dir():
        from gtex_explorer.training.gallery import list_gallery
        gallery_issues = []
        ready = {entry['run_id'] for entry in list_gallery(state_root, issues=gallery_issues)}
        if issues is not None:
            issues.extend({'model': issue['run_id'], 'reason': issue['reason']} for issue in gallery_issues)
        paths.extend(sorted(p for p in model_root.iterdir() if p.is_dir()
                            and (not re.fullmatch(r'run_[0-9a-f]{32}', p.name) or p.name in ready)))
    for path in paths:
        if not (path / 'bundle_manifest.json').is_file():
            continue
        try:
            entry = _entry(path)
            _validate_once(entry)
            entries.append(entry)
        except (ValueError, KeyError, OSError, TypeError, XGBoostError) as error:
            if issues is not None:
                issues.append({'model': path.name, 'reason': str(error)})
            continue
    if len({entry.model_id for entry in entries}) != len(entries):
        raise ValueError('Duplicate model identity')
    return entries


def load_prediction_model(entry: ModelEntry) -> tuple[dict, XGBClassifier]:
    if entry.model_id != 'original_l1_c_1_1000':
        return load_saved_model(entry)
    from gtex_explorer.inference.model_bundle import MODEL_DIR, load_model_bundle
    if entry.bundle_dir.resolve() != MODEL_DIR.resolve():
        raise ValueError('Original model must use bundled resources')
    pre, classifier = load_model_bundle()
    return {**pre, 'model_id': entry.model_id, 'run_id': ''}, classifier
