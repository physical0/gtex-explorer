"""Hidden local training process; no holdout access and no Streamlit imports."""
import argparse
import json
from pathlib import Path
import time
import traceback
import tempfile
import os
import threading
import psutil
import numpy as np
import pandas as pd
from gtex_explorer.training.config import validate_training_config, configuration_fingerprint
from gtex_explorer.training.core import fit_selector, fit_classifier, aggregate_cv, summarize_holdout, runtime_versions, FitResult, SelectorFit, TrainingCancelled
from gtex_explorer.training.data import load_training_data, load_final_prepared, run_split, prepare_partition, stable_hash, file_hash, verify_files, atomic_publish_directory
from gtex_explorer.training.jobs import atomic_json, job_path, process_lock, resolve_job_data
from gtex_explorer.platform.model_registry import publish_model, ModelEntry, load_saved_model
from gtex_explorer.training.gallery import finalize_gallery
from gtex_explorer.platform.workspaces import coordination_directory
from gtex_explorer.platform.cloud.state import enabled as cloud_enabled, worker_checkpoint


def _save_final_fit(result, identity, target):
    with tempfile.TemporaryDirectory(prefix='final-fit-', dir=target.parent) as temporary:
        directory = Path(temporary) / 'data'
        directory.mkdir()
        result.classifier.save_model(directory / 'model.json')
        result.selector.gene_diagnostics.to_csv(directory / 'genes.csv', index=False)
        np.save(directory / 'indices.npy', result.selector.selected_indices, allow_pickle=False)
        manifest = {'identity': identity, 'prepared': result.prepared.fingerprint,
                    'diagnostics': result.selector.fit_diagnostics,
                    'selector_seconds': result.selector.selector_seconds,
                    'classifier_seconds': result.classifier_seconds,
                    'validation_metrics': result.validation_metrics,
                    'files': {p.name: file_hash(p) for p in directory.iterdir()}}
        atomic_json(directory / 'manifest.json', manifest)
        atomic_publish_directory(directory, target)


def _load_final_fit(target, identity, prepared, config):
    from xgboost import XGBClassifier
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['identity'] != identity or manifest['prepared'] != prepared.fingerprint:
        raise ValueError('Final fit checkpoint identity mismatch')
    verify_files(target, manifest['files'])
    frame = pd.read_csv(target / 'genes.csv', float_precision='round_trip')
    indices = np.load(target / 'indices.npy', allow_pickle=False)
    if frame.gene.tolist() != list(prepared.gene_names) or indices.dtype.kind not in 'iu' or indices.shape != (config.top_k,) or len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= len(frame)).any():
        raise ValueError('Final fit checkpoint gene axis mismatch')
    classifier = XGBClassifier()
    classifier.load_model(target / 'model.json')
    if classifier.get_booster().num_features() != config.top_k or classifier.get_booster().num_boosted_rounds() != config.xgboost_parameters['n_estimators'] or classifier.n_classes_ != len(np.unique(prepared.y_train)):
        raise ValueError('Final fit checkpoint binary mismatch')
    selector = SelectorFit(np.argsort(frame.selection_rank.to_numpy()), indices, frame,
                           manifest['diagnostics'], manifest['selector_seconds'])
    return FitResult(classifier, prepared, selector, manifest.get('validation_metrics'), None, manifest['classifier_seconds'])


def run_job(run_id: str, data_root: Path, state_root: Path, *, interruptible=False) -> int:
    root = job_path(run_id, state_root)
    started = time.monotonic()
    status_lock = threading.RLock()
    monitor_finished = threading.Event()
    current = {}
    last_cloud_write = 0.
    clock_path = root / 'pause_clock.json'
    def paused_seconds():
        return json.loads(clock_path.read_text(encoding='utf-8'))['seconds'] if clock_path.exists() else 0.
    initial_paused_seconds = paused_seconds()
    def status(stage, **extra):
        nonlocal last_cloud_write
        with status_lock:
            changed_stage = stage != current.get('stage')
            current.clear()
            current.update({'status': 'running', 'stage': stage,
                            'updated': time.time(), 'elapsed_seconds': max(0, time.monotonic() - started
                                - (paused_seconds() - initial_paused_seconds)),
                            'error': None, **extra})
            atomic_json(root / 'status.json', current)
            # Complete fold/model artifacts are committed with their status.
            # Intermediate rounds are throttled to avoid one remote write per tree.
            if cloud_enabled() and (changed_stage or stage in {'completed', 'failed', 'cancelled'}
                                    or time.monotonic() - last_cloud_write > 5):
                worker_checkpoint()
                last_cloud_write = time.monotonic()
    def cancelled():
        if (root / 'cancel.json').exists():
            with status_lock:
                kept = {key: current[key] for key in ['fold', 'rounds_completed', 'rounds_total'] if key in current}
                kept['cancelled_stage'] = current.get('stage', 'queued')
                status('cancelled', status='cancelled', **kept)
            return True
        return False
    def progress_for(stage, config, fold=None):
        last_update = 0.
        total = config.xgboost_parameters['n_estimators']
        def report(completed):
            nonlocal last_update
            if (root / 'cancel.json').exists():
                raise TrainingCancelled()
            now = time.monotonic()
            if completed in (0, total) or now - last_update >= .25:
                extra = {'fold': fold} if fold is not None else {}
                status(stage, rounds_completed=completed, rounds_total=total, **extra)
                last_update = now
        return report
    def watch_cancellation():
        # This watcher is enabled only in a dedicated worker process. It can stop
        # native sklearn/MI fits that expose no cancellation callback. OS exit releases
        # the worker lock; only atomically published checkpoints are reused on resume.
        while not monitor_finished.wait(.1):
            if not (root / 'cancel.json').exists():
                continue
            with status_lock:
                if monitor_finished.is_set() or current.get('status') in {'completed', 'failed', 'cancelled'}:
                    return
                try:
                    cancelled()
                except OSError:
                    pass  # read_job recovers cancellation if a status write was interrupted.
                os._exit(0)
    coordinator = coordination_directory(state_root)
    with process_lock(coordinator / 'worker.lock'):
        atomic_json(coordinator / 'worker_owner.json', {'run_id': run_id, 'state_root': str(Path(state_root).resolve()), 'updated': time.time()})
        if interruptible:
            atomic_json(root / 'worker.json', {'run_id': run_id, 'pid': os.getpid(),
                        'created': psutil.Process().create_time(), 'pause_supported': True})
            threading.Thread(target=watch_cancellation, name='training-cancellation', daemon=True).start()
        try:
            saved = json.loads((root / 'config.json').read_text(encoding='utf-8'))
            if saved.get('data_root') and resolve_job_data(saved, state_root, data_root) != Path(data_root).resolve():
                raise ValueError('Run dataset path does not match saved data source')
            data = load_training_data(data_root)
            config = validate_training_config(saved['config'], candidate_count=data.manifest['candidate_count'])
            assignments, holdout, split = run_split(data, config)
            identity = configuration_fingerprint(config, data.fingerprint, split, runtime_versions())
            if identity != saved['identity'] or data.fingerprint != saved['data_fingerprint']:
                raise ValueError('Run identity changed. Start a new run instead of resuming.')
            published = Path(state_root) / 'models' / run_id
            if published.is_dir():
                manifest_file = published / 'bundle_manifest.json'
                manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
                load_saved_model(ModelEntry(run_id, manifest['display_name'], published, file_hash(manifest_file), published / 'demo_tpm.csv'))
                if manifest['provenance']['identity'] != identity:
                    raise ValueError('Published model does not match run identity')
                summary_path = published / 'cv_summary.json'
                summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else None
                holdout_path = published / 'holdout_summary.json'
                holdout_result = json.loads(holdout_path.read_text(encoding='utf-8')) if holdout_path.exists() else None
                status('verifying_results')
                finalize_gallery(run_id, state_root)
                status('completed', status='completed', cv_summary=summary, holdout_summary=holdout_result)
                return 0
            rows, sets = [], []
            for assignment in assignments:
                if cancelled():
                    return 0
                status('preparing_fold', fold=assignment.fold)
                prepared = prepare_partition(data, assignment, Path(state_root) / 'preprocessing_cache')
                output = root / f'fold_{assignment.fold}.json'
                if output.exists():
                    payload = json.loads(output.read_text(encoding='utf-8'))
                    if payload['identity'] != identity or payload['checksum'] != stable_hash(payload['result']):
                        raise ValueError('Completed fold identity/checksum mismatch')
                    row = payload['result']
                else:
                    status('fitting_selector', fold=assignment.fold)
                    selector = fit_selector(prepared, config, Path(state_root) / 'ranking_cache')
                    if cancelled():
                        return 0
                    status('fitting_xgboost', fold=assignment.fold)
                    result = fit_classifier(prepared, selector, config,
                                            progress_for('fitting_xgboost', config, assignment.fold))
                    row = {'fold': assignment.fold, **result.validation_metrics,
                           'tissue_f1': dict(zip(data.classes, result.validation_metrics['tissue_f1'])),
                           'training_samples': len(prepared.y_train), 'validation_samples': len(prepared.y_validation),
                           'selector_seconds': selector.selector_seconds, 'classifier_seconds': result.classifier_seconds,
                           'selected_genes': [prepared.gene_names[i] for i in selector.selected_indices]}
                    from gtex_explorer.training.panels import save_fold_panel
                    panel_path = root / f'fold_{assignment.fold}_panel'
                    if panel_path.exists():
                        # A cancelled fit may leave a complete panel before its fold result.
                        from gtex_explorer.training.panels import read_fold_panel
                        read_fold_panel(panel_path, identity)
                    else:
                        save_fold_panel(panel_path, identity, selector, prepared)
                    atomic_json(output, {'identity': identity, 'result': row, 'checksum': stable_hash(row)})
                rows.append(row)
                sets.append(set(row['selected_genes']))
            if cancelled():
                return 0
            summary = aggregate_cv(rows, sets, expected_folds=config.cv_folds,
                                   supervised_selector=config.selector != 'baseline') if assignments else None
            status('final_selector')
            # With a hold-out, the saved model learns from the development part only and is scored on the rest.
            prepared = (prepare_partition(data, holdout, Path(state_root) / 'preprocessing_cache')
                        if holdout else load_final_prepared(data))
            checkpoint = root / 'final_fit'
            if checkpoint.exists():
                result = _load_final_fit(checkpoint, identity, prepared, config)
                selector = result.selector
            else:
                selector = fit_selector(prepared, config, Path(state_root) / 'ranking_cache')
                if cancelled():
                    return 0
                status('final_xgboost')
                result = fit_classifier(prepared, selector, config, progress_for('final_xgboost', config))
                _save_final_fit(result, identity, checkpoint)
            if cancelled():
                return 0
            holdout_result = summarize_holdout(result, list(data.classes), config.holdout_percent) if holdout else None
            genes = [prepared.gene_names[i] for i in selector.selected_indices]
            demo = pd.DataFrame(np.exp2(prepared.training_log[:3, selector.selected_indices]) - 1,
                                index=prepared.train_sample_ids[:3], columns=genes)
            # Publication is atomic and deliberately not interrupted halfway through.
            with status_lock:
                if cancelled():
                    return 0
                status('saving_results')
                publish_model(result, config, {'classes': list(data.classes), 'run_id': run_id,
                              'data_fingerprint': data.fingerprint, 'identity': identity,
                              'dataset': saved.get('dataset')},
                              Path(state_root) / 'models' / run_id, demo, summary, holdout_result)
                status('verifying_results')
                finalize_gallery(run_id, state_root)
                status('completed', status='completed', cv_summary=summary, holdout_summary=holdout_result,
                       rounds_completed=config.xgboost_parameters['n_estimators'],
                       rounds_total=config.xgboost_parameters['n_estimators'])
            return 0
        except TrainingCancelled:
            cancelled()
            return 0
        except Exception as error:
            status('failed', status='failed', error={'type': type(error).__name__, 'message': str(error),
                                                   'diagnostics': getattr(error, 'diagnostics', None)})
            traceback.print_exc()
            return 1
        finally:
            monitor_finished.set()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    if cloud_enabled():
        from gtex_explorer.platform.cloud.state import state
        state().pull()
    else:
        from gtex_explorer.platform.desktop_runtime import enforce_offline_network
        enforce_offline_network()
    raise SystemExit(run_job(args.run_id, args.data, args.state, interruptible=True))
