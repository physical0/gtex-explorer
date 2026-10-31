"""Read recorded panels only. Browsing never fits a selector or model."""
import json
from pathlib import Path
import tempfile

import pandas as pd

from gtex_explorer.training.data import atomic_publish_directory, file_hash, verify_files
from gtex_explorer.training.jobs import job_path


def save_fold_panel(target, identity, selector, prepared):
    with tempfile.TemporaryDirectory(prefix='panel-', dir=target.parent) as temporary:
        root = Path(temporary) / 'data'
        root.mkdir()
        selector.gene_diagnostics.iloc[selector.selected_indices].to_csv(root / 'genes.csv', index=False)
        payload = {'identity': identity, 'diagnostics': {
                    **selector.fit_diagnostics, 'training_samples': len(prepared.y_train),
                    'validation_samples': len(prepared.y_validation)},
                   'files': {'genes.csv': file_hash(root / 'genes.csv')}}
        (root / 'manifest.json').write_text(json.dumps(payload), encoding='utf-8')
        atomic_publish_directory(root, target)


def read_fold_panel(root, identity):
    payload = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if payload['identity'] != identity:
        raise ValueError('Panel identity mismatch')
    verify_files(root, payload['files'])
    return pd.read_csv(root / 'genes.csv'), payload['diagnostics']


def load_run_panel(state_root, run_id, scope):
    root = job_path(run_id, state_root)
    saved = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    if scope == 'final':
        bundle = Path(state_root) / 'models' / run_id
        manifest = json.loads((bundle / 'bundle_manifest.json').read_text(encoding='utf-8'))
        if manifest['provenance']['identity'] != saved['identity']:
            raise ValueError('Model/panel identity mismatch')
        verify_files(bundle, manifest['files'])
        return (pd.read_csv(bundle / 'selected_genes.csv'),
                json.loads((bundle / 'diagnostics.json').read_text(encoding='utf-8')))
    if type(scope) is not int or not 1 <= scope <= 10:
        raise ValueError('Invalid panel scope')
    panel = root / f'fold_{scope}_panel'
    if panel.is_dir():
        frame, info = read_fold_panel(panel, saved['identity'])
        fold = json.loads((root / f'fold_{scope}.json').read_text(encoding='utf-8'))
        from gtex_explorer.training.data import stable_hash
        if fold['identity'] != saved['identity'] or fold['checksum'] != stable_hash(fold['result']):
            raise ValueError('Fold identity/checksum mismatch')
        if frame.gene.tolist() != fold['result']['selected_genes']:
            raise ValueError('Fold panel gene axis mismatch')
        return frame, info
    # Legacy runs retain gene identities but never invent missing scores.
    fold = json.loads((root / f'fold_{scope}.json').read_text(encoding='utf-8'))
    from gtex_explorer.training.data import stable_hash
    if fold['identity'] != saved['identity'] or fold['checksum'] != stable_hash(fold['result']):
        raise ValueError('Fold identity/checksum mismatch')
    return pd.DataFrame({'gene': fold['result']['selected_genes']}), {'legacy_scores_unavailable': True}
