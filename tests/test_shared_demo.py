"""The common example preserves measured values and does not depend on the model."""
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_shared_example_is_union_without_fabricated_measurements():
    shared = pd.read_csv(ROOT / 'data/demo_tpm_shared.csv', index_col=0)
    assert shared.shape == (6, 5000)
    assert shared.index.is_unique and shared.columns.is_unique
    assert not np.isinf(shared.to_numpy()).any()
    for filename in ['data/demo_tpm.csv', 'baseline_bundle/demo_tpm.csv']:
        original = pd.read_csv(ROOT / filename, index_col=0)
        np.testing.assert_allclose(shared.loc[original.index, original.columns], original)
    original = pd.read_csv(ROOT / 'data/demo_tpm.csv', index_col=0)
    extra_genes = shared.columns.difference(original.columns)
    assert shared.loc[original.index, extra_genes].isna().all().all()
    provenance = json.loads((ROOT / 'data/demo_tpm_shared_provenance.json').read_text())
    assert len(provenance['sources']) == 5


def test_loader_ignores_model_specific_demo_path(monkeypatch):
    import gtex_explorer.ui.pages.prediction as prediction_page
    seen = []
    monkeypatch.setattr(prediction_page, '_audit_matrix', lambda **kwargs: seen.append(kwargs))
    for genes in [['ENSG00000170549.4'], ['ENSG_OTHER']]:
        prediction_page._load_local_demo({'selected_genes': genes, 'demo_path': '/does/not/exist.csv'})
    pd.testing.assert_frame_equal(seen[0]['matrix'], seen[1]['matrix'])
    assert seen[0]['filename'] == seen[1]['filename'] == 'demo_tpm_shared.csv'
