import hashlib
import json

import pytest

from scripts.bundled_training_data import pack, restore


def dataset(root):
    root.mkdir()
    values = b'GTEx expression example\x00' * 40
    (root / 'raw_tpm.npy').write_bytes(values)
    (root / 'manifest.json').write_text(json.dumps({
        'files': {'raw_tpm.npy': hashlib.sha256(values).hexdigest()}
    }))
    return values


def test_split_bundle_restores_exact_bytes(tmp_path):
    values = dataset(tmp_path / 'data')
    pack(tmp_path / 'data', tmp_path / 'resources', chunk_size=24)
    assert len(list((tmp_path / 'resources').glob('raw_tpm.npy.gz.*'))) > 1
    restore(tmp_path / 'resources', tmp_path / 'restored')
    assert (tmp_path / 'restored/raw_tpm.npy').read_bytes() == values
    assert (tmp_path / 'restored/manifest.json').read_bytes() == (tmp_path / 'data/manifest.json').read_bytes()


def test_corrupt_chunk_cannot_replace_existing_data(tmp_path):
    dataset(tmp_path / 'data')
    pack(tmp_path / 'data', tmp_path / 'resources', chunk_size=24)
    chunk = next((tmp_path / 'resources').glob('raw_tpm.npy.gz.*'))
    chunk.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum'):
        restore(tmp_path / 'resources', tmp_path / 'restored')
    assert not (tmp_path / 'restored/raw_tpm.npy').exists()


def test_bundle_cannot_escape_output_directory(tmp_path):
    dataset(tmp_path / 'data')
    pack(tmp_path / 'data', tmp_path / 'resources')
    manifest = tmp_path / 'resources/bundle.json'
    data = json.loads(manifest.read_text())
    data['files'][0]['name'] = '../outside'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='filename'):
        restore(tmp_path / 'resources', tmp_path / 'restored')
    assert not (tmp_path / 'outside').exists()
