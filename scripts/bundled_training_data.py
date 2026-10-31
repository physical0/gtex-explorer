"""Package notebook exports below GitHub's file limit and restore verified bytes."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from gtex_explorer.paths import APP_ROOT


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def filename(value):
    if not isinstance(value, str) or not value or Path(value).name != value or any(
        character in value for character in ('/', '\\', ':')
    ) or value in {'.', '..'}:
        raise ValueError('Invalid bundle filename')
    return value


def pack(source, destination, chunk_size=64 * 1024 * 1024):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    for name in sorted([*manifest['files'], 'manifest.json']):
        path = source / filename(name)
        checksum = digest(path)
        if name != 'manifest.json' and checksum != manifest['files'][name]:
            raise ValueError('Notebook source checksum mismatch: ' + name)
        chunks = []
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            compressed = Path(temporary) / 'file.gz'
            with path.open('rb') as input_file, gzip.GzipFile(
                filename=str(compressed), mode='wb', mtime=0
            ) as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            with compressed.open('rb') as input_file:
                for index, block in enumerate(iter(lambda: input_file.read(chunk_size), b'')):
                    part = name + f'.gz.{index:03d}'
                    (destination / part).write_bytes(block)
                    chunks.append({'name': part, 'sha256': hashlib.sha256(block).hexdigest()})
        entries.append({'name': name, 'sha256': checksum, 'chunks': chunks})
    (destination / 'bundle.json').write_text(
        json.dumps({'schema_version': 1, 'files': entries}, indent=2) + '\n', encoding='utf-8'
    )


def restore(source, destination):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source / 'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported training bundle')
    # Validate names and every compressed checksum before creating any output.
    names = [filename(entry['name']) for entry in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate bundle filename')
    for entry in manifest['files']:
        for part in entry['chunks']:
            if digest(source / filename(part['name'])) != part['sha256']:
                raise ValueError('Training bundle checksum mismatch')
    destination.mkdir(parents=True, exist_ok=True)
    for entry in manifest['files']:
        output = destination / entry['name']
        if output.is_file():
            if digest(output) == entry['sha256']:
                continue
            raise FileExistsError('Refusing to replace different data: ' + output.name)
        with tempfile.TemporaryDirectory(dir=destination) as temporary:
            compressed, restored = Path(temporary) / 'file.gz', Path(temporary) / 'restored'
            with compressed.open('wb') as stream:
                for part in entry['chunks']:
                    with (source / part['name']).open('rb') as chunk:
                        shutil.copyfileobj(chunk, stream)
            with gzip.open(compressed, 'rb') as input_file, restored.open('wb') as output_file:
                shutil.copyfileobj(input_file, output_file, length=1024 * 1024)
            if digest(restored) != entry['sha256']:
                raise ValueError('Restored training data checksum mismatch')
            restored.replace(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['pack', 'restore'])
    args = parser.parse_args()
    packaged = APP_ROOT / 'resources/notebook_training'
    training = APP_ROOT / 'training_data'
    if args.action == 'pack':
        pack(training, packaged)
    else:
        restore(packaged, training)
    print('Notebook training data ' + args.action + ' complete.')
