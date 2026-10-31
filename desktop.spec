# Build on Windows with: python -m PyInstaller desktop.spec --noconfirm
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs, copy_metadata

root = Path(SPECPATH)
import ssl
runtime_ca = Path(ssl.__file__).with_name('codex-rejected-cas.pem')
streamlit_data, streamlit_bins, streamlit_imports = collect_all('streamlit')
datas = streamlit_data + copy_metadata('streamlit', recursive=True) + collect_data_files('xgboost')
if runtime_ca.is_file():
    datas.append((str(runtime_ca), '.'))
for folder in ['data', 'model_bundle', 'assets', '.streamlit', 'training_data', 'baseline_bundle']:
    datas.append((str(root / folder), folder))
app_modules = [
    '.'.join(path.relative_to(root).with_suffix('').parts)
    for path in (root / 'gtex_explorer').rglob('*.py')
    if path.name != '__init__.py'
]
for package in ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'scipy']:
    datas += copy_metadata(package)
for filename in ['app.py', 'desktop.py']:
    datas.append((str(root / filename), '.'))
datas.append((str(root / 'gtex_explorer'), 'gtex_explorer'))

a = Analysis(
    [str(root / 'desktop.py')], pathex=[str(root)],
    binaries=streamlit_bins + collect_dynamic_libs('xgboost'), datas=datas,
    hiddenimports=[name for name in streamlit_imports if not name.startswith('streamlit.testing')] + app_modules + ['sklearn', 'scipy', 'webview.platforms.winforms', 'webview.platforms.edgechromium'],
    excludes=['pytest', 'streamlit.testing', 'IPython', 'matplotlib'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='GTExTissueExplorer',
          debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               name='GTExTissueExplorer')
