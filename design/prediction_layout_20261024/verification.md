# Prediction layout release verification - 25 October 2026

Implemented the approved compact prediction design and grouped existing navigation.
Original routes and model algorithms are retained. The separate training-panel
extension remains under its own plan, with no nonfunctional placeholder destinations.

## Tests

Fresh final command: `.venv/Scripts/python.exe -m pytest tests -q --junitxml=.prediction-layout-final-tests.xml`.
Result: **275 passed in 403.58s**, zero failures/errors, exit0. This includes independent
selected-sample audit measures, zeros/missing values, invalid-input rejection,
pre-audit and navigation threshold retention, nondefault model retention, prediction
round consistency, original pages, model integrity, local training and worker resume.

Earlier findings were reproduced before fixes. One earlier full run observed a
training-worker startup timing state, whose saved record later completed; isolated
and fresh final runs passed. Training job code was not changed. Formal black-box
acceptance testing and SUS were not resumed.

## Review and visual evidence

One fresh read-only review found no confirmed Critical/Important regression. Its
default-only model-selection test caveat led to a stronger nondefault-model test
and remembered model-selection fix. See `design-qa.md`, `04-final-comparison.png`,
`04-final-detail.png`, and `05-final-normal.png` for actual visual checks and limitations.
The controlled fixture is synthetic, not scientific performance evidence.

## Desktop release

`dist/prediction_layout_20261025/GTExTissueExplorer/GTExTissueExplorer.exe`

`build_desktop.ps1` source startup check, PyInstaller packaging and packaged `--check`
all completed with exit0. Keep the complete release directory, including `_internal`.
All15 packaged Python source modules match source. All65 compared original
model/data/assets/configuration resources match the preceding release byte for byte.
The model SHA256 is unchanged. Exact results: `build-verification.json`.
The development fixture loader is not packaged. Prior releases were preserved.

Both temporary preview servers and the browser tab were stopped. The temporary
viewport override was reset. Current source and the new desktop release contain
the changes. No notebook, thesis document, CV fit, or holdout score was modified.
