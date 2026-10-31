# Development verification - 24 October 2026

## Revised behavior

- Model-gene overlap is advisory, including zero matches.
- A separate nonzero-expression reminder counts finite positive TPM among all
  uploaded genes per sample. Its threshold is adjustable from 0 to 100%, default 90%.
- Model-value completeness is another adjustable reminder. Observed TPM = 0
  counts as available and never as positive. Both are calculated before imputation.
- Missing values and absent model genes use saved training medians. The app
  explicitly identifies a fully imputed model input. All-zero input also runs.
- Reminder changes leave aligned values, existing predictions and probabilities
  unchanged. Invalid numeric/structural inputs retain their rejection rules.

## Executed verification

1. Added failing regression tests before changing behavior. Existing coverage and
   all-zero gates caused the expected failures. A separate failing transformation
   test located the additional all-zero rejection before log/scaling.
2. Final full regression command: `.venv/Scripts/python.exe -m pytest tests -q`.
   Result: **267 passed in 159.00 seconds**, exit 0. Desktop/subprocess tests ran
   with a workspace TEMP/TMP directory and the required local execution permission.
3. Live localhost UI check used the documented synthetic two-gene fixture. Its
   model-ID overlap was 0% and nonzero expression 50%. Prediction completed with
   all 1,000 model values imputed. Lowering the positive-expression reminder to
   50% removed only that reminder. Screenshot saved as `independent-reminders.png`.
4. Built `dist/prediction_reminders_20261024/GTExTissueExplorer/GTExTissueExplorer.exe`
   with `build_desktop.ps1`. Source startup check, PyInstaller build, and packaged
   `--check` completed with exit 0.
5. SHA256 comparison: all **65** original resources compared with the previous
   release are unchanged. All **15** packaged Python source modules match source.
   Exact results are in `build_verification.json`.
6. Temporary localhost preview and its browser tab were stopped after checking.

This is development verification. Formal black-box acceptance testing and SUS
were not resumed. Original notebook/holdout results and all earlier distributions
were preserved. The separately planned grouped workspaces and training-gene-panel
extension are not implemented by this focused prediction-policy change.
