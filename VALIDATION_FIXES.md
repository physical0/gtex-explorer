# Validation of the six app fixes

Completed 1 October 2026. Ready within the six repaired workflows.

## Changes

1. Input parsing rejects nonnumeric expression tokens. QC requires matching gene
   identifiers and at least 90% finite observed model-gene values per sample before
   imputation, with at least one positive TPM value. Missing columns and missing
   values share the 10% imputation allowance. A single failed sample blocks the
   batch. The audit displays per-sample coverage and imputation counts. Rejected
   aligned data is blocked again before reaching the classifier.
2. Pending uploads are identified by SHA256 content plus filename. Replacement,
   removal, and switching from the demo to a new upload invalidate old audit data,
   probabilities, predictions, sample selection, and download controls. Repeated
   audits use a new in-memory stream. Audit/prediction errors do not preserve old
   prediction results.
3. A dedicated app `.venv` was created using the functioning local Python and
   existing installed dependencies, without a network package download. Neither
   old venv was overwritten. PowerShell validates interpreter imports and XGBoost
   version, skips broken candidates, and supports `-CheckOnly`, `-Headless`, and
   `-Port`. BAT delegates to the same validated launcher.
4. The app's final-test CSV now matches the notebook's `part_a_only` export,
   containing only the selected XGBoost model. Original older results in the
   notebook result directories were preserved. The loader rejects mixed/stale
   final configurations rather than hiding them. CSV/model cache keys include
   file modification times and sizes so refreshed exports are loaded on rerun.
5. Gene search now uses literal matching, not regex interpretation.
6. The CV chart has a shared, opaque, high-contrast selector legend. Method colors
   match between lines, uncertainty bands, and legend. Baseline and eligibility
   dash meanings remain explained in the caption.

## Executed checks

Command from the thesis workspace:

```powershell
.\gtex_tissue_explorer\.venv\Scripts\python.exe -m pytest gtex_tissue_explorer/tests -q
```

Result: **42 passed**. This includes parsing/QC edge cases, mixed-quality batches,
fractional median imputation into nullable integer columns, rejected-input guards,
upload replacement/removal/demo transitions, repeated audit, five literal search
queries, stale-export rejection, current overview values, five metric controls,
legend specifications, model integrity, and demo prediction.

Independent exported-result checks confirm:

- 16 configurations, five folds each, 80 unique configuration/fold results.
- Five 5,000-gene baseline rows with no supervised feature selection.
- One exported fold identity per fold and one XGBoost parameter hash across methods.
- CV means and sample standard deviations match for all five evaluation metrics.
- Both holdout models' five metrics match recalculation from saved predictions.
- Holdout contains 193 unique samples from 124 donors.
- Overview metrics match the selected-model holdout result and the Part A-only CSV.
- Existing XGBoost model SHA256 still matches the validated manifest.

`pip check` reports **No broken requirements found**. Syntax compilation passed.
PowerShell and BAT passed `-CheckOnly`. A deliberately broken `THESIS_PYTHON`
candidate was skipped in favor of the dedicated app environment. Both launchers
were tested with actual server starts and successful localhost health responses.
The temporary BAT server on port 8502 was stopped after verification. The repaired
app preview is running on port 8501.

Rendered browser checks verified the legend at 1440px and 900px widths, the
corrected overview, and a literal `[` search yielding an empty table instead of
an exception. Screenshots are in `.codex_tmp/app-fixes` in the thesis workspace.
The preview was restarted after source edits because automatic file watching is
disabled in the offline configuration.

## Scope and remaining limits

The trained model, its frozen transformations, CV splits, and experiment metrics
were not changed or retrained. This fixes input completeness and obvious invalid
samples, not all out-of-distribution inputs. The 90% threshold is an operational
QC policy, not a GTEx-derived biological validity threshold. Model probabilities
remain uncalibrated outputs and are not guarantees of correctness.

The app environment still requires its base Python installation, so the app is
not a self-contained executable. Anatomy coverage and large-file stress testing
were outside these six fixes. Saved fold identity hashes were checked, but donor
assignments and the 80 model fits were not independently rebuilt.
