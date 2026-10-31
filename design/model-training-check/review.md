# Training implementation verification

Approved design: `../model-training-page-20261023.png`.
Live desktop-layout capture: `training-desktop.png`.

Verified flows:

- Selector controls, baseline fixed candidate count, optional CV and disabled
  fold selector in training-only mode. Invalid/cleared inputs disable fitting.
- Original/baseline/custom model switching, invalidation of old audits and
  predictions, model-specific examples and boosting-round controls.
- Actual UI training run: ANOVA, 500 genes, 3 rounds. Completed, saved a separate
  model, audited its 500/500-gene example, and produced prediction results.
- Source worker cancellation/resume, final-fit checkpoint reuse, reconciliation
  after publication-status failure, and malformed-model isolation.
- Raw development data has zero missing values. Avoiding unnecessary full-width
  imputation reduced cold preparation of the first real 3-fold partition to
  2.25 seconds on this machine. No timing guarantee is implied for other runs.
- Independently recomputed all five original partitions: candidate gene order,
  training matrices and validation matrices match the notebook cache exactly.
- Actual 3-fold ANOVA/500/2-round run completed with all three fold results,
  three Jaccard pairs and final development refit.
- Actual L1 C=1 default refit: converged in 609 iterations, 1,623 nonzero genes,
  exact notebook ordering for all 1,000 selected genes.
- Final source regression: 241 passed in 176.56 seconds. Includes numeric-input,
  model-identity, worker recovery, artifact integrity and desktop check tests.
- Packaged MI exposed loky's incompatible frozen Windows child-process entry
  point. MI now uses thread-based parallel scoring in both source and desktop.
  Regression reproduces unavailable process workers and verifies exact scores.
  Actual development data: all 5,000 scores exactly match seeded single-worker
  scoring. Rebuilt desktop startup and ANOVA worker checks passed again.
- Final packaged MI run completed in 39.906 seconds: 500 selected genes,
  2 boosting rounds, 13 classes, separate model save/load, successful demo
  audit and prediction for three samples with normalized class probabilities.
  Run ID: run_308b4f3d578841c89ca8cd70b45c0707. Timing is a smoke-test
  observation, not an estimate for the default 300-round configuration.

Fresh reviewer found three Important issues. Each was reproduced and fixed:
completed final-fit preservation, published-model completion recovery and
malformed custom manifest isolation. No Critical issues or deferred Minors.

The UI uses actual stage/fold progress, not the illustrative per-tree progress
from the mockup. The original experiment pages still report only the original
reference run. New training outputs are kept in the training page and state
directory. No new holdout evaluation is run by training.

Packaging note: use the final distribution named `model_training_final_20261023`.
Intermediate `model_training_update_20261023` was not delivered after a QA process
held one of its DLLs during rebuilding. Earlier pre-task distributions remain
unchanged. The final distribution was created in a fresh directory.

Final package accepted: startup check and separate hidden-worker check both
returned exit 0. All 15 packaged source modules and 59 data/model/training
resources match their source hashes. All 33 original data/model artifacts
are unchanged. Use `GTExTissueExplorer.exe` from the final distribution, keeping
its `_internal` directory beside it.
