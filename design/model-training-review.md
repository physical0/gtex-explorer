# Training page preview

Visual: `model-training-page-20261023.png`, generated with built-in ImageGen.
Status: draft for user review, not an implemented application page.

The preview groups selector settings, XGBoost settings and optional CV on the
left, with process status and saved models on the right. Advanced parameters
remain collapsed. The visible dataset counts are the actual development cohort.
No classifier evaluation results are invented for the preview.

Implementation qualifications:

- The lower two panels illustrate running/failure states. They will not be
  permanent panels beside an idle form.
- The worker reports stages and completed folds, not live tree-by-tree progress.
  The illustrative 60%/180-round text is not an implementation requirement and
  will be replaced by actual stage/fold status and elapsed time.
- Display method-specific controls: ANOVA top-k only, MI top-k/neighbors, L1
  top-k/C plus advanced controls, baseline all 5,000 candidates.
- Changing settings does not start training. Saving a model does not silently
  replace the original prediction model.
- The whole training page, prediction model chooser, and new desktop package
  still require UI implementation after this visual is approved.

Prompt direction: restrained white/navy/teal desktop layout, concise Indonesian
copy, fixed development counts 807/494/13/5,000, L1 C=1/k=1,000 preset, XGBoost
defaults, optional five-fold evaluation, status and saved-model panels.
