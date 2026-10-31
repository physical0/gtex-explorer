# Error presentation previews

Three complementary states of the already selected collapsible-workspace design, not three replacement choices for the workspace layout. Generated independently with the built-in image-generation tool using the selected visual and actual prediction/training screenshots as references. No application code was edited and formal black-box testing remains paused.

Displayed order:

1. `input-coverage-advisory-v2.png`: current prediction-page design, incorporating independent gene-ID, nonzero-expression, and completeness measurements. Prediction remains enabled at zero matching model genes. This replaces `input-coverage.png`, preserved only as the superseded hard-coverage draft. See `docs/prediction-input-policy.md`.
2. `training-l1-support.png`: illustrative L1 support failure with known notebook diagnostic counts, distinct from nonconvergence, configuration correction creating a new run.
3. `legacy-panel-details.png`: informational missing-score state for the existing ANOVA test run, saved gene list remains usable, no fabricated per-gene values.

The previews propose inline notices with expandable technical details, not dismissible-only toasts or repeated blocking modals. They are not evidence of errors occurring in a newly implemented app. The actual stored ANOVA test run supplies the third view's genes, 500-gene count, three folds, two boosting rounds, and fold-1 549/258 sample counts.

Implementation must use actual current data rather than copy static text from an image. The obsolete coverage image inherited an incorrect 71.9 KB label. The revised prediction image instead uses the documented synthetic `no_overlap_demo.csv` fixture: 62 B, one sample, two uploaded genes with TPM values 5 and 0, zero model-ID matches, and zero observed model values. Fold rows not reached after a failure must say Belum dijalankan, without implying automatic queued continuation. Accordion chevrons must accurately reflect whether their child links are displayed. The existing selected option 2 controls the grouped workspace layout, with the active group expanded as appropriate.

Preview footers and illustrative example names are not normal product copy. Error codes/reasons/actions must be driven by actual outcomes and recorded context. Neither clinical disclaimers nor notebook-derived values are inserted into a custom run that did not compute them.

## Revised prediction design

The current image retains the selected grouped sidebar and two-column prediction layout.
Both sidebar groups are collapsed with right-pointing chevrons because the shared
Prediksi Jaringan page is active. The old red coverage error is replaced by a quiet
informational notice. No coverage rejection code or minimum matching percentage remains.

The audit summary separates model-ID matching, model-value completeness, and positive
expression among all uploaded genes per sample. An expanded Pengaturan pengingat
section contains two independent 0–100% controls. Both default to 90% in the app,
while the preview deliberately shows nonzero at 50% and completeness at 90% to
demonstrate independence. Neither disables prediction. TPM = 0 counts as available
for completeness and does not count as positive expression. Missing model values use
stored training medians. The not-yet-run right panel has an enabled prediction button
and no invented probability results.

Generated with the built-in Image Generation tool using `input-coverage.png` as the
edit target, `collapsible-inspector.png` as the selected grouped-navigation reference,
and the actual `prediction_reminders_20261024/independent-reminders.png` screenshot as
the behavior/control reference. The final prompt requested: preserve the existing
white/navy/teal academic design, replace hard coverage rejection with information,
show the three independent measurements and two adjustable reminders, and keep the
prediction action enabled at zero overlap. This is a design preview, not a screenshot
of implemented grouped navigation. No app source or executable changed in this revision.
