# Independent prediction reminders

`independent-reminders.png` is a screenshot of the running application, not a generated mockup.
It shows the separate nonzero and completeness controls, set to 50% and 90%.

`no_overlap_demo.csv` is a synthetic development-check fixture with two invented
gene IDs and one sample. One value is positive and one is zero, so the uploaded
nonzero percentage is 50% while model-gene overlap is 0%. These are not real GTEx
measurements or a scientific evaluation cohort.

The running app allowed prediction and explicitly reported that the 1,000 model
values were imputed with stored training medians. The screenshot's probabilities
come from that fully imputed vector and are not model-performance metrics.

Changing the nonzero reminder to 50% removed only its reminder. Model-ID coverage
remained 0%. Source regression tests also check that reminder changes retain the
same model input and existing predictions.

See `docs/prediction-input-policy.md` for the current measurement definitions.
Formal black-box acceptance testing remains paused.
