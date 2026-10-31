# Prediction layout verification

final result: passed

Verified on 25 October 2026 against
`design/error_states_20261024/input-coverage-advisory-v2.png`.

## Evidence and states

- Source and final desktop capture both measure 1487 × 1058 pixels at a matching
  browser CSS viewport. `04-final-comparison.png` places them together at unchanged
  scale. `04-final-detail.png` compares the audit, reminders and informational notice.
- `05-final-normal.png` checks the normal 1280 × 720 desktop viewport. The three
  audit values remain visible without horizontal scrolling. Taller content uses
  ordinary vertical scrolling. This is not a mobile-window acceptance claim.
- The controlled CSV has one sample and two invented gene IDs, TPM 5 and 0. Its
  model-ID overlap and observed model completeness are both 0%, while positive
  expression is independently 50%. Reminder values are 50% and 90%. Prediction
  remains enabled and imputation is explicitly reported.
- `03-final-desktop.png` used the actual upload interaction. Browser file-chooser
  calls were intermittently extremely delayed, so the final spacing comparison
  used `.superpowers/prediction-layout/preview.py`. That development harness changes
  only the local demo file path and runs the same application, model, audit and
  prediction code. Its final capture shows an empty upload picker because the
  fixture was opened through Gunakan contoh lokal. It does not pretend that an
  upload is present. The harness and synthetic fixture are not production datasets.

## Findings and corrections

1. P1: a threshold set before the first audit reverted to 90% after widget cleanup.
   Reproduced in a failing AppTest and fixed with durable reminder preferences.
2. P2: the original large dataframe summary hid its values behind horizontal
   scrolling at normal width. Replaced with a wrapping semantic three-row table.
3. P2: sidebar groups inherited boxed borders and left-side chevrons. Actual details
   elements now use plain divider styling and correctly placed disclosure arrows.
4. P2: style-only layout gaps, excessive sidebar header height, repeated metadata and
   the summary header pushed the information notice below the reference frame.
   The final capture contains the complete summary, controls and notice. Detailed
   structural checks remain accessible below them.
5. P2: the empty-state file icon and copy were left aligned. They now use a centered
   native Material icon and aligned text, with one separator instead of two.

## Required fidelity surfaces

- Typography: retained the existing application's Segoe UI/native Streamlit scale.
  This is slightly smaller than the image-generated reference, but labels, values,
  helper text and errors are readable and do not collide. Remaining scale difference
  is P3 polish, not missing content or a claim of pixel-identical typography.
- Spacing: two comparable columns, narrow model chooser, simple sidebar groups,
  three-row summary, controls below it, concise notice and centered ready state.
- Colors: existing white/off-white, navy, teal and quiet blue information styling.
  There is no red coverage rejection or disabled prediction for advisory thresholds.
- Assets: existing DNA brand image and actual Material library icons are preserved.
  No generated mock is embedded as application UI and no custom diagram art replaces controls.
- Copy: independent measurement meanings, selected-sample counts, finite positive
  TPM denominator, zero-as-observed behavior and stored-training-median imputation
  are stated. Repeated success/coverage notices are removed. Reminder ranges remain
  0–100%. Invalid numerical/structural inputs retain prominent rejection messages.

The grouped sidebar exposes existing pages only. The separate training-gene-panel
extension remains under its own pending plan, without nonfunctional placeholder links.
Formal black-box acceptance and SUS testing were not resumed.
