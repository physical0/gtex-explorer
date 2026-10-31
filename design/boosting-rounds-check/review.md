# Prediction-page round control review

Scope: the existing local Prediksi Jaringan flow. Captures were taken in the
in-app browser during this review at its default 1280 x 720 viewport. The
existing page is the visual reference, not a newly generated mockup.

## 1. Original empty state - checked

Input auditing and prediction are separate. The original disabled prediction
button looked active because the primary-button theme remained teal.

![Original empty state](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/01-before-empty.jpg>)

## 2. Original result - checked

The run/download row left room for a compact round selector. The result had no
round-count context because inference previously always used the full model.

![Original result](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/02-before-result.jpg>)

## 3. Updated empty state - passed

The native selector defaults to 300 and shares a wrapping action row with the
run button. The run button is grey and disabled until input passes its audit.
Labels and help are retained. No threshold control or new settings panel was added.

![Updated empty state](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/03-after-empty.jpg>)

## 4. Full-model result - passed

All three controls fit on one row. The result identifies its sample and round
count. Demo Thyroid probability remains 98.45% at 300 rounds, matching the
previous full-model result. Typography, colors, audit layout, and result cards
were compared with the original capture together.

![300-round result](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/04-after-300-rounds.jpg>)

## 5. Change of rounds - passed

Selecting 100 removes the old results and CSV button. The audited matrix and
sample selector remain available. The user must run prediction again.

![Old results cleared](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/05-round-change-clears-results.jpg>)

## 6. New result and export - passed

The 100-round result displays the matching context and probabilities. The CSV
downloaded through the actual browser control contains three demo samples,
their probabilities, and boosting_rounds=100 on every row. Saved experiment
results are not replaced by these interactive predictions.

![100-round result](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/06-after-100-rounds.jpg>)

## 7. Contributions - passed

The contribution section identifies the same round count. Automated checks
compare all five displayed contributions and selected-class probabilities with
native XGBoost at 100, 200, and 300 rounds.

![Matching contribution scope](<C:/Users/Univ_and_Work_files/bioinformatics thesis/gtex_tissue_explorer/design/boosting-rounds-check/07-contributions-100-rounds.jpg>)

## Verification and limits

- All 19 new cases failed before implementation and pass afterward. The final
  full suite passed 182 tests in 120.73 seconds.
- The 300-round probabilities remain exactly equal to the previous full-model
  output, and inference does not change serialized weights.
- The desktop release at dist/boosting_rounds_update/GTExTissueExplorer passed
  its packaged model/demo/localhost startup check. All 41 packaged source,
  data, and model files match current source. The 33 data/model files are
  byte-identical to the previous diagnostic-scope release.
- A focused independent code review found no blocking issues.
- No browser errors occurred after the intentional preview restart. Earlier
  disconnect/health messages coincide with that restart.
- This is a desktop-state visual check, not a full accessibility certification.
  Narrow-screen layouts, complete keyboard navigation, and screen-reader
  behavior were not exhaustively tested. The existing contribution grid's
  internal scrolling and muted helper-text sizing were retained.
