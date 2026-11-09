# Design QA: user-data training

## Current review - lower sidebar workspace control, 30 October 2026

**final result: passed**

Approved source: `designs/workspace_switch_placement_20261030/02-lower-sidebar-context.png`,
1832 x 859 pixels. Implementation: `evidence/desktop-final.jpg` in that directory,
1514 x 710 captured pixels for a requested 1515 x 710 CSS viewport (native capture
rounding of one pixel). The source is normalized to the capture dimensions for
comparison. State: Default active, Latih Model open, reference menu collapsed.
Full and focused combined comparisons were opened and inspected:
`evidence/comparison-final.jpg` and `evidence/comparison-context.jpg`.

- Main navigation now follows the brand directly. The active-name/Ganti block is
  grouped with Referensi Eksperimen in a normal-flow lower support container.
  Margin-top:auto moves that group down; no fixed overlay is introduced.
- Resolved P2: native Markdown paragraph spacing made the folder/name row and
  action visually misaligned. Scoped spacing resets align the row and reduce the
  block from 81.6px to 72.6px. Before: `evidence/before-spacing-fix.jpg`. After:
  `evidence/desktop-final.jpg` and the final combined comparisons.
- Typography and colors retain the existing app's local sans-serif, navy/teal
  palette and native controls. The active name stays semibold, with a quieter
  caption. P3: native body text is slightly smaller than the raster mock.
- Layout, gaps and corner radius follow the selected lower-sidebar arrangement.
  Main content remains unchanged. Context plus reference menu scroll normally
  when expanded, without overlap (`evidence/references-expanded.jpg`).
- Assets use the existing DNA brand and native Material folder/swap icons.
- Content uses the real workspace name. At 1024 x 768, a 66-character fixture
  name is ellipsized within 104px and retains its full title attribute; no page
  overflow (`evidence/long-name-narrow.jpg`). Ganti still returns to the launcher.

Focused interface checks: nine passed (`qa-workspace-switch-tests.log`). Syntax
and diff checks passed. No outstanding P0/P1/P2 in verified states. Temporary
viewport override is reset before handoff.

## Current review - standalone workspace launcher, 30 October 2026

**final result: passed**

Source: `designs/workspace_launcher_20261030/01-launcher-list.png`, 1672 x 941px.
Implementation: `designs/workspace_launcher_20261030/evidence/launcher-default-final.jpg`,
1280 x 720px at the browser's normal 1x CSS viewport. Source normalized to
1280 x 720 for comparison only. State: Default selected, two workspace rows,
original model availability shown. Later approved additions are search, count,
compact rows, long-name handling and a responsive scrolling list.

Combined full-view and focused comparisons were opened together and reviewed:
`evidence/comparison-final.jpg` and `evidence/comparison-controls.jpg` under that
design directory. They compare the real implementation with the selected mock,
not screenshots viewed independently.

- Resolved P2: native full-width workspace-name buttons centered their internal
  content. Names now align to the workspace column header.
- Resolved P2: with 13 workspaces, the native 420px wrapper pushed the opening
  action below a 720px viewport. Both native wrapper and scroll area now have a
  viewport-dependent maximum height. `evidence/many-before.jpg` records the
  issue; `evidence/many-final.jpg` shows the corrected final layout. Footer bottom
  is 666.5px within the 720px viewport.
- Final narrow desktop evidence: `evidence/many-narrow-1024.jpg`, 1024 x 768,
  no horizontal overflow and footer bottom 718.4px. Temporary override reset.

Required fidelity surfaces:

- Typography: existing local sans-serif, native 14px body and smaller captions,
  clear title/brand hierarchy. Native compact labels replace the mock's larger
  row names and secondary row descriptions, as approved for a longer list.
- Spacing/layout: centered standalone surface, no sidebar, creation above the
  list, selection and opening action outside scrolling. Search adds one row.
  Column proportions were adjusted after comparison to align model policy with
  the source's general hierarchy.
- Colors: retained warm-white, navy, teal and pale selected-row tint. Native
  secondary creation button replaces the raster's teal outline. This preserves
  the current application button language.
- Assets: actual existing DNA image and native Material icons. No invented
  raster decoration or CSS-drawn replacement assets.
- Copy/content: workspace names/policies/counts come from the registry. Checkmark
  beside the name replaces the raster's trailing checkmark. Full selected name
  remains below the list. No new model or research result is invented.

No outstanding P0/P1/P2 in the verified desktop states. P3: native widget font
metrics and secondary-button styling differ slightly from the raster mock.
Functional/browser evidence and test runs are recorded in
`docs/verification/2026-10-30-workspace-launcher.md`.

The following sections retain earlier design-review history.

**Final result: pass with documented native-widget adaptations.**

Source: `design/training_user_data_20261028/configuration-review.png`.
Implementation: `../qa_user_training_20261028/evidence/configuration_desktop_1488_final.jpg`.
Viewport: 1488 x 1056 CSS pixels. Source 1487 x 1058 pixels, implementation 1488 x 1056 pixels at 1x density. Source was normalized to 1488 x 1056 for comparison only.
State: actual 100-sample upload, 100 donors, 13 classes, 74,628 input genes, Custom ANOVA, 500 selected genes, 5,000 candidate target, default 300 rounds, two-fold CV. These configuration settings do not claim a 300-round training result.

## Findings and iterations
- [Resolved P2] The first implementation had oversized metric cards and excessive divider spacing, pushing evaluation controls below the reference viewport. Evidence: `../qa_user_training_20261028/design-comparison-before.jpg`.
- Replaced metric cards with compact values/labels, increased heading hierarchy, replaced the verbose review with a compact key-value table, and corrected spacing using the actual rendered `hr` elements.
- Final same-state comparison: `../qa_user_training_20261028/design-comparison-final.jpg`. Focused controls/review comparison: `../qa_user_training_20261028/design-comparison-detail.jpg`. Both were opened and visually compared after the fixes.
- No outstanding P0/P1/P2 issue in the verified desktop state. All evaluation controls and the training action fit the reference viewport.
- Narrow desktop evidence: `../qa_user_training_20261028/evidence/configuration_narrow_1024.jpg`, 1024 x 768. Review labels/button wrap within their column, document/body width remains 1024, and vertical scrolling retains access to lower controls. Temporary viewport override was reset.

## Required fidelity surfaces
- **Typography:** retained the application's local font and existing brand. Training title and section headings now give clear hierarchy. Native widget body text remains consistent with the other pages and is smaller than the raster mock.
- **Spacing/layout:** separate Data training and Konfigurasi tabs, compact real-data strip, two-column form/review, flat selector/XGBoost/evaluation sections. Historical outputs live on their own route.
- **Colors/tokens:** existing white, navy and teal theme preserved. Native filled inputs and a bordered review table replace the mock's outlined inputs and loose table rows.
- **Assets:** reused the actual local brand image and Material icons. No raster screenshot is used as functional UI and no new decorative asset substitutes were introduced.
- **Copy/content:** Custom replaces Kustom. Counts derive from uploaded data. The Data training tab supplies the data-review/replacement route rather than duplicating the mock's Ganti data/Tinjau data controls.

## Functional acceptance
Real 100-sample upload, donor-grouped training, final/fold panels, persisted model prediction on five real samples, custom gene IDs, input rejection, and recovery were exercised through the browser. Details and evidence are in `../qa_user_training_20261028/VERIFIKASI.md`. AppTest and pure Python checks supplement these interface observations.

## Residual scope
Desktop and narrow-desktop layouts were checked. Phone-specific redesign and a complete repetition of every historical black-box case were not part of this change.

## Training workflow refinement - 29 October 2026

Reference: `designs/pelatihan_pengguna_20261029/`. Implementation evidence and
combined comparisons: `qa_training_workflow_20261029/`. Each comparison places
the approved reference beside the application at the same 1487 × 1058 viewport
and corresponding real QA configuration. The three combined images were opened
and visually inspected.

- Preserved the existing navy/teal theme, local brand, sidebar groups, native
  inputs, tables, diagnostic tabs, and training controls. Numbered native tabs
  with Next/Back implement the three-step reference without replacing familiar
  Streamlit controls. Their typography remains consistent with the existing app.
- Latih Model now places the final action with the reviewed dataset, selector,
  XGBoost rounds and CV choice. Optional controls remain available in expanders.
- Initial results spacing pushed saved-model buttons below the first viewport.
  Scoped heading/gap/divider adjustments bring both actions into view. Final
  screenshot `03-results.png` shows status, CV metrics, Jaccard and model actions.
- The initial gene table also pushed its full-panel download below the first
  viewport. A scrollable 320-pixel table retains the selected row limit and all
  records while keeping the full-panel export accessible. Numeric p-values use
  scientific notation and continue to sort numerically.
- Final and fold scopes expose actual sample counts. Method scores use their
  corresponding units. Jaccard summaries and pairwise values use percentages.
- Pause/Resume/Stop retain Material icons and actual worker behavior. Checkpoint
  recovery is separately labelled Continue from checkpoint. Verified in the
  packaged app on an isolated run.

Final desktop result: pass with documented native-widget adaptations. No new
phone-layout claim. Browser CSV-download automation timed out, so that attempted
download is not reported as successful. Data/export regression checks passed.
Detailed observations: `qa_training_workflow_20261029/VERIFIKASI.md`.

## 30 October 2026 - saved-model gallery, selected design 2

final result: passed

Implemented the searchable model list and selected-result pane, separate lazy
gallery/process tabs, and model actions before CV metrics. Actual source-app
browser checks and 39 targeted regression tests passed. Pale selected-row styling
and narrow-screen Jaccard truncation were corrected. Detailed comparison,
native-widget adaptations, evidence and distribution limits:
`designs/gallery_master_detail_20261030/design-qa.md`.

