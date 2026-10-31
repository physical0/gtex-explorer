# Kamus Biologi visual verification

Date: 20 October 2026.

Source: `../biology-dictionary-target-20261020.png`, the second displayed mockup.
Source pixels and temporary browser CSS viewport: 1487 x 1058, DPR approximately 1.
State: Kamus Biologi, all categories, Ekspresi gen open, navigation expanded.

## Initial comparison

Evidence: `01-initial-comparison.png` and `01-initial-detail.png`.

- P2: Index labels were centered, the active row used a solid teal fill, and
  default vertical gaps clipped the active row at the bottom of the index.
  Fix: left-align the inner button content, override the inherited primary
  button color, and remove index-container gaps.
- P2: The main title and term text were too small, and an invisible style
  element added unwanted space above the heading.
  Fix: page-scoped heading/body sizes and hidden style-element containers.

## Final comparison

Evidence: `02-final-comparison.png` and `02-final-detail.png`, source on the
left and the actual browser capture on the right. Both are 1487 x 1058 pixels,
without stretching. `02-final.jpg` is the unmodified browser capture.

The initial P2 findings are resolved: the index is left aligned, the selected
term is fully visible with a pale teal highlight, and the invisible style
element no longer creates space above the page heading.

Required fidelity surfaces:

- Typography: existing offline application sans-serif family, 16px reading
  text and index labels, 38.4px page/term headings. Native captions remain
  subordinate. The generated mock has slightly larger secondary typography,
  but native controls preserve consistency with the other application pages.
- Layout: search/category controls above the term index and reading panel,
  fine vertical separator, related links and reference/download controls below
  the definition. Long indexes scroll independently; the page scrolls vertically
  at normal 1280 x 720 desktop size (`05-default-width.jpg`).
- Colors: white background, navy text, muted captions and teal selection/link
  colors. No decorative cards or added illustrations.
- Assets: the existing bundled DNA brand image is reused unchanged. Material
  icons and native Streamlit controls remain, rather than rasterizing the UI.
- Content: 28 real dictionary entries instead of the mock's abbreviated index.
  All 13 GTEx tissue labels have Indonesian names and searchable English labels.
  Only molecular biology, tissue/anatomy and expression-measurement entries
  remain. Statistics, machine learning, and software definitions are removed.

## Interaction checks

- English `Thyroid` search opens `Tiroid` (`03-tissue-search.jpg`).
- Related `Hipofisis` opens that definition and clears the restrictive query.
- The tissue category excludes molecular entries and retains the open tissue.
- The entry reference expands and shows its source address
  (`04-category-and-reference.jpg`). These addresses are displayed as text.
- An unmatched query removes the old definition and displays a real empty state.
- Clearing the query restores the index. Related entries and index entries
  use keyboard-accessible native buttons; search/filter retain accessible labels.
- The CSV payload is checked by the automated test: all filtered entries, not
  only the selected definition, with human-readable reference titles/addresses.
  Browser download persistence was not separately exercised.
- Browser logs contain only errors at 09:01:02-09:01:05 UTC from the intentional
  preview restart. No later browser errors were observed during the dictionary
  interaction checks. The temporary viewport override was reset after comparison.

No actionable P0/P1/P2 findings remain. P3 differences are the application's
existing sidebar spacing, native control typography, and native reference
disclosure border rather than the mock's lightly styled disclosure.
No mobile, assistive-technology, or packaged native-window visual claim is made.

## Automated verification

- Dictionary tests: 28 passed. The revised tests first failed against the old
  mixed technical dictionary, then passed with the biological-only implementation.
- Full application suite: 194 passed in 291.51 seconds, exit code 0.
- All 33 model/data files are byte-identical to the preceding boosting-rounds
  desktop build. No training or experiment/export regeneration was performed.
- The updated desktop package completed successfully and its bundled startup
  check passed. All 41 packaged source/model/data files match the verified source.
  Package: `dist/biology_dictionary_update/GTExTissueExplorer/GTExTissueExplorer.exe`.
- Sandboxed test startup was interrupted because the desktop subprocess checks
  did not finish there. The complete successful run used the approved local
  execution path with test temporary files inside the project.

final result: passed
