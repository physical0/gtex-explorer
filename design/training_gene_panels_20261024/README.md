# Grouped application workspaces and training gene panels

Design previews only. No product code was changed. Formal black-box testing remains stopped.

The user requested that the original notebook's Ringkasan, Perbandingan Metode, and Panel Gen be grouped separately from user retraining and its results. Prediksi Jaringan is shared; Kamus Biologi is standalone.

## Current visual selection set

These are the three revised images displayed after the comment that the first draft was superseded. Resolve any subsequent numbered selection against this set, not the preceding obsolete ungrouped draft.

1. `sectioned-cv.png`: both sidebar groups expanded, wide CV-fold gene table.
2. `collapsible-inspector.png`: collapsible workspace groups, CV-fold gene table and context inspector.
3. `workspace-final.png`: workspace selector, final development-fit gene panel.

All were generated independently using the built-in image-generation tool. The original gene-page target and actual training-page screenshot were attached as style references. Preserve the restrained white/navy/teal application language. Images are visual targets, not proof of implementation.

Data shown come from the actual saved smoke-test run `run_e7f7ed9df1984ff883a548a7677411be` in `.training-fast-cv-check`: ANOVA, 500 genes, 3 folds, XGBoost 2 rounds. Fold 1 has 549 training and 258 validation samples. Final fit uses 807 development samples and 494 donors. Fold table scores and final table scores deliberately differ and were read from their respective stored outputs.

The first ungrouped draft, generated before the user's grouping request, is not part of this selection set. Do not implement from it.
