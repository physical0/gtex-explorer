# Training on user data: redesign previews

Date: 28 October 2026. These images are design references. Implementation follows the corrected first direction, with **Custom** in the application as requested.

## Previously agreed scope

The user confirmed again that training must accept their own samples. This is an existing requirement, not a new proposal awaiting approval. The 24 October training gene-panel design explicitly says that user-dataset training must derive features from that dataset without requiring matches to the original model. The 22 October built-in-development-only scope must not override this later requirement.

The implemented source app was tested through its interface on 28 October and still trains on the bundled 807 development samples. Its upload-independent training controls are not an implementation of user-dataset training. That is an implementation gap.

Preserve the previously selected grouped navigation:

- Shared Prediksi Jaringan.
- Eksperimen Notebook: original Ringkasan, Perbandingan Metode, Panel Gen.
- Pelatihan Pengguna: Latih Model, Hasil Pelatihan, Panel Gen.
- Shared Kamus Biologi.

## Three final visual choices

Resolve later numbered choices against this set and this order:

1. `configuration-review.png`: configuration and a review of the new run, with a separate Data training tab. The first uncorrected draft was superseded. The retained version correctly shows Kustom for ANOVA with 500 genes.
2. `guided-data-upload.png`: a guided flow beginning with the TPM and metadata files, column mapping, and sample preview.
3. `compact-sections.png`: one page with expandable data and advanced sections, compact settings, and the training action below the review.

Each direction is generated with the built-in image-generation tool, using the user's current-page screenshot and the previously selected `training_gene_panels_20261024/collapsible-inspector.png` as visual references. The configuration layout received one focused correction to synchronize the preset label with its method. The displayed final images are the corrected configuration direction, then the upload direction, then the compact direction.

The corrected first direction is the implementation target. User-data training, separate saved results, and final/per-fold panels are implemented. Verification artifacts are in the workspace's `qa_user_training_20261028` directory.

## Data and behavior shared by every direction

The entry flow accepts an expression matrix and separate labeled metadata. Metadata sample IDs match the expression sample axis. Donor and tissue columns are mapped explicitly. Uploaded genes do not have to match an original or selected saved model. Feature candidates, imputation, transformations, scaling where applicable, supervised ranking, and model fitting must derive from the new training dataset. CV fits all learned preprocessing and selection inside each training fold.

User training results remain separate from the immutable original notebook exports. New model results must identify their dataset, selector parameters, gene count, XGBoost parameters, and final/fold scope. Only an explicitly selected saved model is used for prediction. Browsing results or a gene panel must not change the selected prediction model.

Latih Model configures a new run. It must not place a historical run's metrics or gene panel beside unrelated editable settings. Hasil Pelatihan displays recorded parameters, metrics, and status using descriptive names. Panel Gen shows final and available per-fold panels. Human-readable diagnostics replace raw JSON in the main presentation.

The example upload counts in the previews come from the actual prepared sample package: 100 samples, 100 donors, 13 tissue classes, and 74,628 raw gene rows. These are example dataset values, not hardcoded application requirements or training results. The matrix is 46,421,431 bytes, approximately 44.3 MiB. Metadata is 5,316 bytes, approximately 5.2 KiB. The proposed 5,000 candidate genes are a configurable preprocessing target, not an already measured count of retained genes. Two-fold CV is the illustrated choice because that count preserved all classes for this particular 100-sample dataset in the earlier splitter check.

No classification scores are invented for this 100-sample dataset. The preview does not claim the missing upload workflow already exists. In implementation, the uploaded dataset supplies the feature axis and tissue labels rather than reusing the bundled development matrix or its fitted preprocessing constants.
