"""Latih Model page: choose the data, set up the model, and start training."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from gtex_explorer.display import METHODS, decimal
from gtex_explorer.display import count as format_count
from gtex_explorer.platform.cloud.uploads import file_uploader
from gtex_explorer.training.config import TrainingConfig, validate_training_config
from gtex_explorer.training.data import (
    TECHNICAL_FILTER_RULE,
    TrainingData,
    load_training_data,
    make_cv_assignments,
    make_holdout_assignment,
    usable_candidate_genes,
)
from gtex_explorer.training.jobs import start_job
from gtex_explorer.training.run_labels import L1_OPTION_LABELS, NOTEBOOK_DATA, RUN_NAME_LIMIT, XGBOOST_LABELS
from gtex_explorer.training.uploads import (
    UploadAudit,
    audit_uploads,
    read_upload_table,
    save_training_upload,
    upload_preview,
)
from gtex_explorer.ui.components.training_guide import render_training_guide
from gtex_explorer.ui.components.training_runs import show_settings
from gtex_explorer.ui.shell.error_messages import user_error
from gtex_explorer.ui.shell.theme import page_style
from gtex_explorer.ui.workspace import upload_widget_key

TRAINING_STEPS = ["1. Data training", "2. Konfigurasi", "3. Tinjau dan latih"]

# XGBoost fields of step 2 as (name, low, high, default, step, format). Labels come from XGBOOST_LABELS,
# which the review and the result pages also use.
XGBOOST_FIELDS = [
    ("n_estimators", 1, 1000, 300, 1, None),
    ("learning_rate", 0.001, 1, 0.05, 0.01, "%.3f"),
    ("max_depth", 1, 12, 6, 1, None),
    ("min_child_weight", 0, 100, 1.0, 0.1, "%.2f"),
    ("subsample", 0.1, 1, 0.8, 0.05, "%.2f"),
    ("colsample_bytree", 0.1, 1, 0.8, 0.05, "%.2f"),
]
# Under Parameter lanjutan, as (name, high, default); the low bound is 0.
XGBOOST_ADVANCED_FIELDS = [
    ("reg_lambda", 10000.0, 1.0),
    ("reg_alpha", 10000.0, 0.0),
    ("gamma", 1000.0, 0.0),
]


# --- Cached data ---


@st.cache_resource
def _data(root: str, version: tuple):
    """Load a training dataset once per version."""
    return load_training_data(Path(root))


@st.cache_data(max_entries=16, show_spinner=False)
def _usable_genes(_audit, source: tuple, holdout_percent: int, cv_folds: int | None) -> int:
    """Return how many genes pass the filter for this split."""
    return usable_candidate_genes(upload_preview(_audit), holdout_percent, cv_folds)


def cached_data(root):
    """Load a training dataset, reloading when its files change."""
    root = Path(root)
    version = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in sorted(root.iterdir()) if p.is_file())
    return _data(str(root), version)


# --- Form helpers ---


def _range_help(low, high):
    """Build the help text that states the allowed range."""

    number = lambda value: format_count(value) if isinstance(value, int) else decimal(value)
    return f"Rentang {number(low)}–{number(high)}."


def _top_k_typed():
    """Keep the gene count the user typed instead of fitting the default to the data."""
    st.session_state.training_top_k_typed = True


def _set_training_step(step):
    """Switch to another tab of the training form."""
    st.session_state.training_step = step


def _invalidate_training_upload():
    """Require a new check after the uploaded files change."""
    # File changes/removals invalidate the audited matrix, not only its local view.
    st.session_state.pop("training_upload_audit", None)
    st.session_state.pop("training_upload_signature", None)
    st.session_state.training_data_ready = False


def _dataset_strip(dataset):
    """Show the name and size of the chosen dataset."""
    columns = st.columns([3, 1, 1, 1, 1.3], vertical_alignment="center")
    with columns[0]:
        notebook = dataset["name"] == NOTEBOOK_DATA
        st.markdown("**" + (NOTEBOOK_DATA if notebook else "Data pengguna") + "**")
        # The notebook data has no file name, so only its description is repeated below the title.
        st.caption(
            dataset.get("metadata_name", "") if notebook else dataset["name"] + "  \n" + dataset.get("metadata_name", "")
        )
    for column, label, value in zip(
        columns[1:],
        ["Sampel", "Donor", "Jaringan", "Gen input"],
        [dataset["samples"], dataset["donors"], dataset["classes"], dataset["genes"]],
    ):
        with column:
            st.markdown(f"**{format_count(value)}**")
            st.caption(label)


# --- Values passed between the steps ---


@dataclass
class TrainingSamples:
    """The samples chosen in step 1."""

    audit: UploadAudit | None
    data: TrainingData | None
    dataset: dict | None


@dataclass
class TrainingChoices:
    """Everything chosen in step 2."""

    values: dict
    candidate_genes: int
    config: TrainingConfig | None
    validation_error: str | None


# --- Page ---


def render_training_page(data_root: Path, state_root: Path) -> None:
    """Show the Latih Model page."""
    st.html(page_style("training"))
    with st.container(key="training_setup"):
        _render_training_setup(data_root, state_root)


def _render_training_setup(data_root: Path, state_root: Path) -> None:
    """Show the form in the order the user works through it."""
    st.title("Latih model")
    st.caption("Bangun model dari data ekspresi Anda.")
    render_training_guide()
    data_tab, configuration_tab, review_tab = st.tabs(TRAINING_STEPS, key="training_step", on_change="rerun")
    with data_tab:
        samples = _samples_and_metadata_step(data_root)
    with configuration_tab:
        choices = _configuration_step(samples)
    with review_tab:
        _review_step(samples, choices, data_root, state_root)


# --- Step 1: samples and metadata ---


def _samples_and_metadata_step(data_root: Path) -> TrainingSamples:
    """Step 1: choose the samples and their metadata."""
    audit = st.session_state.get("training_upload_audit")
    data = None
    dataset = None
    source = st.radio(
        "Sumber data", ["Data pengguna", NOTEBOOK_DATA], horizontal=True, key="training_source"
    )
    if source == "Data pengguna":
        audit = _uploaded_samples(audit)
        if audit is not None:
            dataset = _describe_upload(audit)
    else:
        data, dataset = _bundled_samples(data_root)
    st.divider()
    st.button(
        "Lanjut ke konfigurasi",
        key="training_to_configuration",
        type="primary",
        icon=":material/arrow_forward:",
        on_click=_set_training_step,
        args=(TRAINING_STEPS[1],),
        disabled=dataset is None,
        width="stretch",
    )
    return TrainingSamples(audit, data, dataset)


def _uploaded_samples(audit: UploadAudit | None) -> UploadAudit | None:
    """Take the TPM matrix and metadata, and return the checked upload."""
    st.subheader("Unggah data training")
    st.caption("Matriks TPM dan metadata berlabel. ID sampel harus sama pada kedua berkas.")
    first, second = st.columns(2)
    with first:
        expression = file_uploader(
            "Matriks ekspresi TPM",
            type=["csv", "tsv", "txt", "gct"],
            key=upload_widget_key("training_expression_upload"),
            on_change=_invalidate_training_upload,
        )
    with second:
        metadata = file_uploader(
            "Metadata sampel",
            type=["csv", "tsv", "txt"],
            key=upload_widget_key("training_metadata_upload"),
            on_change=_invalidate_training_upload,
        )
    if expression is not None and metadata is not None:
        # Upload IDs change with every new file; hashing both files each rerun copied them.
        signature = f"{expression.file_id}:{metadata.file_id}"
        try:
            columns = list(read_upload_table(metadata.getvalue(), metadata.name).columns)
            sample, donor, tissue = _choose_metadata_columns(columns)
            signature += f":{sample}:{donor}:{tissue}"
            if st.session_state.get("training_upload_signature") != signature:
                audit = None
                st.session_state.pop("training_upload_audit", None)
            if st.button("Periksa data training", type="primary", key="training_audit"):
                audit = _check_uploaded_files(expression, metadata, sample, donor, tissue, signature)
        except (ValueError, OSError, UnicodeError) as error:
            audit = None
            st.session_state.pop("training_upload_audit", None)
            st.error(f"Metadata tidak dapat dibaca: {error}")
    elif expression is not None or metadata is not None:
        audit = None
        st.info("Lengkapi kedua berkas sebelum memeriksa data training.")
    return audit


def _choose_metadata_columns(columns: list) -> tuple:
    """Ask which metadata columns hold the sample, donor, and tissue."""
    st.caption(":primary[Pilihan pada setiap kolom di bawah diambil dari judul kolom pada berkas metadata.]")
    mapping = st.columns(3)
    sample = mapping[0].selectbox(
        "Kolom ID sampel",
        columns,
        index=columns.index("SAMPID") if "SAMPID" in columns else 0,
        key="training_sample_column",
    )
    mapping[0].caption("ID Sampel menghubungkan metadata dengan matriks TPM. ID harus sama persis.")
    donor = mapping[1].selectbox(
        "Kolom donor",
        columns,
        index=columns.index("SUBJID") if "SUBJID" in columns else min(1, len(columns) - 1),
        key="training_donor_column",
    )
    mapping[1].caption("Sampel dari donor yang sama tidak dipisah antara data pelatihan dan validasi.")
    tissue = mapping[2].selectbox(
        "Kolom jaringan",
        columns,
        index=columns.index("SMTSD") if "SMTSD" in columns else min(2, len(columns) - 1),
        key="training_tissue_column",
    )
    mapping[2].caption("Label jaringan yang dipelajari dan diprediksi model.")
    st.caption("Periksa kembali kolom donor dan jaringan. Pilihan yang keliru tidak selalu terdeteksi aplikasi.")
    return sample, donor, tissue


def _check_uploaded_files(expression, metadata, sample, donor, tissue, signature) -> UploadAudit | None:
    """Check the two files together and remember the result."""
    st.session_state.pop("training_upload_audit", None)
    try:
        with st.spinner("Memeriksa matriks dan mencocokkan metadata..."):
            audit = audit_uploads(
                expression.getvalue(),
                expression.name,
                metadata.getvalue(),
                metadata.name,
                sample,
                donor,
                tissue,
            )
        st.session_state.training_upload_audit = audit
        st.session_state.training_upload_signature = signature
        st.session_state.training_data_ready = True
        st.success("Data telah diperiksa. Lanjutkan ke konfigurasi.")
        return audit
    except (ValueError, OSError, UnicodeError) as error:
        st.error(str(error))
        return None


def _describe_upload(audit: UploadAudit) -> dict:
    """Show a checked upload and return its description."""
    dataset = {
        "name": audit.source["expression_file"],
        "metadata_name": audit.source["metadata_file"],
        "samples": len(audit.sample_ids),
        "donors": len(set(audit.donor_ids)),
        "classes": len(audit.classes),
        "genes": len(audit.gene_names),
    }
    _dataset_strip(dataset)
    with st.expander("Distribusi jaringan dan contoh metadata"):
        counts = (
            pd.Series([audit.classes[i] for i in audit.labels])
            .value_counts()
            .rename_axis("Jaringan")
            .reset_index(name="Sampel")
        )
        st.dataframe(counts, hide_index=True)
        st.dataframe(
            pd.DataFrame(
                {
                    "SAMPID": audit.sample_ids,
                    "Donor": audit.donor_ids,
                    "Jaringan": [audit.classes[i] for i in audit.labels],
                }
            ).head(10),
            hide_index=True,
        )
    return dataset


def _bundled_samples(data_root: Path) -> tuple:
    """Use the development data saved from the notebook."""
    data = None
    dataset = None
    try:
        data = cached_data(data_root)
        dataset = {
            "name": NOTEBOOK_DATA,
            "metadata_name": "Subset development tersimpan",
            "samples": len(data.sample_ids),
            "donors": len(set(data.donor_ids)),
            "classes": len(data.classes),
            "genes": len(data.raw_gene_names),
        }
        _dataset_strip(dataset)
        st.caption("Menggunakan subset development eksperimen. Sampel holdout tidak dimasukkan.")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"{NOTEBOOK_DATA} tidak tersedia: {error}")
    return data, dataset


# --- Step 2: split, preprocessing, feature selection, evaluation, and hyperparameters ---


def _configuration_step(samples: TrainingSamples) -> TrainingChoices:
    """Step 2, in pipeline order: holdout, candidate genes and transforms, selection, cross-validation, XGBoost."""
    dataset = samples.dataset
    if dataset:
        _dataset_strip(dataset)
    else:
        st.info("Unggah dan periksa TPM beserta metadata pada tab Data training.")
    st.divider()
    available = _limit_gene_choices(dataset)

    holdout, evaluation_note = _development_holdout_step()
    usable = _genes_passing_filter(samples, holdout)
    if usable is not None:
        available = max(1, min(available, usable))
    transforms, count, candidate_blank = _preprocessing_step(samples, available, usable)
    values = _method_and_genes_step(count)
    values.update(transforms)
    fold_note = _cross_validation_step(values, holdout, evaluation_note)
    _xgboost_step(values)

    config, validation_error = _check_choices(
        samples, values, count, candidate_blank, usable, evaluation_note, fold_note
    )
    _configuration_buttons(dataset, validation_error)
    return TrainingChoices(values, count, config, validation_error)


def _limit_gene_choices(dataset: dict | None) -> int:
    """Return the most genes this dataset offers, lowering a saved choice above it."""
    available = min(5000, dataset["genes"]) if dataset else 5000
    if st.session_state.get("training_top_k", 0) is not None and st.session_state.get("training_top_k", 0) > available:
        st.session_state.training_top_k = available
    return available


def _development_holdout_step() -> tuple:
    """Choose the share of samples held out from development."""
    st.subheader("Pembagian data utama")
    st.session_state.setdefault("training_holdout", 20)
    holdout = st.slider(
        "Porsi data holdout",
        0,
        50,
        step=5,
        format="%d%%",
        key="training_holdout",
        help="Dipisahkan lebih dulu untuk menguji model. Sisanya menjadi data development untuk melatih model.",
    )
    evaluation_note = st.container()
    st.divider()
    return holdout, evaluation_note


def _preprocessing_step(samples: TrainingSamples, available: int, usable: int | None) -> tuple:
    """Choose the candidate genes and the transforms applied to them before feature selection and XGBoost."""
    st.subheader("Pra-pemrosesan")
    st.session_state.setdefault("training_log2", True)
    st.session_state.setdefault("training_standardize", True)
    genes_column, transform_column = st.columns(2)
    with genes_column:
        count = _candidate_genes_input(samples, available, usable)
    with transform_column:
        log2 = st.checkbox(
            "Transformasi log2(TPM+1)",
            key="training_log2",
            help="Setiap nilai TPM diubah menjadi log2(TPM+1) sebelum seleksi fitur dan XGBoost.",
        )
        standardize = st.checkbox(
            "Standardisasi (z-score)",
            key="training_standardize",
            help=(
                "Setiap gen dipusatkan ke rerata 0 dengan simpangan baku 1. Rerata dan simpangan baku dihitung "
                "dari data training saja, termasuk pada setiap fold, lalu dipakai ulang untuk validasi dan prediksi."
            ),
        )
    st.caption(
        f"Gen kandidat disaring dengan filter teknis ({TECHNICAL_FILTER_RULE}) lalu diperingkat menurut varians "
        "log2(TPM+1), sehingga sama untuk setiap pilihan transformasi."
    )
    st.divider()
    candidate_blank = count is None
    return {"log2_transform": log2, "standardize": standardize}, available if candidate_blank else count, candidate_blank


def _genes_passing_filter(samples: TrainingSamples, holdout: int) -> int | None:
    """Count the genes of an upload that pass the filter for this split."""
    # A checked upload stays in the session after the user switches source, so it counts only while it is in use.
    if samples.dataset is None or samples.audit is None or samples.data is not None:
        return None
    return _usable_genes(
        samples.audit,
        tuple(sorted(samples.audit.source.items())),
        holdout,
        st.session_state.get("training_folds", 5) if st.session_state.get("training_cv") else None,
    )


def _method_and_genes_step(count: int) -> dict:
    """Choose the selection method and how many of the `count` candidate genes it keeps."""
    st.subheader("Seleksi fitur")
    method_column, genes_column = st.columns(2)
    with method_column:
        method = st.selectbox(
            "Metode seleksi fitur",
            list(METHODS),
            format_func=METHODS.get,
            key="training_selector",
        )
    # Until the user types a number, the gene count follows the data: up to 1.000 genes.
    # A number the user typed is only capped. Baseline always uses every candidate gene.
    if method == "baseline":
        st.session_state.training_top_k = count
    elif not st.session_state.get("training_top_k_typed"):
        st.session_state.training_top_k = min(1000, count)
    elif st.session_state.get("training_top_k") is not None and st.session_state.training_top_k > count:
        st.session_state.training_top_k = count
    with genes_column:
        k = st.number_input(
            "Top-k gen",
            value=None,
            step=1,
            help=_range_help(1, count),
            key="training_top_k",
            disabled=method == "baseline",
            on_change=_top_k_typed,
        )
    values = {"selector": method, "top_k": k}
    # The method's own settings sit under the method they belong to.
    _selector_options(values, method_column)
    st.divider()
    return values


def _candidate_genes_input(samples: TrainingSamples, available: int, usable: int | None) -> int | None:
    """Ask how many candidate genes to rank."""
    data, dataset = samples.data, samples.dataset
    st.session_state.setdefault("training_candidates", available)
    if data is not None:
        st.session_state.training_candidates = int(data.manifest["candidate_count"])
    elif st.session_state.training_candidates is not None and st.session_state.training_candidates > available:
        st.session_state.training_candidates = available
    count = st.number_input(
        "Gen kandidat",
        min_value=1,
        max_value=available,
        value=None,
        key="training_candidates",
        disabled=data is not None,
        help=(
            f"Jumlah gen yang dipertahankan setelah filter teknis ({TECHNICAL_FILTER_RULE}) dan pemeringkatan "
            "varians. Dipelajari dari data training setiap fold."
        ),
    )
    if usable is not None and usable < min(5000, dataset["genes"]):
        st.caption(
            f"Maksimum {usable} dari {dataset['genes']} gen. Gen lain tidak lolos filter teknis pada pembagian data ini "
            f"({TECHNICAL_FILTER_RULE})."
        )
    return count


def _selector_options(values: dict, method_column) -> None:
    """Show the settings that only the chosen selector has."""
    method = values["selector"]
    if method == "l1":
        with method_column:
            st.session_state.setdefault("training_l1_c", 1.0)
            values["l1_c"] = st.number_input(
                "C (regularisasi)",
                value=None,
                step=0.1,
                format="%.4f",
                help=_range_help(0.0001, 1000),
                key="training_l1_c",
            )
        with st.expander("Opsi selector"):
            values["l1_max_iter"] = st.number_input(
                L1_OPTION_LABELS["l1_max_iter"],
                value=20000,
                step=1,
                help=_range_help(100, 50000),
                key="training_l1_iterations",
            )
            values["l1_tol"] = st.number_input(
                L1_OPTION_LABELS["l1_tol"],
                value=0.001,
                step=0.0001,
                format="%.6f",
                help=_range_help(0.000001, 0.01),
                key="training_l1_tol",
            )
            values["l1_class_weight"] = st.selectbox(
                L1_OPTION_LABELS["l1_class_weight"],
                ["balanced", None], format_func=lambda x: x or "None", key="training_l1_weight"
            )
    elif method == "mutual_information":
        with method_column:
            values["mi_neighbors"] = st.number_input(
                "n_neighbors", value=3, step=1, help=_range_help(1, 20), key="training_mi_neighbors"
            )


def _cross_validation_step(values: dict, holdout: int, evaluation_note):
    """Choose whether to run cross-validation, and return the place for its note."""
    st.subheader("Evaluasi tambahan")
    values["evaluate_cv"] = st.checkbox("Evaluasi dengan cross-validation", key="training_cv")
    st.session_state.setdefault("training_folds", 5)
    values["cv_folds"] = st.slider(
        "Jumlah fold",
        2,
        10,
        key="training_folds",
        disabled=not values["evaluate_cv"],
        help="Setiap fold memakai 1 bagian untuk validasi dan sisanya untuk pelatihan.",
    )
    fold_note = st.container()
    if not values["evaluate_cv"]:
        fold_note.caption("Opsional. Cross-validation hanya memakai data development.")
    if holdout:
        values["holdout_percent"] = holdout
    else:
        evaluation_note.caption(
            "Dengan 0%, semua sampel menjadi data development dan model disimpan tanpa skor holdout."
        )
    return fold_note


def _xgboost_step(values: dict) -> None:
    """Collect the XGBoost hyperparameters."""
    st.subheader("Parameter XGBoost")
    columns = st.columns(3)
    xgb = {}
    # Bounds are checked by validate_training_config so an out-of-range entry stays visible as an error
    # instead of the widget silently keeping the previous value.
    for i, (name, low, high, default, step, number_format) in enumerate(XGBOOST_FIELDS):
        with columns[i % 3]:
            xgb[name] = st.number_input(
                XGBOOST_LABELS[name],
                value=default,
                step=step,
                format=number_format,
                help=_range_help(low, high),
                key="training_xgb_" + name,
            )
    with st.expander("Parameter lanjutan"):
        advanced = st.columns(2)
        for i, (name, high, default) in enumerate(XGBOOST_ADVANCED_FIELDS):
            with advanced[i % 2]:
                xgb[name] = st.number_input(
                    XGBOOST_LABELS[name],
                    value=default,
                    step=0.1,
                    format="%.2f",
                    help=_range_help(0, high),
                    key="training_xgb_" + name,
                )
        with advanced[1]:
            xgb["n_jobs"] = st.number_input(
                XGBOOST_LABELS["n_jobs"],
                1,
                max(1, os.cpu_count() or 1),
                min(4, os.cpu_count() or 1),
                key="training_xgb_threads",
            )
    values["xgboost"] = xgb
    st.divider()


def _check_choices(
    samples: TrainingSamples, values, count, candidate_blank, usable, evaluation_note, fold_note
) -> tuple:
    """Validate the form and return the configuration and any error."""
    validation_error = None
    config = None
    try:
        if candidate_blank:
            raise ValueError("Isi jumlah gen kandidat sebelum memulai training.")
        if usable == 0:
            raise ValueError(
                f"Tidak ada gen yang lolos filter teknis ({TECHNICAL_FILTER_RULE}) pada pembagian data ini. "
                "Periksa nilai TPM atau ubah pembagian data."
            )
        config = validate_training_config(values, candidate_count=count)
        _show_split_sizes(samples, config, values, evaluation_note, fold_note)
    except (ValueError, OSError, TypeError) as error:
        validation_error = str(error)
        st.error(user_error(error))
    st.divider()
    return config, validation_error


def _show_split_sizes(samples: TrainingSamples, config, values, evaluation_note, fold_note) -> None:
    """Show how many samples each part of the split will get."""
    audit, data, dataset = samples.audit, samples.data, samples.dataset
    if dataset and (values["evaluate_cv"] or values.get("holdout_percent")):
        preview = data if data is not None else upload_preview(audit)
        # The main split is settled first, so its counts stay visible even if the folds are refused.
        split = make_holdout_assignment(preview, config.holdout_percent) if config.holdout_percent else None
        if split:
            evaluation_note.caption(
                f"Data development: {len(split.train_indices)} sampel, untuk melatih model yang disimpan. "
                f"Data holdout: {len(split.validation_indices)} sampel, untuk menguji model. "
                "Donor tidak terbagi di antara keduanya."
            )
            if data is not None:
                evaluation_note.caption(
                    "Kedua bagian ini berasal dari subset development eksperimen. "
                    "Data holdout eksperimen notebook tetap tidak dipakai."
                )
        if config.evaluate_cv:
            folds = make_cv_assignments(preview, config.cv_folds, split.train_indices if split else None)
            train, validation = len(folds[0].train_indices), len(folds[0].validation_indices)
            fold_count = len(folds)
            fold_note.caption(
                f"{train + validation} sampel data development dibagi menjadi {fold_count} fold. "
                f"Setiap fold memakai sekitar {train} sampel ({round(100 * (fold_count - 1) / fold_count)}%) untuk pelatihan "
                f"dan {validation} sampel ({round(100 / fold_count)}%) untuk validasi. Donor tidak terbagi pada setiap fold."
                + (" Data holdout tidak ikut." if split else "")
            )


def _configuration_buttons(dataset: dict | None, validation_error: str | None) -> None:
    """Show the buttons that move between steps."""
    back, forward = st.columns(2)
    back.button(
        "Kembali ke data",
        key="training_back_to_data",
        icon=":material/arrow_back:",
        on_click=_set_training_step,
        args=(TRAINING_STEPS[0],),
        width="stretch",
    )
    forward.button(
        "Tinjau dan latih",
        key="training_to_review",
        type="primary",
        icon=":material/arrow_forward:",
        on_click=_set_training_step,
        args=(TRAINING_STEPS[2],),
        disabled=dataset is None or validation_error is not None,
        width="stretch",
    )


# --- Step 3: review and start training ---


def _review_step(samples: TrainingSamples, choices: TrainingChoices, data_root: Path, state_root: Path) -> None:
    """Step 3: review the choices and start training."""
    dataset = samples.dataset
    st.subheader("Tinjau data dan konfigurasi")
    if dataset:
        _dataset_strip(dataset)
        st.success("Data telah diperiksa.")
    else:
        st.info("Unggah dan periksa data pada langkah 1 sebelum memulai training.")
    st.divider()
    # The same view as a saved run on Hasil Pelatihan, in the order and with the labels of step 2.
    show_settings(choices.values, choices.candidate_genes)
    _show_readiness(dataset, choices)
    st.divider()
    st.text_input(
        "Nama pelatihan",
        key="training_run_name",
        max_chars=RUN_NAME_LIMIT,
        placeholder="Opsional, misalnya ANOVA 500 gen - holdout 20%",
        help="Memudahkan mencari dan membedakan hasil di Hasil Pelatihan, Perbandingan Model, Panel Gen, dan "
        "Prediksi Jaringan. Dapat diubah nanti di Hasil Pelatihan.",
    )
    _start_training_buttons(samples, choices, data_root, state_root)


def _show_readiness(dataset: dict | None, choices: TrainingChoices) -> None:
    """Say whether training can start and how the model will be scored."""
    values, validation_error = choices.values, choices.validation_error
    if validation_error:
        st.error(user_error(validation_error))
    elif dataset:
        st.success("Konfigurasi siap dilatih. Model dan panel gen akan disimpan.")
    if values.get("holdout_percent"):
        st.caption("Model yang disimpan hanya dilatih pada data development. Data holdout dipakai untuk menguji model.")
    elif not values["evaluate_cv"]:
        st.caption("Tanpa data holdout dan CV, model disimpan tanpa skor validation baru.")


def _start_training_buttons(
    samples: TrainingSamples, choices: TrainingChoices, data_root: Path, state_root: Path
) -> None:
    """Show the buttons that go back or start training."""
    audit, data, dataset = samples.audit, samples.data, samples.dataset
    count, config, validation_error = choices.candidate_genes, choices.config, choices.validation_error
    back, action = st.columns(2)
    back.button(
        "Kembali ke konfigurasi",
        key="training_review_back",
        icon=":material/arrow_back:",
        on_click=_set_training_step,
        args=(TRAINING_STEPS[1],),
        width="stretch",
    )
    with action:
        if st.button(
            "Latih dan simpan model",
            type="primary",
            key="training_start",
            width="stretch",
            disabled=dataset is None or validation_error is not None,
        ):
            try:
                with st.spinner("Menyimpan data dan memulai training..."):
                    actual_root = (
                        data_root if data is not None else save_training_upload(audit, state_root / "datasets", count)
                    )
                    record = start_job(config, actual_root, state_root, name=st.session_state.get("training_run_name"))
                # The next run starts unnamed.
                st.session_state.pop("training_run_name", None)
                st.session_state.training_selected_run = record.run_id
                st.session_state.training_run = record.run_id
                st.session_state.training_results_view = "Status proses"
                st.session_state.page = "Hasil Pelatihan"
                st.rerun()
            except (ValueError, OSError, KeyError) as error:
                st.error(user_error(error))
        st.caption("Model dan panel gen disimpan sebagai hasil pelatihan baru.")
