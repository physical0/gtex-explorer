"""Latih Model page: choose the data, set up the model, and start training."""

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from gtex_explorer.platform.cloud.uploads import file_uploader
from gtex_explorer.training.config import TrainingConfig, training_presets, validate_training_config
from gtex_explorer.training.data import (
    TrainingData,
    load_training_data,
    make_cv_assignments,
    make_holdout_assignment,
    usable_candidate_genes,
)
from gtex_explorer.training.jobs import start_job
from gtex_explorer.training.run_labels import METHODS
from gtex_explorer.training.uploads import (
    UploadAudit,
    audit_uploads,
    read_upload_table,
    save_training_upload,
    upload_preview,
)
from gtex_explorer.ui.shell.error_messages import user_error
from gtex_explorer.ui.shell.theme import page_style
from gtex_explorer.ui.workspace import upload_widget_key

TRAINING_STEPS = ["1. Data training", "2. Konfigurasi", "3. Tinjau dan latih"]


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


def _preset_changed():
    """Copy the chosen preset into the form."""
    preset = training_presets().get(st.session_state.training_preset)
    if preset:
        st.session_state.training_selector = preset["selector"]
        st.session_state.training_l1_c = float(preset.get("l1_c", 1))


def _range_help(low, high):
    """Build the help text that states the allowed range."""

    def number(value):
        """Format a bound the way the form shows it."""
        return f"{value:,}".replace(",", ".") if isinstance(value, int) else f"{value:g}".replace(".", ",")

    return f"Rentang {number(low)}–{number(high)}."


def _selection_changed():
    """Mark the form as custom after a manual change."""
    st.session_state.training_preset = "Custom"


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
        st.markdown(
            "**"
            + ("Data eksperimen notebook" if dataset["name"] == "Data eksperimen notebook" else "Data pengguna")
            + "**"
        )
        st.caption(dataset["name"] + "  \n" + dataset.get("metadata_name", ""))
    for column, label, value in zip(
        columns[1:],
        ["Sampel", "Donor", "Jaringan", "Gen input"],
        [dataset["samples"], dataset["donors"], dataset["classes"], dataset["genes"]],
    ):
        with column:
            st.markdown("**" + f"{value:,}".replace(",", ".") + "**")
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
        "Sumber data", ["Data pengguna", "Data eksperimen notebook"], horizontal=True, key="training_source"
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
            "name": "Data eksperimen notebook",
            "metadata_name": "Subset development tersimpan",
            "samples": len(data.sample_ids),
            "donors": len(set(data.donor_ids)),
            "classes": len(data.classes),
            "genes": len(data.raw_gene_names),
        }
        _dataset_strip(dataset)
        st.caption("Menggunakan subset development eksperimen. Sampel holdout tidak dimasukkan.")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Data notebook tidak tersedia: {error}")
    return data, dataset


# --- Step 2: split, method and genes, evaluation, and hyperparameters ---


def _configuration_step(samples: TrainingSamples) -> TrainingChoices:
    """Step 2: holdout, method and genes, cross-validation, then XGBoost."""
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
    values, count, candidate_blank = _method_and_genes_step(samples, available, usable)
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


def _method_and_genes_step(samples: TrainingSamples, available: int, usable: int | None) -> tuple:
    """Choose feature selection plus XGBoost, or XGBoost only, and the number of genes."""
    st.subheader("Seleksi fitur")
    selector_left, selector_right = st.columns(2)
    with selector_left:
        st.selectbox(
            "Preset seleksi fitur",
            [*training_presets(), "Custom"],
            key="training_preset",
            on_change=_preset_changed,
        )
    with selector_right:
        method = st.selectbox(
            "Metode seleksi fitur",
            list(METHODS),
            format_func=METHODS.get,
            key="training_selector",
            on_change=_selection_changed,
        )
    with selector_right:
        count = _candidate_genes_input(samples, available, usable)
    candidate_blank = count is None
    if candidate_blank:
        count = available
    # A template fits its gene count to the data. A number the user typed is only capped.
    template = training_presets().get(st.session_state.training_preset, {})
    if method == "baseline":
        st.session_state.training_top_k = count
    elif "top_k" in template:
        st.session_state.training_top_k = min(template["top_k"], count)
    elif st.session_state.get("training_top_k") is not None and st.session_state.training_top_k > count:
        st.session_state.training_top_k = count
    st.session_state.setdefault("training_top_k", min(1000, count))
    with selector_left:
        k = st.number_input(
            "Top-k gen",
            value=None,
            step=1,
            help=_range_help(1, count),
            key="training_top_k",
            disabled=method == "baseline",
            on_change=_selection_changed,
        )
    values = {"selector": method, "top_k": k}
    _selector_options(values, selector_right)
    st.divider()
    return values, count, candidate_blank


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
        on_change=_selection_changed,
        help="Jumlah gen setelah filter teknis dan pemeringkatan varians. Dipelajari dari data training setiap fold.",
    )
    if usable is not None and usable < min(5000, dataset["genes"]):
        st.caption(
            f"Maksimum {usable} dari {dataset['genes']} gen. Gen lain tidak lolos filter teknis pada pembagian data ini: "
            "bernilai sedikitnya 1 TPM pada 20% sampel dan bervariasi antarsampel."
        )
    return count


def _selector_options(values: dict, selector_right) -> None:
    """Show the settings that only the chosen selector has."""
    method = values["selector"]
    if method == "l1":
        with selector_right:
            st.session_state.setdefault("training_l1_c", 1.0)
            values["l1_c"] = st.number_input(
                "C (regularisasi)",
                value=None,
                step=0.1,
                format="%.4f",
                help=_range_help(0.0001, 1000),
                key="training_l1_c",
                on_change=_selection_changed,
            )
        with st.expander("Opsi selector"):
            values["l1_max_iter"] = st.number_input(
                "Maksimum iterasi", value=20000, step=1, help=_range_help(100, 50000), key="training_l1_iterations"
            )
            values["l1_tol"] = st.number_input(
                "Toleransi",
                value=0.001,
                step=0.0001,
                format="%.6f",
                help=_range_help(0.000001, 0.01),
                key="training_l1_tol",
            )
            values["l1_class_weight"] = st.selectbox(
                "Bobot kelas", ["balanced", None], format_func=lambda x: x or "None", key="training_l1_weight"
            )
    elif method == "mutual_information":
        with selector_right:
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
    specs = [
        ("n_estimators", "Putaran boosting", 1, 1000, 300, 1, None),
        ("learning_rate", "Learning rate", 0.001, 1, 0.05, 0.01, "%.3f"),
        ("max_depth", "Max depth", 1, 12, 6, 1, None),
        ("min_child_weight", "Min child weight", 0, 100, 1.0, 0.1, "%.2f"),
        ("subsample", "Subsample", 0.1, 1, 0.8, 0.05, "%.2f"),
        ("colsample_bytree", "Colsample (bytree)", 0.1, 1, 0.8, 0.05, "%.2f"),
    ]
    for i, (name, label, low, high, default, step, number_format) in enumerate(specs):
        with columns[i % 3]:
            xgb[name] = st.number_input(
                label,
                value=default,
                step=step,
                format=number_format,
                help=_range_help(low, high),
                key="training_xgb_" + name,
            )
    with st.expander("Parameter lanjutan"):
        advanced = st.columns(2)
        for i, (name, label, high, default) in enumerate(
            [
                ("reg_lambda", "L2 XGBoost (lambda)", 10000.0, 1.0),
                ("reg_alpha", "L1 XGBoost (alpha)", 10000.0, 0.0),
                ("gamma", "Gamma", 1000.0, 0.0),
            ]
        ):
            with advanced[i % 2]:
                xgb[name] = st.number_input(
                    label,
                    value=default,
                    step=0.1,
                    format="%.2f",
                    help=_range_help(0, high),
                    key="training_xgb_" + name,
                )
        with advanced[1]:
            xgb["n_jobs"] = st.number_input(
                "Thread CPU",
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
                "Tidak ada gen yang lolos filter teknis pada pembagian data ini. Periksa nilai TPM atau ubah pembagian data."
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
    _show_choices(choices)
    _show_readiness(dataset, choices)
    st.divider()
    _start_training_buttons(samples, choices, data_root, state_root)


def _show_choices(choices: TrainingChoices) -> None:
    """Show a summary of the chosen method and evaluation."""
    values, count = choices.values, choices.candidate_genes
    method, k, xgb = values["selector"], values["top_k"], values["xgboost"]
    selection, model = st.columns(2, gap="large")
    with selection:
        st.subheader("Seleksi fitur")
        rows = [
            ("Metode", METHODS[method]),
            ("Gen kandidat", f"{count:,}".replace(",", ".")),
            ("Top-k gen", f"{k or 0:,}".replace(",", ".")),
        ]
        if method == "l1":
            rows.append(("C", str(values.get("l1_c"))))
        elif method == "mutual_information":
            rows.append(("n_neighbors", str(values["mi_neighbors"])))
        st.table(pd.DataFrame(rows, columns=["Pengaturan", "Nilai"]).set_index("Pengaturan"))
    with model:
        st.subheader("Model dan evaluasi")
        rows = [
            ("Classifier", "XGBoost"),
            ("Putaran boosting", str(xgb["n_estimators"])),
            ("Data development", f"{100 - values.get('holdout_percent', 0)}%"),
            ("Data holdout", f"{values['holdout_percent']}%" if values.get("holdout_percent") else "Tidak ada"),
            (
                "Evaluasi tambahan",
                f"Cross-validation {values['cv_folds']} fold" if values["evaluate_cv"] else "Tidak ada",
            ),
        ]
        if values["evaluate_cv"] or values.get("holdout_percent"):
            rows.append(("Pembagian", "Donor terpisah"))
        st.table(pd.DataFrame(rows, columns=["Pengaturan", "Nilai"]).set_index("Pengaturan"))
    with st.expander("Parameter XGBoost dan opsi selector"):
        st.dataframe(
            pd.DataFrame(
                [{"Parameter": name, "Nilai": str(value)} for name, value in values.items() if name != "xgboost"]
                + [{"Parameter": name, "Nilai": str(value)} for name, value in xgb.items()]
            ),
            hide_index=True,
        )


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
                    record = start_job(config, actual_root, state_root)
                st.session_state.training_selected_run = record.run_id
                st.session_state.training_run = record.run_id
                st.session_state.training_results_view = "Status proses"
                st.session_state.page = "Hasil Pelatihan"
                st.rerun()
            except (ValueError, OSError, KeyError) as error:
                st.error(user_error(error))
        st.caption("Model dan panel gen disimpan sebagai hasil pelatihan baru.")
