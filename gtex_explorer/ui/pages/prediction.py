"""Prediksi Jaringan page."""

from __future__ import annotations

import html
import io

import altair as alt
import pandas as pd
import streamlit as st

from gtex_explorer.experiments.reference_tables import DATA_DIR
from gtex_explorer.inference.audit import InputAudit, audit_and_align_input
from gtex_explorer.inference.parsing import parse_expression_file
from gtex_explorer.inference.prediction import explain_prediction, predict_tissues
from gtex_explorer.platform.cloud.media import download_button
from gtex_explorer.platform.cloud.uploads import file_uploader
from gtex_explorer.ui.shell.error_messages import user_error
from gtex_explorer.ui.workspace import current_workspace_root, upload_widget_key

ANALYSIS_STATE_KEYS = (
    "audited_tpm",
    "input_matrix_before_imputation",
    "input_audit",
    "input_filename",
    "input_file_size",
    "input_source",
    "predictions",
    "probabilities",
    "active_sample",
    "active_sample_selector",
)


# --- Model selection and session state ---


def clear_model_analysis_state(state) -> None:
    """Forget the audit and prediction that belong to the previous model."""
    for key in ANALYSIS_STATE_KEYS:
        state.pop(key, None)
    # Assign explicitly: deleting a widget key can restore its old browser value.
    state["prediction_boosting_rounds"] = None


def sync_prediction_model(state, entry) -> None:
    """Clear old results when the chosen model changes."""
    identity = (entry.model_id, entry.manifest_hash)
    if state.get("prediction_model_identity") != identity:
        clear_model_analysis_state(state)
        state["prediction_model_identity"] = identity
    state["active_model_id"] = entry.model_id


# Models are large once loaded, so an instance keeps only the few in recent use.
@st.cache_resource(max_entries=3, show_spinner="Memuat model prediksi...")
def _selected_bundle(entry):
    """Load the chosen model."""
    from gtex_explorer.platform.model_registry import load_prediction_model

    return load_prediction_model(entry)


def _choose_model():
    """Show the model selector and return the loaded model."""
    from gtex_explorer.platform.desktop_runtime import state_directory
    from gtex_explorer.platform.model_registry import list_models
    from gtex_explorer.platform.paths import APP_ROOT as APP_DIR

    issues = []
    entries = list_models(APP_DIR, current_workspace_root(state_directory()), issues=issues)
    if issues:
        with st.expander(f"Model tidak tersedia ({len(issues)})", expanded=True):
            for issue in issues:
                st.warning(f"{issue['model']}: {user_error(issue['reason'])}")
    if not entries:
        clear_model_analysis_state(st.session_state)
        st.info("Workspace ini belum memiliki model yang dapat digunakan. Latih model untuk memulai prediksi.")
        if st.button("Latih model", key="prediction_empty_train", icon=":material/settings:"):
            st.session_state.page = "Pelatihan Model"
            st.rerun()
        return None, None
    by_id = {entry.model_id: entry for entry in entries}
    if st.session_state.get("prediction_model") not in by_id:
        remembered = st.session_state.get("active_model_id")
        st.session_state.prediction_model = remembered if remembered in by_id else entries[0].model_id
    selected = st.selectbox(
        "Model prediksi",
        list(by_id),
        format_func=lambda key: by_id[key].display_name,
        key="prediction_model",
        label_visibility="collapsed",
        width=500,
    )
    entry = by_id[selected]
    reload_shared_demo = st.session_state.get("input_source") == "demo" and st.session_state.get(
        "prediction_model_identity"
    ) != (entry.model_id, entry.manifest_hash)
    previous_sample = st.session_state.get("active_sample")
    sync_prediction_model(st.session_state, entry)
    try:
        pre, model = _selected_bundle(entry)
    except Exception as error:
        clear_model_analysis_state(st.session_state)
        st.error(f"Model tidak dapat dimuat: {error}")
        return None, None
    st.session_state.active_run_id = pre.get("run_id", "")
    st.caption("Contoh lokal yang sama digunakan untuk semua model.")
    if reload_shared_demo:
        with st.spinner("Menyesuaikan contoh lokal dengan model..."):
            _load_local_demo(pre)
        if previous_sample in st.session_state.audited_tpm.index:
            st.session_state.active_sample = previous_sample
            st.session_state.active_sample_selector = previous_sample
    return pre, model


def _clear_analysis_state() -> None:
    """Forget the current audit and prediction."""
    for key in ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)


def _clear_prediction_results() -> None:
    """Changing rounds keeps the audited input but invalidates its old results."""
    st.session_state.pop("predictions", None)
    st.session_state.pop("probabilities", None)


def _sync_uploaded_input(uploaded_file) -> None:
    """Invalidate audited data/results whenever the pending input changes.

    Every upload gets a new file_id, so the file is not read or hashed on each rerun.
    """
    signature = None if uploaded_file is None else (uploaded_file.name, uploaded_file.file_id, int(uploaded_file.size))
    if signature != st.session_state.get("pending_upload_signature"):
        _clear_analysis_state()
        st.session_state.pending_upload_signature = signature


def _reminder_percentages() -> tuple[int, int]:
    """Return the reminder thresholds in use."""
    # Widget state is cleaned up on navigation and interrupted audit reruns.
    # Durable preferences retain user choices independently of rendered widgets.
    saved = st.session_state.get("prediction_reminder_preferences", {})
    return tuple(
        st.session_state.get(key, saved.get(key, 90))
        for key in (
            "prediction_positive_percent",
            "prediction_completeness_percent",
        )
    )


def _save_reminder_preferences() -> None:
    """Remember the thresholds across page changes."""
    st.session_state.prediction_reminder_preferences = {
        key: st.session_state.get(key, 90) for key in ("prediction_positive_percent", "prediction_completeness_percent")
    }


# --- Input handling ---


def _audit_matrix(
    matrix: pd.DataFrame,
    filename: str,
    file_size: int,
    source: str,
    preprocessor: dict,
) -> None:
    """Audit an expression matrix and store the result in the session."""
    positive_percent, completeness_percent = _reminder_percentages()
    aligned, audit = audit_and_align_input(
        matrix,
        preprocessor,
        minimum_observed_fraction=completeness_percent / 100,
        minimum_positive_fraction=positive_percent / 100,
    )
    _clear_analysis_state()
    st.session_state.audited_tpm = aligned
    st.session_state.input_audit = audit
    st.session_state.input_matrix_before_imputation = matrix
    st.session_state.input_filename = filename
    st.session_state.input_file_size = int(file_size)
    st.session_state.input_source = source
    if len(aligned):
        st.session_state.active_sample = str(aligned.index[0])


def _load_local_demo(preprocessor: dict) -> None:
    """Load and audit the bundled example data."""
    demo_path = DATA_DIR / "demo_tpm_shared.csv"
    if not demo_path.is_file():
        raise FileNotFoundError("Data contoh lokal tidak ditemukan.")
    with demo_path.open("rb") as handle:
        matrix = parse_expression_file(handle, demo_path.name, model_genes=preprocessor["selected_genes"])
    _audit_matrix(
        matrix=matrix,
        filename=demo_path.name,
        file_size=demo_path.stat().st_size,
        source="demo",
        preprocessor=preprocessor,
    )


# --- Formatting ---


def _format_file_size(size: int) -> str:
    """Format a file size for display."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _format_integer(value: int) -> str:
    """Format a whole number with dot separators."""
    return f"{int(value):,}".replace(",", ".")


def _format_percent(value: float, digits: int = 1) -> str:
    """Format a share as a percentage with a decimal comma."""
    return f"{value:.{digits}%}".replace(".", ",")


# --- Audit panel ---


def _render_audit_summary(audit: InputAudit, feature_count: int, sample_id: str) -> None:
    """Model-ID overlap is file-wide. Expression counts belong to this sample."""
    quality = audit.sample_quality.loc[audit.sample_quality["SAMPID"].eq(sample_id)].iloc[0]
    rows = pd.DataFrame(
        [
            (
                "Kecocokan ID gen model",
                f"{_format_integer(audit.matched_gene_count)} / "
                f"{_format_integer(feature_count)} ({_format_percent(audit.matched_fraction)})",
            ),
            (
                "Kelengkapan nilai gen model",
                f"{_format_integer(quality['observed_model_genes'])} / "
                f"{_format_integer(feature_count)} ({_format_percent(quality['observed_fraction'])})",
            ),
            (
                "Gen input dengan TPM > 0",
                f"{_format_integer(quality['uploaded_positive_genes'])} / "
                f"{_format_integer(audit.supplied_gene_count)} ({_format_percent(quality['uploaded_positive_fraction'])})",
            ),
        ],
        columns=["Pemeriksaan", "Hasil"],
    )
    # A small semantic table wraps at narrow widths instead of hiding its values
    # behind the dataframe widget's horizontal scroll area.
    st.markdown(rows.to_html(index=False, classes="prediction-audit-summary", border=0), unsafe_allow_html=True)


def _render_audit_details(audit: InputAudit, feature_count: int) -> None:
    """Show every audit check in a table."""
    matched_status = "Informasi"
    rows = pd.DataFrame(
        [
            ("Orientasi matriks", audit.source_orientation, "Lolos"),
            (
                "Kecocokan gen model",
                f"{_format_integer(audit.matched_gene_count)}/"
                f"{_format_integer(feature_count)} "
                f"({_format_percent(audit.matched_fraction)})",
                matched_status,
            ),
            ("Gen model tidak ditemukan", _format_integer(audit.missing_gene_count), "Informasi"),
            (
                "Gen input di luar model",
                _format_integer(audit.supplied_gene_count - audit.matched_gene_count),
                "Informasi",
            ),
            (
                "Nilai negatif",
                f"{audit.negative_value_count:,}",
                "Lolos" if audit.negative_value_count == 0 else "Tinjau",
            ),
            (
                "Nilai tak hingga",
                f"{audit.infinite_value_count:,}",
                "Lolos" if audit.infinite_value_count == 0 else "Tinjau",
            ),
            (
                "ID gen duplikat",
                f"{audit.duplicate_gene_count:,}",
                "Lolos" if audit.duplicate_gene_count == 0 else "Tinjau",
            ),
            (
                "Nilai non-numerik",
                f"{audit.invalid_numeric_value_count:,}",
                "Lolos" if audit.invalid_numeric_value_count == 0 else "Tinjau",
            ),
        ],
        columns=["Pemeriksaan", "Hasil", "Status"],
    )
    st.dataframe(
        rows,
        hide_index=True,
        width="stretch",
        height=315,
        column_config={
            "Pemeriksaan": st.column_config.TextColumn(width="medium"),
            "Hasil": st.column_config.TextColumn(width="medium"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption("Kelengkapan dan ekspresi nonzero dihitung sebelum imputasi untuk setiap sampel.")
    st.dataframe(audit.sample_quality, hide_index=True, width="stretch")


def _render_reminder_controls(audit: InputAudit | None) -> None:
    """Show the settings for the data quality reminders."""
    positive_percent, completeness_percent = _reminder_percentages()
    with st.expander("Pengaturan pengingat", expanded=True, key="prediction_reminders"):
        label_column, value_column = st.columns([1.4, 1], vertical_alignment="center")
        label_column.markdown("Gen nonzero (%)")
        with value_column:
            st.number_input(
                "Gen nonzero (%)",
                min_value=0,
                max_value=100,
                value=positive_percent,
                step=1,
                key="prediction_positive_percent",
                label_visibility="collapsed",
                on_change=_save_reminder_preferences,
            )
        st.caption("Dihitung dari seluruh gen input per sampel.")
        label_column, value_column = st.columns([1.4, 1], vertical_alignment="center")
        label_column.markdown("Kelengkapan gen model (%)")
        with value_column:
            st.number_input(
                "Kelengkapan gen model (%)",
                min_value=0,
                max_value=100,
                value=completeness_percent,
                step=1,
                key="prediction_completeness_percent",
                label_visibility="collapsed",
                on_change=_save_reminder_preferences,
            )
        st.caption("Nilai tersedia sebelum imputasi. TPM = 0 tetap dihitung.")
        st.caption("Ambang hanya mengatur pengingat. Rentang 0–100%.")
        if audit is not None and audit.ready:
            if audit.low_positive_sample_count:
                st.caption(
                    f"{audit.low_positive_sample_count} sampel di bawah ambang gen nonzero {audit.positive_expression_threshold:.0%}."
                )
            if audit.below_threshold_sample_count:
                st.caption(
                    f"{audit.below_threshold_sample_count} sampel di bawah ambang kelengkapan {audit.completeness_threshold:.0%}."
                )


def _render_input_panel(preprocessor: dict | None) -> None:
    """Show the upload and audit column."""
    st.markdown('<div class="panel-heading">Audit matriks input</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-copy">Periksa struktur dan nilai TPM sebelum model dijalankan.</div>',
        unsafe_allow_html=True,
    )

    # Upload.
    st.markdown("**Unggah matriks TPM**")
    st.caption("CSV, TSV, TXT, atau GCT. Baris dapat berupa sampel atau gen.")
    uploaded_file = file_uploader(
        "Pilih berkas TPM",
        type=["csv", "tsv", "txt", "gct"],
        label_visibility="collapsed",
        key=upload_widget_key("offline_tpm_upload"),
    )

    _sync_uploaded_input(uploaded_file)

    # Recount the reminders if their thresholds changed.
    positive_percent, completeness_percent = _reminder_percentages()
    original_matrix = st.session_state.get("input_matrix_before_imputation")
    existing_audit = st.session_state.get("input_audit")
    if (
        preprocessor is not None
        and original_matrix is not None
        and existing_audit is not None
        and (
            existing_audit.completeness_threshold != completeness_percent / 100
            or existing_audit.positive_expression_threshold != positive_percent / 100
        )
    ):
        # Recount original measurements, never the already imputed matrix.
        # A reminder threshold does not alter the model input or its predictions.
        _, st.session_state.input_audit = audit_and_align_input(
            original_matrix,
            preprocessor,
            minimum_observed_fraction=completeness_percent / 100,
            minimum_positive_fraction=positive_percent / 100,
        )

    # Check the upload, or load the example.
    button_columns = st.columns([1, 1])
    with button_columns[0]:
        inspect_clicked = st.button(
            "Periksa input",
            icon=":material/fact_check:",
            type="primary",
            width="stretch",
            disabled=uploaded_file is None or preprocessor is None,
            key="inspect_uploaded_input",
        )
    with button_columns[1]:
        demo_clicked = st.button(
            "Gunakan contoh lokal",
            icon=":material/folder_open:",
            width="stretch",
            disabled=preprocessor is None,
            key="load_local_demo",
        )

    if inspect_clicked and uploaded_file is not None and preprocessor is not None:
        _clear_analysis_state()
        try:
            # A fresh stream makes repeated audits independent of file cursor state.
            with st.spinner("Membaca berkas dan memeriksa input..."):
                matrix = parse_expression_file(
                    io.BytesIO(uploaded_file.getvalue()), uploaded_file.name, model_genes=preprocessor["selected_genes"]
                )
                _audit_matrix(
                    matrix=matrix,
                    filename=uploaded_file.name,
                    file_size=int(uploaded_file.size),
                    source="upload",
                    preprocessor=preprocessor,
                )
            st.rerun()
        except Exception as error:
            st.error(f"Berkas tidak dapat diperiksa: {error}")

    if demo_clicked and preprocessor is not None:
        try:
            with st.spinner("Memuat dan memeriksa contoh lokal..."):
                _load_local_demo(preprocessor)
            st.rerun()
        except Exception as error:
            st.error(f"Data contoh tidak dapat dimuat: {error}")

    # Nothing more to show until an audit exists.
    audit = st.session_state.get("input_audit")
    aligned = st.session_state.get("audited_tpm")
    if audit is None or aligned is None:
        _render_reminder_controls(None)
        return

    # Choose which sample to review.
    sample_ids = aligned.index.astype(str).tolist()
    if not sample_ids:
        st.error(audit.message)
        return
    if st.session_state.get("active_sample") not in sample_ids:
        st.session_state.active_sample = sample_ids[0]
    if st.session_state.get("active_sample_selector") not in sample_ids:
        st.session_state.active_sample_selector = st.session_state.active_sample
    if len(sample_ids) > 1:
        st.caption(
            f"{_format_file_size(st.session_state.input_file_size)} - {_format_integer(audit.sample_count)} sampel"
        )
        active_sample = st.selectbox("Pilih sampel", sample_ids, key="active_sample_selector")
    else:
        active_sample = sample_ids[0]
        st.caption(f"{_format_file_size(st.session_state.input_file_size)} - 1 sampel - {active_sample}")
    st.session_state.active_sample = active_sample

    # Audit result.
    _render_audit_summary(audit, int(preprocessor["feature_count"]), active_sample)
    if not audit.ready:
        st.error(f"**Input perlu diperbaiki**\n\n{audit.message}")
    _render_reminder_controls(audit)

    # Explain any filled-in values and possibly log-transformed input.
    imputed_count = int(audit.sample_quality["imputed_model_genes"].sum())
    if audit.ready and (audit.missing_gene_count or imputed_count):
        fully_imputed = int(audit.sample_quality["observed_model_genes"].eq(0).sum())
        imputation_text = f"{_format_integer(imputed_count)} nilai yang tidak tersedia"
        if fully_imputed == audit.sample_count:
            imputation_text = f"Seluruh {_format_integer(imputed_count)} nilai gen model"
        notice = (
            f"**Informasi gen model**\n\n{_format_integer(audit.matched_gene_count)} gen model ditemukan. "
            f"{imputation_text} diisi dengan median data training tersimpan."
        )
        if audit.supplied_gene_count > audit.matched_gene_count:
            notice += " Gen input di luar model tidak digunakan."
        if fully_imputed and fully_imputed < audit.sample_count:
            notice += f" Seluruh nilai gen model diimputasi pada {fully_imputed} sampel."
        st.info(notice)
    if audit.appears_log_transformed:
        st.warning(
            "**Periksa satuan data**\n\nRentang nilai menyerupai data yang sudah ditransformasi. Pastikan berkas memuat TPM mentah."
        )
    with st.expander("Detail pemeriksaan input"):
        _render_audit_details(audit, int(preprocessor["feature_count"]))


# --- Prediction results ---


def _prediction_export() -> bytes:
    """Build the CSV of predictions and probabilities."""
    export_table = st.session_state.predictions.copy()
    export_table["model_id"] = st.session_state.get("active_model_id", "original_l1_c_1_1000")
    export_table["run_id"] = st.session_state.get("active_run_id", "")
    probabilities = st.session_state.probabilities.reset_index()
    export_table = export_table.merge(probabilities, on="SAMPID")
    return export_table.to_csv(index=False).encode("utf-8")


def _probability_percent(value: float) -> str:
    """Readable percentages without rounding tiny positive probabilities to zero."""
    if 0 < value < 0.0001:
        return "<0,01%"
    return f"{value * 100:.2f}%".replace(".", ",")


def _probability_chart(probability_row: pd.Series, predicted_tissue: str) -> alt.LayerChart:
    """Build the chart of probabilities for every tissue."""
    distribution = probability_row.sort_values(ascending=False, kind="stable").rename("probability").reset_index()
    distribution.columns = ["tissue", "probability"]
    distribution["rank"] = range(1, len(distribution) + 1)
    distribution["percentage_label"] = distribution["probability"].map(_probability_percent)
    base = alt.Chart(distribution).encode(
        y=alt.Y(
            "tissue:N",
            sort=distribution["tissue"].tolist(),
            title=None,
            axis=alt.Axis(
                labelLimit=0, labelFontSize=13, labelOverlap=False, ticks=False, domain=False, labelPadding=10
            ),
        ),
        tooltip=[
            alt.Tooltip("tissue:N", title="Jaringan"),
            alt.Tooltip("percentage_label:N", title="Probabilitas"),
        ],
    )
    bars = base.mark_bar(size=17, cornerRadiusEnd=2).encode(
        x=alt.X(
            "probability:Q",
            title=None,
            scale=alt.Scale(domain=[0, 1], nice=False),
            axis=alt.Axis(format="%", values=[0, 0.25, 0.5, 0.75, 1], labelFontSize=12, gridColor="#edf0f2"),
        ),
        color=alt.condition(
            alt.datum.tissue == predicted_tissue,
            alt.value("#078c92"),
            alt.value("#a6b5bd"),
        ),
    )
    # Fixed right-hand value column makes small probabilities readable without
    # inflating their bars or changing the shared zero-to-one scale.
    labels = base.mark_text(align="left", dx=10, fontSize=13, color="#344054", clip=False).encode(
        x=alt.XDatum(1),
        text="percentage_label:N",
    )
    return (
        (bars + labels)
        .properties(
            height=max(120, len(distribution) * 30),
            padding={"left": 0, "right": 75, "top": 5, "bottom": 0},
        )
        .configure_view(stroke=None)
    )


def _render_prediction_result(
    preprocessor: dict,
    classifier,
) -> None:
    """Show the prediction, probabilities, and top genes for one sample."""
    predictions = st.session_state.predictions
    probabilities = st.session_state.probabilities
    aligned = st.session_state.audited_tpm
    sample_ids = predictions["SAMPID"].astype(str).tolist()
    active_sample = str(st.session_state.get("active_sample", sample_ids[0]))
    if active_sample not in sample_ids:
        active_sample = sample_ids[0]
        st.session_state.active_sample = active_sample

    prediction_row = predictions.loc[predictions["SAMPID"].eq(active_sample)].iloc[0]
    boosting_rounds = int(prediction_row["boosting_rounds"])
    total_rounds = int(classifier.get_booster().num_boosted_rounds())
    probability_row = probabilities.loc[active_sample].sort_values(ascending=False, kind="stable")
    predicted_tissue = str(prediction_row["predicted_tissue"])
    top_probability = float(prediction_row["top_probability"])
    runner_up_probability = float(probability_row.iloc[1])
    margin = top_probability - runner_up_probability
    ranked_cards = "".join(
        '<div class="probability-summary-item">'
        f'<div class="probability-summary-label">{rank_label}</div>'
        f'<div class="probability-summary-name">{html.escape(str(tissue))}</div>'
        f'<div class="probability-summary-value">{_probability_percent(float(probability))}</div></div>'
        for rank_label, (tissue, probability) in zip(
            ["Peringkat kedua", "Peringkat ketiga"], probability_row.iloc[1:3].items()
        )
    )

    st.markdown(f"### Hasil prediksi ({len(probability_row)} jaringan)")
    st.caption(f"Sampel: {active_sample} - {boosting_rounds} dari {total_rounds} putaran")
    st.markdown(
        '<div class="probability-summary">'
        '<div class="probability-summary-item winner"><div class="probability-summary-label">Prediksi terpilih</div>'
        f'<div class="result-name">{html.escape(predicted_tissue)}</div>'
        f'<div class="result-probability">{_probability_percent(top_probability)}</div></div>'
        f"{ranked_cards}</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Satu sampel memiliki probabilitas untuk seluruh kelas, dengan total sekitar 100%. "
        "Label prediksi dipilih dari probabilitas tertinggi. Angka ini bukan F1-score."
    )
    st.markdown("#### Probabilitas setiap kelas jaringan")
    st.altair_chart(_probability_chart(probability_row, predicted_tissue), width="stretch")
    st.caption(
        f"Sampel: {active_sample} - Selisih dua probabilitas tertinggi: "
        + f"{margin * 100:.2f}".replace(".", ",")
        + " poin persentase."
    )

    st.markdown("### Gen dengan kontribusi model tertinggi")
    st.caption(f"Kontribusi lokal untuk kelas {predicted_tissue} - {boosting_rounds} dari {total_rounds} putaran.")
    with st.spinner("Menghitung kontribusi gen..."):
        explanation = explain_prediction(
            aligned_tpm=aligned,
            sample_id=active_sample,
            preprocessor=preprocessor,
            classifier=classifier,
            limit=5,
            boosting_rounds=boosting_rounds,
        )
    shown = explanation[["rank", "gene", "raw_tpm", "contribution", "direction"]].copy()
    shown.columns = ["Peringkat", "ID Gen", "TPM sampel", "Kontribusi model", "Arah"]
    st.dataframe(
        shown,
        hide_index=True,
        width="stretch",
        height=206,
        column_config={
            "Peringkat": st.column_config.NumberColumn(format="%d", width="small"),
            "ID Gen": st.column_config.TextColumn(width="medium"),
            "TPM sampel": st.column_config.NumberColumn(format="%.4f"),
            "Kontribusi model": st.column_config.NumberColumn(format="%+.4f"),
            "Arah": st.column_config.TextColumn(width="medium"),
        },
    )
    with st.expander("Makna skor kontribusi"):
        st.write(
            "Nilai positif menaikkan skor internal kelas prediksi dan nilai negatif menurunkannya. Kontribusi dihitung pada skor sebelum softmax, bukan dalam satuan TPM atau persen probabilitas."
        )


def _render_prediction_panel(preprocessor: dict | None, classifier) -> None:
    """Show the run controls and the results column."""
    st.markdown('<div class="panel-heading">Prediksi jaringan</div>', unsafe_allow_html=True)
    audit = st.session_state.get("input_audit")
    aligned = st.session_state.get("audited_tpm")

    if preprocessor is None or classifier is None:
        st.error("Model lokal belum tersedia. Ekspor bundle model terlebih dahulu.")
        return

    # What the panel says depends on the audit.
    has_input = audit is not None and aligned is not None
    ready = has_input and audit.ready
    if not has_input:
        copy = "Prediksi tersedia setelah matriks lolos pemeriksaan."
        run_key = "run_prediction_disabled"
    elif not ready:
        copy = "Audit belum lolos. Model tidak dijalankan."
        run_key = "run_prediction_invalid"
    else:
        copy = "Model tersimpan - tanpa training ulang."
        run_key = "run_prediction"
    st.markdown(f'<div class="panel-copy">{copy}</div>', unsafe_allow_html=True)

    # Run controls.
    total_rounds = int(classifier.get_booster().num_boosted_rounds())
    choices = sorted({rounds for rounds in [100, 200, total_rounds] if rounds <= total_rounds})
    if st.session_state.get("prediction_boosting_rounds") not in choices:
        st.session_state.prediction_boosting_rounds = total_rounds
    with st.container(horizontal=True, vertical_alignment="bottom", gap="medium", key="prediction_controls"):
        boosting_rounds = st.selectbox(
            "Jumlah putaran boosting",
            choices,
            index=0,
            key="prediction_boosting_rounds",
            width=210,
            on_change=_clear_prediction_results,
            help="Gunakan putaran awal dari model tersimpan tanpa pelatihan ulang. Pilihan penuh memakai seluruh putaran model.",
        )
        # Widget state can reset after navigation, even without an on_change event.
        cached = st.session_state.get("predictions")
        if cached is not None and (
            "boosting_rounds" not in cached or not cached["boosting_rounds"].eq(boosting_rounds).all()
        ):
            _clear_prediction_results()
        run_clicked = st.button(
            "Jalankan ulang" if "predictions" in st.session_state else "Jalankan prediksi",
            icon=":material/play_arrow:",
            type="primary",
            disabled=not ready,
            key=run_key,
        )
        if ready and "predictions" in st.session_state:
            download_button(
                "Unduh hasil CSV",
                data=_prediction_export(),
                file_name="gtex_tissue_predictions.csv",
                mime="text/csv",
                icon=":material/download:",
                key="download_predictions",
            )
    st.caption(f"{boosting_rounds} dari {total_rounds} putaran model tersimpan. Tanpa pelatihan ulang.")

    # Stop here until there is a valid input.
    if not has_input:
        st.markdown(
            '<div class="empty-result"><strong>Menunggu input yang valid</strong>'
            "<span>Hasil, probabilitas, dan bukti model akan tampil di sini.</span></div>",
            unsafe_allow_html=True,
        )
        return

    if not ready:
        st.markdown(
            '<div class="empty-result"><strong>Perbaiki matriks input</strong>'
            "<span>Ikuti hasil pemeriksaan di panel sebelah kiri.</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Run the model.
    if run_clicked:
        _clear_prediction_results()
        try:
            with st.spinner("Menghitung prediksi jaringan..."):
                predictions, probabilities = predict_tissues(
                    aligned,
                    preprocessor,
                    classifier,
                    boosting_rounds=boosting_rounds,
                )
            st.session_state.predictions = predictions
            st.session_state.probabilities = probabilities
            st.rerun()
        except Exception as error:
            st.error(f"Prediksi tidak dapat dijalankan: {error}")

    st.divider()

    # Show the results, if there are any.
    if "predictions" not in st.session_state:
        with st.container(key="prediction_ready"):
            st.markdown(":material/description:")
            st.markdown("**Input siap diprediksi**")
            st.caption("Hasil akan tampil setelah model dijalankan.")
        return

    _render_prediction_result(preprocessor, classifier)


# --- Page ---


def render_prediction_page(tables: dict) -> None:
    """Show the Prediksi Jaringan page."""
    st.title("Prediksi jaringan")
    preprocessor, classifier = _choose_model()
    if preprocessor is None:
        return
    if preprocessor:
        selector_id = preprocessor.get("selector_id", "")
        selector_rows = tables["cv_summary"].loc[tables["cv_summary"]["selector_id"].eq(selector_id)]
        selector_text = (
            str(selector_rows.iloc[0]["selector_label"])
            if not selector_rows.empty
            else str(preprocessor.get("feature_selection", ""))
        )
        if preprocessor.get("feature_selection") == "l1":
            selector_text = f"L1 - C = {float(preprocessor['l1_c']):g}"
        model_text = (
            f"XGBoost - {selector_text} - "
            f"{_format_integer(preprocessor['feature_count'])} gen - "
            f"{len(preprocessor['classes'])} jaringan"
        )
    else:
        model_text = "Model lokal belum tersedia"

    st.markdown(
        '<div class="workspace-header">'
        '<div class="workspace-subtitle">Unggah TPM, periksa input, lalu jalankan prediksi.</div>'
        f'<div class="model-provenance">{html.escape(model_text)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    with st.container(key="prediction_workspace"):
        input_column, prediction_column = st.columns([1, 1.12], gap="large")
        with input_column:
            with st.container(key="prediction_input"):
                _render_input_panel(preprocessor)
        with prediction_column:
            with st.container(key="prediction_output"):
                _render_prediction_panel(preprocessor, classifier)
