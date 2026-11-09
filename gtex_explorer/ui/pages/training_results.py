"""Hasil Pelatihan page: follow a training run and browse the finished models."""

import json
import time
from pathlib import Path

import altair as alt
import pandas as pd
import psutil
import streamlit as st

from gtex_explorer.platform.cloud.media import download_button
from gtex_explorer.training.gallery import list_gallery
from gtex_explorer.training.jobs import (
    ACTIVE_STATUSES,
    FINALIZING_STAGES,
    can_resume,
    delete_run,
    has_active_run,
    pause_job,
    read_job,
    request_cancel,
    resume_job,
    unpause_job,
)
from gtex_explorer.training.run_labels import method_description, run_created, run_label
from gtex_explorer.ui.components.training_runs import choose_run, show_diagnostics
from gtex_explorer.ui.shell.error_messages import user_error
from gtex_explorer.ui.shell.theme import page_style

METRICS = {
    "macro_f1": "Macro-F1",
    "macro_precision": "Macro-precision",
    "macro_recall": "Macro-recall",
    "balanced_accuracy": "Balanced accuracy",
    "accuracy": "Accuracy",
}


# --- Run status ---


def _follow_active_run(state_root, polling):
    """Redraw the page when a run starts or ends, so polling is on only while one is active."""
    if has_active_run(state_root) != polling:
        st.rerun(scope="app")


def _run_details(data_root: Path, state_root: Path, polling: bool = False):
    """Show the progress and results of the chosen run."""
    _follow_active_run(state_root, polling)
    gallery_only = st.session_state.get("training_results_view") == "Gallery model"
    run_id = choose_run(state_root, "training_run", gallery_only=gallery_only)
    if run_id is None:
        return
    try:
        # Read the saved run.
        root = state_root / "runs" / run_id
        saved = json.loads((root / "config.json").read_text(encoding="utf-8"))
        record = read_job(run_id, state_root)
        config = saved["config"]
        dataset = saved.get("dataset")
        st.subheader(method_description(config))
        if dataset:
            source = dataset["source"]
            st.caption(
                f"{source.get('expression_file', 'Data notebook')} - {dataset['sample_count']} sampel - "
                f"{dataset['donor_count']} donor - {dataset['class_count']} jaringan"
            )

        # Progress.
        st.subheader("Status proses")
        labels = {
            "queued": "Menunggu worker",
            "preparing_fold": "Persiapan fold",
            "fitting_selector": "Seleksi fitur",
            "fitting_xgboost": "Pelatihan XGBoost",
            "final_selector": (
                "Seleksi fitur pada data development"
                if config.get("holdout_percent")
                else "Seleksi fitur pada seluruh data development"
            ),
            "final_xgboost": "Pelatihan model final",
            "completed": "Selesai",
            "saving_results": "Menyimpan model dan hasil",
            "verifying_results": "Memverifikasi model, metrik, dan panel gen",
            "cancelled": "Dihentikan",
            "paused": "Dijeda",
            "cancelling": "Menghentikan pelatihan",
            "failed": "Gagal",
        }
        # A resumed run keeps the stage it stopped at until the worker reports a new one.
        display_stage = (
            record.status
            if record.status in {"paused", "cancelling", "queued", "cancelled"}
            else record.details.get("stage")
        )
        elapsed = record.details.get("elapsed_seconds", 0)
        if record.status in {"running", "queued"}:
            elapsed += max(0, time.time() - record.details["updated"])
        stage_column, time_column, rounds_column = st.columns([1, 1, 1.5])
        stage_column.markdown("**" + labels.get(display_stage, record.status) + "**")
        if record.details.get("fold"):
            stage_column.caption(f"Fold {record.details['fold']} dari {config['cv_folds']}")
        time_column.caption(f"Waktu proses: {elapsed:.0f} detik")
        rounds_column.caption(f"Target XGBoost: {config['xgboost']['n_estimators']} putaran per fit.")
        completed_rounds = record.details.get("rounds_completed")
        total_rounds = record.details.get("rounds_total")
        if completed_rounds is not None and total_rounds:
            prefix = "Putaran terakhir tercatat" if record.status == "cancelled" else "Putaran XGBoost pada fit ini"
            st.progress(
                int(round(min(1.0, max(0.0, completed_rounds / total_rounds)) * 100)),
                text=f"{prefix}: {completed_rounds} dari {total_rounds}",
            )
        if record.status == "cancelled":
            st.info(
                "Pelatihan dihentikan. Fold yang sudah selesai tetap tersimpan. "
                "Jika dilanjutkan, fit yang terhenti akan diulang."
            )
        if record.status == "paused":
            st.info("Proses berhenti sementara dan tetap tersimpan di memori. Resume meneruskan fit yang sama.")
            st.caption("Pause aktif selama worker masih hidup. Menutup atau memulai ulang komputer mengakhiri proses.")

        # Controls while the run is active.
        if record.status in {"running", "queued", "cancelling", "paused"}:
            cancelling = (root / "cancel.json").exists()
            finalizing = record.status == "running" and record.details.get("stage") in FINALIZING_STAGES
            if finalizing:
                st.info("Hasil sedang disimpan dan diverifikasi. Model akan masuk gallery setelah proses ini selesai.")
            if cancelling:
                st.info("Permintaan berhenti diterima. Menunggu worker menghentikan pelatihan.")
            pause_column, stop_column = st.columns(2)
            with pause_column:
                if record.status == "paused":
                    if st.button(
                        "Resume", key="training_unpause", type="primary", width="stretch", icon=":material/play_arrow:"
                    ):
                        try:
                            unpause_job(run_id, state_root)
                            st.rerun()
                        except (ValueError, OSError, psutil.Error) as error:
                            st.error(user_error(error))
                elif st.button(
                    "Pause",
                    key="training_pause",
                    type="primary",
                    width="stretch",
                    icon=":material/pause:",
                    disabled=finalizing or record.status != "running" or not (root / "worker.json").exists(),
                ):
                    try:
                        pause_job(run_id, state_root)
                        st.rerun()
                    except (ValueError, OSError, psutil.Error) as error:
                        st.error(user_error(error))
            with stop_column:
                if st.button(
                    "Stop",
                    key="training_cancel",
                    disabled=cancelling or finalizing,
                    width="stretch",
                    icon=":material/stop:",
                    help="Akhiri worker. Fold yang selesai tetap tersimpan, fit yang terputus dibuang.",
                ):
                    try:
                        request_cancel(run_id, state_root)
                        st.rerun()
                    except (ValueError, OSError, psutil.Error) as error:
                        st.error(user_error(error))

        # Failure, and whether the run can continue.
        if record.error:
            st.error(user_error(record.error["message"]))
            if record.error.get("diagnostics"):
                show_diagnostics(record.error["diagnostics"])
        if record.status == "failed" and not can_resume(record):
            st.caption("Kegagalan ini berasal dari data atau konfigurasi. Ubah pengaturan lalu latih run baru.")
        if can_resume(record):
            if st.button("Continue from checkpoint", key="training_resume", icon=":material/restart_alt:"):
                try:
                    resume_job(run_id, data_root, state_root)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(user_error(error))

        # Settings and scores.
        _show_saved_parameters(saved)
        summary = record.details.get("cv_summary")
        st.divider()
        if record.details.get("holdout_summary"):
            _show_holdout_results(record.details["holdout_summary"])
        if summary:
            _show_cv_results(summary, config)
        elif not record.details.get("holdout_summary") and record.status == "completed":
            st.info("Training selesai tanpa CV. Tidak ada skor validation atau holdout baru.")

        # Actions for a finished model.
        ready = record.status == "completed" and any(entry["run_id"] == run_id for entry in list_gallery(state_root))
        if record.status == "completed" and not ready:
            st.warning(
                "Hasil belum lolos verifikasi gallery. Periksa berkas model, metrik, dan panel gen yang tersimpan."
            )
        if not ready and record.status not in ACTIVE_STATUSES:
            _delete_button(run_id, "Hapus hasil pelatihan")
        if ready:
            st.divider()
            st.subheader("Model final tersimpan")
            st.markdown("**" + method_description(config) + " - XGBoost**")
            panel_note = (
                "Panel final dan panel tiap fold tersedia pada Panel Gen."
                if config["evaluate_cv"]
                else "Panel final tersedia pada Panel Gen."
            )
            st.caption(
                (
                    "Dilatih pada data development. Data holdout hanya dipakai untuk menguji. "
                    if config.get("holdout_percent")
                    else "Dilatih pada seluruh dataset training ini. "
                )
                + panel_note
            )
            _show_model_actions(run_id)
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f"Hasil pelatihan tidak dapat dibaca: {user_error(error)}")


# --- Removing a run ---


def _delete_button(run_id, label):
    """Show the button that opens the delete dialog."""
    if st.button(label, key="training_delete_run", icon=":material/delete:", wrap=True):
        st.session_state.training_delete_target = run_id
        # Dialogs open from a full page run, not from inside these panels.
        st.rerun(scope="app")


def _dismiss_delete():
    """Close the delete dialog."""
    st.session_state.pop("training_delete_target", None)


@st.dialog("Hapus hasil pelatihan", on_dismiss=_dismiss_delete)
def _delete_run_dialog(state_root, run_id):
    """Ask for confirmation before deleting a run."""
    st.write(run_label(Path(state_root) / "runs" / run_id))
    st.warning(
        "Hasil pelatihan, model, dan panel gennya akan dihapus permanen. Tindakan ini tidak dapat dibatalkan. "
        "Dataset unggahan tetap tersimpan sampai workspace dihapus."
    )
    confirm, cancel = st.columns(2)
    if confirm.button(
        "Hapus permanen", key="training_delete_confirm", type="primary", icon=":material/delete:", width="stretch"
    ):
        try:
            delete_run(run_id, state_root)
            _dismiss_delete()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(user_error(error))
    if cancel.button("Batal", key="training_delete_cancel", width="stretch"):
        _dismiss_delete()
        st.rerun()


# --- Result sections ---


def _show_saved_parameters(saved):
    """Show the data and settings a run used."""
    with st.expander("Data dan parameter tersimpan"):
        dataset = saved.get("dataset")
        if dataset:
            st.write("Metadata: " + dataset.get("source", {}).get("metadata_file", "Metadata notebook"))
        if saved.get("data_fingerprint"):
            st.caption("Identitas dataset: " + saved["data_fingerprint"][:16])
        st.dataframe(
            pd.DataFrame(
                [{"Parameter": key, "Nilai": str(value)} for key, value in saved["config"]["xgboost"].items()]
            ),
            hide_index=True,
        )


def _show_model_actions(run_id):
    """Show the buttons for using a finished model."""
    with st.container(horizontal=True, gap="small"):
        if st.button(
            "Gunakan untuk Prediksi Jaringan",
            key="training_open_prediction",
            type="primary",
            icon=":material/biotech:",
            wrap=True,
        ):
            # The prediction page loads this model and invalidates previous model results.
            st.session_state.prediction_model = run_id
            for key in ["input_audit", "predictions", "probabilities", "prediction_rounds"]:
                st.session_state.pop(key, None)
            st.session_state.page = "Prediksi Jaringan"
            st.rerun()
        if st.button("Buka Panel Gen", key="training_open_genes", icon=":material/table_view:", wrap=True):
            st.session_state.training_panel_run = run_id
            st.session_state.page = "Panel Gen Pelatihan"
            st.rerun()
        _delete_button(run_id, "Hapus model")


def _show_cv_results(summary, config):
    """Show the cross-validation scores."""
    st.subheader("Evaluasi CV")
    st.caption(
        f"Rerata dan simpangan baku dari {summary.get('fold_count', config['cv_folds'])} fold validation."
        + (" Fold hanya dibentuk dari data development." if config.get("holdout_percent") else "")
    )
    with st.container(key="training_cv_results"):
        metrics_column, summary_column = st.columns([3, 1], gap="medium")
        metrics_column.dataframe(
            pd.DataFrame(
                {
                    "Metrik": list(METRICS.values()),
                    "Rerata": [f"{summary['mean_' + key]:.2%}".replace(".", ",") for key in METRICS],
                    "SD (poin persentase)": [f"{summary['std_' + key] * 100:.2f}".replace(".", ",") for key in METRICS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        summary_column.metric("Rerata Macro-F1", f"{summary['mean_macro_f1']:.2%}".replace(".", ","))
        if summary["mean_jaccard"] is not None:
            summary_column.metric("Jaccard antar-fold", f"{summary['mean_jaccard']:.2%}".replace(".", ","))
            summary_column.caption("Kesamaan gen terpilih")
    folds = pd.DataFrame(summary["fold_results"]).drop(columns=["selected_genes", "tissue_f1"], errors="ignore")
    with st.container(horizontal=True, vertical_alignment="top"):
        with st.expander("Hasil setiap fold"):
            st.dataframe(
                folds.rename(columns=METRICS).style.format({label: "{:.2%}" for label in METRICS.values()}),
                hide_index=True,
            )
        download_button(
            "Unduh hasil fold",
            folds.to_csv(index=False),
            "cv_folds.csv",
            key="training_download_folds",
            icon=":material/download:",
        )
    _show_tissue_f1(summary)


def _tissue_chart(tissues, tooltip):
    """Bars of `score`, lowest first. Names sit above the bars, so long names leave them their full width on small screens."""
    tissues = tissues.sort_values(["score", "tissue"], kind="stable")
    base = alt.Chart(tissues).encode(
        y=alt.Y("tissue:N", sort=tissues["tissue"].tolist(), title=None, axis=None),
        tooltip=[alt.Tooltip("tissue:N", title="Jaringan"), *tooltip],
    )
    bars = base.mark_bar(color="#0f8b8d", size=10, yOffset=9).encode(
        x=alt.X(
            "score:Q",
            title="F1-score",
            scale=alt.Scale(domain=[0, 1], nice=False),
            axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".0%", gridColor="#e2e8ec"),
        )
    )
    names = base.mark_text(align="left", dy=-8, fontSize=13, color="#344054").encode(x=alt.XDatum(0), text="tissue:N")
    values = base.mark_text(align="right", dy=-8, fontSize=13, color="#344054").encode(x=alt.XDatum(1), text="label:N")
    st.altair_chart(
        (bars + names + values)
        .properties(height=len(tissues) * 42)
        .configure_view(stroke=None)
        .configure_axis(labelColor="#657184", labelFont="Segoe UI", titleFont="Segoe UI"),
        width="stretch",
        theme=None,
    )


def _show_tissue_f1(summary):
    """Show the mean F1-score of each tissue across folds."""
    st.subheader("F1-score per jaringan")
    if "tissue_f1" not in summary:
        st.caption("Model ini dilatih sebelum F1-score per jaringan tersedia. Latih ulang model untuk melihatnya.")
        return
    st.caption("Rerata ± SD dari setiap fold validation, diurutkan dari skor terendah.")
    tissues = pd.DataFrame(summary["tissue_f1"]).sort_values(["mean_f1", "tissue"], kind="stable")
    tissues["score"] = tissues["mean_f1"]
    tissues["label"] = [
        f"{mean:.2%} ± {std * 100:.2f}".replace(".", ",") for mean, std in zip(tissues["mean_f1"], tissues["std_f1"])
    ]
    _tissue_chart(
        tissues,
        [
            alt.Tooltip("mean_f1:Q", title="Rerata F1-score", format=".2%"),
            alt.Tooltip("std_f1:Q", title="SD", format=".2%"),
        ],
    )
    table = tissues[["tissue", "mean_f1", "std_f1"]].copy()
    for row in summary["fold_results"]:
        table[f"fold_{row['fold']}"] = table["tissue"].map(row["tissue_f1"])
    deviation = "SD (poin persentase)"
    shown = table.rename(
        columns=lambda name: {"tissue": "Jaringan", "mean_f1": "Rerata", "std_f1": deviation}.get(
            name, name.replace("fold_", "Fold ")
        )
    )
    shown[deviation] *= 100
    with st.container(horizontal=True, vertical_alignment="top"):
        with st.expander("Nilai setiap jaringan"):
            st.dataframe(
                shown.style.format(
                    lambda value: f"{value:.2%}".replace(".", ","),
                    subset=[name for name in shown if name not in ("Jaringan", deviation)],
                ).format(lambda value: f"{value:.2f}".replace(".", ","), subset=[deviation]),
                hide_index=True,
            )
        download_button(
            "Unduh F1 per jaringan",
            table.to_csv(index=False),
            "cv_tissue_f1.csv",
            key="training_download_tissue_f1",
            icon=":material/download:",
        )


def _show_holdout_results(summary):
    """Show the holdout scores."""
    percent = lambda value: f"{value:.2%}".replace(".", ",")
    st.subheader("Evaluasi holdout")
    st.caption(
        f"Model yang disimpan dilatih pada {summary['training_samples']} sampel data development "
        f"({summary['training_donors']} donor) dan diuji pada {summary['validation_samples']} sampel data holdout "
        f"({summary['validation_donors']} donor). Donor tidak terbagi di antara keduanya."
    )
    with st.container(key="training_holdout_results"):
        metrics_column, summary_column = st.columns([3, 1], gap="medium")
        metrics_column.dataframe(
            pd.DataFrame({"Metrik": list(METRICS.values()), "Nilai": [percent(summary[key]) for key in METRICS]}),
            hide_index=True,
            width="stretch",
        )
        summary_column.metric("Macro-F1", percent(summary["macro_f1"]))
    st.subheader("F1-score per jaringan")
    st.caption("Skor pada data holdout, diurutkan dari skor terendah. n = jumlah sampel holdout.")
    tissues = pd.DataFrame({"tissue": list(summary["tissue_f1"]), "score": list(summary["tissue_f1"].values())})
    tissues["samples"] = tissues["tissue"].map(summary["tissue_support"])
    tissues["label"] = [
        f"{percent(score)} (n = {samples})" for score, samples in zip(tissues["score"], tissues["samples"])
    ]
    _tissue_chart(
        tissues,
        [
            alt.Tooltip("score:Q", title="F1-score", format=".2%"),
            alt.Tooltip("samples:Q", title="Sampel holdout", format=".0f"),
        ],
    )
    download_button(
        "Unduh F1 per jaringan",
        tissues[["tissue", "score", "samples"]].rename(columns={"score": "f1"}).to_csv(index=False),
        "holdout_tissue_f1.csv",
        key="training_download_holdout_f1",
        icon=":material/download:",
    )


# --- Model gallery ---


def _evaluation_label(config):
    """Describe how a model was evaluated."""
    parts = [f"Holdout {config['holdout_percent']}%"] if config.get("holdout_percent") else []
    if config["evaluate_cv"]:
        parts.append(f"{config['cv_folds']} fold")
    return " dan ".join(parts) or "Tanpa CV"


def _select_gallery_run(run_id):
    """Open a model from the gallery."""
    st.session_state.training_gallery_run = run_id
    st.session_state.training_selected_run = run_id


def _gallery_results(state_root, polling: bool = False):
    """Show the list of models and the chosen one."""
    _follow_active_run(state_root, polling)
    issues = []
    entries = list_gallery(state_root, issues=issues)
    for issue in issues:
        st.warning(f"Hasil {issue['run_id'][-6:]} belum lolos verifikasi: {user_error(issue['reason'])}")
    if not entries:
        st.info("Belum ada model terverifikasi di gallery. Periksa Status proses.")
        return
    try:
        records = []
        for entry in entries:
            path = Path(state_root) / "runs" / entry["run_id"] / "config.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            records.append(
                {
                    **entry,
                    "saved": saved,
                    "method": method_description(saved["config"]),
                    "filename": saved.get("dataset", {})
                    .get("source", {})
                    .get("expression_file", "Data eksperimen notebook"),
                    "date": time.strftime("%d/%m/%Y %H:%M", time.localtime(run_created(path.parent))),
                }
            )
        library, details = st.columns([1, 2.6], gap="large")
        with library, st.container(key="gallery_models"):
            st.subheader("Model tersimpan")
            query = (
                st.text_input(
                    "Cari model atau dataset",
                    placeholder="Cari model atau dataset",
                    key="training_gallery_search",
                    label_visibility="collapsed",
                )
                .strip()
                .casefold()
            )
            visible = [
                record for record in records if query in (record["method"] + " " + record["filename"]).casefold()
            ]
            if not visible:
                st.info("Tidak ada model yang cocok dengan pencarian.")
                return
            selected = st.session_state.get("training_gallery_run", st.session_state.get("training_selected_run"))
            if selected not in {record["run_id"] for record in visible}:
                selected = visible[0]["run_id"]
                _select_gallery_run(selected)
            for record in visible:
                label = f"**{record['method']}**  \n{record['filename']}  \n{record['date']}"
                st.button(
                    label,
                    key="gallery_select_" + record["run_id"],
                    type="primary" if record["run_id"] == selected else "tertiary",
                    width="stretch",
                    wrap=True,
                    on_click=_select_gallery_run,
                    args=(record["run_id"],),
                )
        record = next(record for record in visible if record["run_id"] == selected)
        saved = record["saved"]
        config = saved["config"]
        with details, st.container(key="gallery_detail"):
            with st.container(horizontal=True, vertical_alignment="center", gap="small"):
                st.subheader(record["method"] + ", XGBoost", width="content")
                st.badge("Tersimpan dan terverifikasi", icon=":material/check:", color="green")
            evaluation = _evaluation_label(config)
            context = [f"{config['xgboost']['n_estimators']} putaran", evaluation]
            dataset = saved.get("dataset", {})
            for key, label in [("sample_count", "sampel"), ("donor_count", "donor"), ("class_count", "jaringan")]:
                if key in dataset:
                    context.append(f"{dataset[key]:,} {label}".replace(",", "."))
            st.write(" - ".join(context))
            st.caption("Dataset: " + record["filename"])
            _show_model_actions(selected)
            st.divider()
            if record.get("holdout_summary"):
                _show_holdout_results(record["holdout_summary"])
            if record["cv_summary"]:
                _show_cv_results(record["cv_summary"], config)
            elif not record.get("holdout_summary"):
                st.caption("Tanpa CV. Tidak ada skor validation atau holdout baru.")
            _show_saved_parameters(saved)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Hasil pelatihan tidak dapat dibaca: {user_error(error)}")


# --- Page ---


def render_training_results_page(data_root: Path, state_root: Path):
    """Show the Hasil Pelatihan page."""
    st.markdown(page_style("training_results"), unsafe_allow_html=True)
    with st.container(key="training_results"):
        st.title("Hasil pelatihan")
        st.caption("Hasil model yang dilatih pengguna, terpisah dari eksperimen notebook.")
        if "training_results_view" not in st.session_state:
            st.session_state.training_results_view = "Gallery model" if list_gallery(state_root) else "Status proses"
        gallery, process = st.tabs(["Gallery model", "Status proses"], key="training_results_view", on_change="rerun")
        # Saved results do not change, so these panels poll only while a run is active.
        polling = has_active_run(state_root)
        run_every = 2 if polling else None
        if gallery.open:
            with gallery:
                st.fragment(_gallery_results, run_every=run_every)(state_root, polling)
        if process.open:
            with process:
                st.fragment(_run_details, run_every=run_every)(data_root, state_root, polling)
        target = st.session_state.get("training_delete_target")
        if target:
            if (Path(state_root) / "runs" / target).is_dir():
                _delete_run_dialog(state_root, target)
            else:
                _dismiss_delete()
