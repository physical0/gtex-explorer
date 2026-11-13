"""Status proses: which stage of a training run is running, and on which data.

The stages carry the names of Alur proses pelatihan in the Cara kerja pelatihan guide, and only the stages
the run's settings use are shown. The worker reports a stage name (worker.py); STAGES places each one.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import streamlit as st

from gtex_explorer.display import RUN_STATUS
from gtex_explorer.ui.components.training_guide import FLOW

INIT, SPLIT, CV, FINAL, HOLDOUT, SAVE = (title for title, _, _ in FLOW)

# The steps inside a stage, in order.
SUBSTEPS = {
    CV: ["Pra-pemrosesan", "Seleksi fitur", "XGBoost", "Evaluasi"],
    FINAL: ["Pra-pemrosesan", "Seleksi fitur", "XGBoost"],
    SAVE: ["Menyimpan", "Verifikasi"],
}

# Where each stage the worker reports belongs, as (stage, step inside it).
STAGES = {
    "queued": (INIT, None),
    "launch": (INIT, None),
    "splitting_data": (SPLIT, None),
    "preparing_fold": (CV, 0),
    "fitting_selector": (CV, 1),
    "fitting_xgboost": (CV, 2),
    "final_preprocessing": (FINAL, 0),
    "final_selector": (FINAL, 1),
    "final_xgboost": (FINAL, 2),
    "saving_results": (SAVE, 0),
    "verifying_results": (SAVE, 1),
}

# The colour of a run's status on the current stage. Its name comes from RUN_STATUS, as in the run picker.
TONES = {
    "queued": "waiting",
    "running": "running",
    "paused": "paused",
    "cancelling": "paused",
    "cancelled": "stopped",
    "failed": "failed",
    "completed": "done",
}

STATE_TEXT = {"done": "selesai", "current": "tahap saat ini", "waiting": "belum dimulai"}


@dataclass(frozen=True)
class Step:
    title: str
    meta: str
    state: str  # done, current or waiting


@dataclass(frozen=True)
class RunProgress:
    steps: list[Step]
    label: str
    tone: str
    title: str | None  # the current stage, with its fold
    substeps: list[tuple[str, str]]  # (name, state)
    text: str | None  # the data the current stage trains or evaluates on
    rounds_label: str  # what the XGBoost progress bar counts


def run_progress(config: dict, status: str, details: dict) -> RunProgress:
    """Work out the stages of a run and where it is, from its settings, status and last reported stage."""
    cv = bool(config.get("evaluate_cv"))
    folds = config.get("cv_folds")
    holdout = config.get("holdout_percent") or 0
    fold = details.get("fold")
    order = [INIT, SPLIT, *([CV] if cv else []), FINAL, *([HOLDOUT] if holdout else []), SAVE]

    # A stopped run keeps the stage it stopped at.
    stage = details.get("stage")
    if status == "cancelled":
        stage = details.get("cancelled_stage", stage)
    elif status == "failed":
        stage = details.get("failed_stage", stage)
    current, step = STAGES.get(stage, (None, None))
    # A fit evaluates its validation data once every boosting round is done.
    total = details.get("rounds_total")
    if total and details.get("rounds_completed") == total and step == 2:
        if current == CV:
            step = 3
        elif holdout:
            current, step = HOLDOUT, None
        else:
            step = len(SUBSTEPS[FINAL])
    if status == "completed" or current not in order:
        current = None
    position = order.index(current) if current else (len(order) if status == "completed" else -1)

    def state(index, at):
        return "done" if index < at else "current" if index == at else "waiting"

    def meta(title, state):
        if title == CV:
            if state == "current" and fold:
                return f"Fold {fold} dari {folds}"
            return f"{folds} fold selesai" if state == "done" else f"{folds} fold"
        return {
            INIT: "Worker latar belakang",
            SPLIT: f"Holdout {holdout}%" if holdout else "Tanpa holdout",
            FINAL: "Data development" if holdout else "Seluruh sampel",
            HOLDOUT: "Data holdout",
            SAVE: "Gallery model",
        }[title]

    steps = []
    for index, title in enumerate(order):
        step_state = state(index, position) if position >= 0 else "waiting"
        steps.append(Step(title, meta(title, step_state), step_state))

    label, tone = RUN_STATUS.get(status, (None, status))[1], TONES.get(status, "waiting")
    title, text = None, None
    if status == "completed":
        title = "Semua tahap selesai"
    elif current == INIT:
        title, text = INIT, "Data dan konfigurasi tersimpan; worker memulai pelatihan."
    elif current == SPLIT:
        title = SPLIT
        text = (
            "Sampel dibagi per donor menjadi data development dan data holdout."
            if holdout
            else "Seluruh sampel menjadi data development."
        )
    elif current == CV:
        title = f"{CV} · fold {fold} dari {folds}" if fold else CV
        text = f"Dilatih pada data training fold {fold}, lalu dievaluasi pada data validation fold {fold}." if fold else None
    elif current == FINAL:
        title = FINAL
        text = "Dilatih pada data development; data holdout belum digunakan." if holdout else "Dilatih pada seluruh sampel."
    elif current == HOLDOUT:
        title, text = HOLDOUT, "Model final memprediksi data holdout tanpa diubah."
    elif current == SAVE:
        title = SAVE
    substeps = [(name, state(index, step)) for index, name in enumerate(SUBSTEPS.get(current, []))]
    rounds_label = f"Putaran XGBoost fold {fold}" if current == CV and fold else (
        "Putaran XGBoost model final" if current in {FINAL, HOLDOUT} else "Putaran XGBoost"
    )
    return RunProgress(steps, label, tone, title, substeps, text, rounds_label)


def _hidden(state: str) -> str:
    """The state of a step, for screen readers."""
    return f'<span class="run-hidden"> ({STATE_TEXT[state]})</span>'


def progress_html(progress: RunProgress) -> str:
    """The stages as a row of steps, then a card for the current one."""
    steps = "".join(
        f'<li class="run-step is-{step.state}"' + (' aria-current="step"' if step.state == "current" else "") + ">"
        '<span class="run-marker" aria-hidden="true"></span>'
        f'<p class="run-step-title">{html.escape(step.title)}{_hidden(step.state)}</p>'
        f'<p class="run-step-meta">{html.escape(step.meta)}</p></li>'
        for step in progress.steps
    )
    substeps = "".join(
        f'<li class="is-{state}">{html.escape(name)}{_hidden(state)}</li>' for name, state in progress.substeps
    )
    activity = (
        '<div class="run-activity"><p class="run-activity-head">'
        f'<span class="run-pill">{html.escape(progress.label)}</span>'
        + (f"<span>{html.escape(progress.title)}</span>" if progress.title else "")
        + "</p>"
        + (f'<ol class="run-substeps" aria-label="Langkah tahap ini">{substeps}</ol>' if substeps else "")
        + (f'<p class="run-activity-text">{html.escape(progress.text)}</p>' if progress.text else "")
        + "</div>"
    )
    return (
        f'<section class="run-progress tone-{progress.tone}" aria-label="Tahapan pelatihan">'
        f'<ol class="run-steps">{steps}</ol>{activity}</section>'
    )


def render_run_progress(progress: RunProgress) -> None:
    """Show the stages of a run and the one it is at."""
    st.markdown(progress_html(progress), unsafe_allow_html=True)
