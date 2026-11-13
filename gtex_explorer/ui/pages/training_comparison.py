"""Perbandingan Model page: saved training models side by side."""

import math
import time

import altair as alt
import pandas as pd
import streamlit as st

from gtex_explorer.display import METRICS, SERIES_COLORS, count, percent, plain, points
from gtex_explorer.platform.cloud.media import download_button
from gtex_explorer.training.comparison import (
    BASES,
    LETTERS,
    MAX_MODELS,
    available_bases,
    candidate_note,
    comparability_issues,
    default_selection,
    final_panels,
    gallery_models,
    metric_frame,
    panel_membership,
    panel_overlap,
    settings_frame,
    shared_data_note,
    source_name,
    tissue_frame,
    unscored_note,
)
from gtex_explorer.training.run_labels import evaluation_label, method_description, preprocessing_label
from gtex_explorer.ui.shell.error_messages import user_error

# One series colour per model letter, the same palette as the method comparison on Ringkasan.
COLORS = SERIES_COLORS[:4]
HIGHLIGHT = "background-color: #dff3f1"
# Room for the x axis and its title, which the chart height includes.
AXIS_HEIGHT = 56
SELECTION_KEY = "training_compare_runs"
BASIS_KEY = "training_compare_basis"


def _date(model):
    return time.strftime("%d/%m/%Y %H:%M", time.localtime(model["created"]))


def _option_label(model):
    """A short picker label: the run's name, method, evaluation, dataset, start time, and run id.

    The picker sends its choice back as this label, so the run id keeps two otherwise identical runs apart.
    """
    stamp = time.strftime("%d/%m %H:%M", time.localtime(model["created"]))
    label = (
        f"{model['method']} · {evaluation_label(model['config'])} · {model['filename']} · {stamp} · "
        f"{model['run_id'][-6:]}"
    )
    return f"{model['name']} · {label}" if model.get("name") else label


def _marker(letter):
    """The coloured dot that ties a model to its colour in the chart."""
    return f'<span style="color:{COLORS[LETTERS.index(letter)]}">&#9679;</span> **Model {letter}**'


def _legend(letters):
    """The chart key as text, so it wraps on narrow screens instead of running past the chart."""
    st.markdown("&emsp;".join(_marker(letter) for letter in letters), unsafe_allow_html=True)


def _highlighted(table, mask):
    """Colour the cells where `mask` is true; `mask` covers some of the table's columns."""
    css = pd.DataFrame("", index=table.index, columns=table.columns)
    for column in mask:
        css.loc[mask[column].to_numpy(), column] = HIGHLIGHT
    return table.style.apply(lambda _: css, axis=None)


def _best(scores):
    """The cells holding the top score of their row. A row where every model scores the same has none."""
    best = scores.eq(scores.max(axis=1), axis=0)
    best.loc[(best | scores.isna()).all(axis=1)] = False
    return best


# --- Choosing what to compare ---


def _choose_models(models):
    """Show the model picker and return the chosen models in the order picked."""
    by_id = {model["run_id"]: model for model in models}
    chosen = st.session_state.get(SELECTION_KEY)
    # Models deleted or from another workspace drop out of a remembered choice.
    st.session_state[SELECTION_KEY] = (
        default_selection(models) if chosen is None else [run for run in chosen if run in by_id][:MAX_MODELS]
    )
    selection = st.multiselect(
        "Model yang dibandingkan",
        list(by_id),
        format_func=lambda run: _option_label(by_id[run]),
        key=SELECTION_KEY,
        max_selections=MAX_MODELS,
        placeholder="Pilih dua sampai empat model",
        wrap=True,
        persist_state="session",
    )
    return [by_id[run] for run in selection]


def _choose_basis(models):
    """Show the evaluation choice and return it, or None when no two models share one."""
    bases = available_bases(models)
    if not bases:
        return None
    if st.session_state.get(BASIS_KEY) not in bases:
        st.session_state[BASIS_KEY] = bases[0]
    return st.segmented_control(
        "Dasar perbandingan",
        bases,
        format_func=BASES.get,
        key=BASIS_KEY,
        required=True,
        persist_state="session",
    )


def _comparability_note(models, basis):
    """Say whether the scores were measured on the same samples, with the details in a closed expander.

    A mismatch stays in view as a warning. When the data match, the expander header keeps the verdict, the
    dataset, and the evaluation in view, and the notes behind them open on request.
    """
    issues = comparability_issues(models, basis)
    notes = [unscored_note(models, basis), candidate_note(models, basis)]
    if issues:
        st.warning(
            "Skor tidak diukur pada sampel yang sama, jadi selisihnya tidak dapat dibaca sebagai perbedaan metode. "
            + " ".join(issues),
            icon=":material/warning:",
        )
        label = "Catatan perbandingan"
    else:
        notes.insert(0, shared_data_note(models, basis))
        first = next(model for model in models if model[basis])
        holdout = first["config"].get("holdout_percent")
        split = f"Holdout {holdout}%" if basis == "holdout" else f"CV {first['config']['cv_folds']} fold"
        label = (
            ":green-badge[:material/check: Data evaluasi sama] "
            f"{plain(source_name(first['filename']))} · {split}"
        )
    notes = [note for note in notes if note]
    if notes:
        with st.expander(label):
            for note in notes:
                st.caption(note)


# --- Sections ---


def _model_cards(models, basis):
    """One card per model: its method, data, and Macro-F1 against the best."""
    scores = metric_frame(models, basis).query("metric == 'Macro-F1'").set_index("model") if basis else None
    best = scores["score"].max() if basis else None
    # When every model has the same score, no card is marked as the highest.
    tied = basis and scores["score"].dropna().nunique() <= 1
    for letter, model, column in zip(LETTERS, models, st.columns(len(models), border=True)):
        with column:
            st.markdown(_marker(letter), unsafe_allow_html=True)
            # Every card names its transforms, the default too, so they can be compared at a glance.
            config = model["config"]
            method = method_description(config, with_preprocessing=False)
            # A named run leads with its name, and its method moves to the line below.
            st.markdown(f"**{plain(model['name']) if model.get('name') else method}**")
            lines = [method] if model.get("name") else []
            lines += [model["filename"], f"{preprocessing_label(config)} - {evaluation_label(config)} - {_date(model)}"]
            st.caption("  \n".join(lines))
            if not basis:
                continue
            score, sd = scores.loc[letter, ["score", "sd"]]
            if pd.isna(score):
                st.metric("Macro-F1", "-")
                continue
            gap = best - score
            if math.isclose(gap, 0, abs_tol=1e-12):
                st.metric("Macro-F1", percent(score))
                if not tied:
                    st.badge("Tertinggi", icon=":material/star:", color="green")
            else:
                st.metric(
                    "Macro-F1",
                    percent(score),
                    delta=f"-{points(gap)} poin",
                    delta_description="dari skor tertinggi",
                )
            if basis == "cv":
                lines = [f"SD antar-fold {points(sd)} poin"]
                if model["cv"].get("mean_jaccard") is not None:
                    lines.append(f"Jaccard antar-fold {percent(model['cv']['mean_jaccard'])}")
                st.caption("  \n".join(lines))


def _metric_section(models, basis):
    """All metrics side by side, the best of each row coloured."""
    st.subheader("Metrik evaluasi")
    st.caption(
        (
            "Rerata ± SD (poin persentase) antar-fold validation."
            if basis == "cv"
            else "Skor model final pada data holdout."
        )
        + " Sel berwarna: skor tertinggi pada baris itu, bila skor model berbeda."
    )
    frame = metric_frame(models, basis)
    letters = list(LETTERS[: len(models)])
    scores = frame.pivot(index="metric", columns="model", values="score").reindex(
        index=list(METRICS.values()), columns=letters
    )
    shown = scores.map(percent)
    if basis == "cv":
        sds = frame.pivot(index="metric", columns="model", values="sd").reindex(index=scores.index, columns=letters)
        shown = shown + sds.map(lambda value: "" if pd.isna(value) else " ± " + points(value))
    names = {letter: f"Model {letter}" for letter in letters}
    table = shown.rename(columns=names).reset_index(names="Metrik")
    best = _best(scores).rename(columns=names).reset_index(drop=True)
    st.dataframe(_highlighted(table, best), hide_index=True, width="stretch")
    export = frame.assign(
        method=frame["model"].map({letter: model["method"] for letter, model in zip(LETTERS, models)}),
        dataset=frame["model"].map({letter: model["filename"] for letter, model in zip(LETTERS, models)}),
        run_id=frame["model"].map({letter: model["run_id"] for letter, model in zip(LETTERS, models)}),
        evaluation=basis,
    )
    download_button(
        "Unduh perbandingan metrik",
        export.to_csv(index=False),
        f"perbandingan_metrik_{basis}.csv",
        key="training_compare_download_metrics",
        icon=":material/download:",
    )


def _tissue_section(models, basis):
    """F1-score of each tissue for every model, weakest tissues first."""
    st.subheader("F1-score per jaringan")
    tissues = tissue_frame(models, basis)
    if tissues.empty:
        st.caption("Model ini dilatih sebelum F1-score per jaringan tersedia. Latih ulang model untuk melihatnya.")
        return
    letters = [letter for letter in LETTERS[: len(models)] if letter in set(tissues["model"])]
    missing = [letter for letter, model in zip(LETTERS, models) if model[basis] and letter not in letters]
    if missing:
        st.caption(f"Model {', '.join(missing)} dilatih sebelum F1-score per jaringan tersedia.")
    st.caption(
        "Diurutkan dari jaringan dengan skor terendah pada salah satu model."
        + (" Garis: ±1 SD antar-fold." if basis == "cv" else " n = jumlah sampel holdout.")
    )
    zoom = st.checkbox("Perbesar rentang skor pada grafik", value=True, key="training_compare_zoom")

    # Tissues sorted by their weakest model, ties by name.
    order = tissues.groupby("tissue")["f1"].min().sort_values(kind="stable").index.tolist()
    tissues["low"] = (tissues["f1"] - tissues["sd"].fillna(0)).clip(0, 1)
    tissues["high"] = (tissues["f1"] + tissues["sd"].fillna(0)).clip(0, 1)
    if zoom:
        floor = min(math.floor(tissues["low"].min() * 20) / 20, 0.95)
        scale = alt.Scale(domain=[floor, 1], nice=False, clamp=True)
    else:
        scale = alt.Scale(domain=[0, 1], nice=False)
    color = alt.Color(
        "model:N",
        scale=alt.Scale(domain=letters, range=[COLORS[LETTERS.index(letter)] for letter in letters]),
        legend=None,
    )
    tooltip = [
        alt.Tooltip("tissue:N", title="Jaringan"),
        alt.Tooltip("model:N", title="Model"),
        alt.Tooltip("f1:Q", title="F1-score", format=".2%"),
        alt.Tooltip("sd:Q", title="SD", format=".2%")
        if basis == "cv"
        else alt.Tooltip("samples:Q", title="Sampel holdout", format=".0f"),
    ]
    base = alt.Chart(tissues).encode(
        y=alt.Y("tissue:N", sort=order, title=None, axis=alt.Axis(labelLimit=180, labelFontSize=12)),
        yOffset=alt.YOffset("model:N", sort=letters),
        color=color,
    )
    markers = base.mark_point(filled=True, size=70, opacity=1).encode(
        x=alt.X(
            "f1:Q",
            title="F1-score" + (", diperbesar" if zoom else ""),
            scale=scale,
            axis=alt.Axis(format=".0%", tickMinStep=0.01),
        ),
        tooltip=tooltip,
    )
    layers = [markers]
    if basis == "cv":
        layers.insert(0, base.mark_rule(strokeWidth=2, opacity=0.45, clip=True).encode(x="low:Q", x2="high:Q"))
    _legend(letters)
    st.altair_chart(
        alt.layer(*layers).properties(height=len(order) * (12 * len(letters) + 16) + AXIS_HEIGHT),
        width="stretch",
    )

    # The same values as a table, with the spread between models.
    scores = tissues.pivot(index="tissue", columns="model", values="f1").reindex(index=order, columns=letters)
    names = {letter: f"Model {letter}" for letter in letters}
    table = scores.map(percent).rename(columns=names)
    if len(letters) > 1:
        table["Selisih (poin)"] = (scores.max(axis=1) - scores.min(axis=1)).map(points)
    table = table.reset_index(names="Jaringan")
    best = _best(scores).rename(columns=names).reset_index(drop=True)
    with st.container(horizontal=True, vertical_alignment="top"):
        with st.expander("Nilai setiap jaringan"):
            st.dataframe(_highlighted(table, best), hide_index=True, width="stretch")
            st.caption("Sel berwarna: skor tertinggi pada jaringan itu, bila skor model berbeda.")
        download_button(
            "Unduh F1 per jaringan",
            tissues.drop(columns=["low", "high"]).to_csv(index=False),
            f"perbandingan_f1_jaringan_{basis}.csv",
            key="training_compare_download_tissues",
            icon=":material/download:",
        )


def _panel_section(state_root, models):
    """How many genes the final panels share."""
    st.subheader("Panel gen final")
    st.caption("Kesamaan gen yang dipilih model final. Jaccard = gen yang sama ÷ gabungan gen kedua panel.")
    try:
        panels = final_panels(state_root, models)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Panel gen tidak dapat dibaca: {user_error(error)}")
        return
    overlap = panel_overlap(panels)
    membership = panel_membership(panels)
    with st.container(horizontal=True):
        st.metric("Gen dipilih semua model", count(int((membership["models"] == len(panels)).sum())), border=True)
        st.metric("Gen gabungan semua panel", count(len(membership)), border=True)
    st.dataframe(
        pd.DataFrame(
            {
                "Pasangan": [f"Model {a} dan {b}" for a, b in zip(overlap["first"], overlap["second"])],
                "Gen sama": overlap["shared"].map(count),
                "Jaccard": overlap["jaccard"].map(percent),
            }
        ),
        hide_index=True,
        width="stretch",
    )
    if any(model["config"]["selector"] == "baseline" for model in models):
        st.caption("Baseline memakai seluruh gen kandidat, jadi kesamaannya terutama mengikuti ukuran panel lain.")


def _settings_section(models):
    """The training settings, with the ones that differ coloured."""
    st.subheader("Perbandingan pengaturan model")
    frame = settings_frame(models)
    only_differences = st.toggle("Hanya yang berbeda", value=True, key="training_compare_differences")
    shown = frame[frame["differs"]] if only_differences else frame
    if shown.empty:
        st.caption("Semua model memakai pengaturan dan data yang sama.")
        return
    names = {letter: f"Model {letter}" for letter in LETTERS[: len(models)]}
    table = shown.drop(columns="differs").rename(columns=names).reset_index(drop=True)
    differs = pd.DataFrame(
        {name: shown["differs"].to_numpy() & (not only_differences) for name in ["Pengaturan", *names.values()]}
    )
    st.dataframe(_highlighted(table, differs), hide_index=True, width="stretch")
    if not only_differences:
        st.caption("Baris berwarna: nilai berbeda antar-model.")


# --- Page ---


def render_training_comparison_page(state_root):
    """Show the Perbandingan Model page."""
    st.title("Perbandingan model")
    st.caption("Model tersimpan di gallery, berdampingan. Skor dibandingkan pada data evaluasi yang sama.")
    try:
        issues = []
        models = gallery_models(state_root, issues=issues)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Gallery tidak dapat dibaca: {user_error(error)}")
        return
    if issues:
        st.caption(f"{len(issues)} hasil belum lolos verifikasi dan tidak ditampilkan. Periksa Hasil Pelatihan.")
    if len(models) < 2:
        st.info(
            "Perbandingan membutuhkan minimal dua model terverifikasi di gallery. "
            "Latih model dengan metode lain untuk mulai membandingkan."
        )
        if st.button("Buka Latih Model", key="training_compare_train", icon=":material/settings:"):
            st.session_state.page = "Pelatihan Model"
            st.rerun()
        return

    picker, basis_column = st.columns([3, 1.4], vertical_alignment="bottom")
    with picker:
        chosen = _choose_models(models)
    if len(chosen) < 2:
        st.info("Pilih minimal dua model untuk dibandingkan.")
        return
    with basis_column:
        basis = _choose_basis(chosen)
    try:
        if basis:
            _comparability_note(chosen, basis)
        else:
            st.info(
                "Model yang dipilih tidak memiliki skor evaluasi yang sama, karena dilatih tanpa CV dan tanpa holdout "
                "atau dengan evaluasi berbeda. Panel gen dan pengaturan tetap dapat dibandingkan."
            )
        _model_cards(chosen, basis)
        if basis:
            _metric_section(chosen, basis)
            _tissue_section(chosen, basis)
        _panel_section(state_root, chosen)
        _settings_section(chosen)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f"Model tidak dapat dibandingkan: {user_error(error)}")
