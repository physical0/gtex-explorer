"""Every feature selection method of the notebook experiment side by side, on the cross-validation scores.

Shown in the Cross-validation tab of Ringkasan, below the configuration the notebook chose.
"""

from __future__ import annotations

import html

import altair as alt
import streamlit as st

from gtex_explorer.display import METHODS, METRICS, SERIES_COLORS, count, percent, points
from gtex_explorer.experiments.gene_panels import clarify_eligibility_reasons, readable_eligibility_reason

INK, MUTED, GRID = "#172033", "#657184", "#e2e8ec"
# Horizontal shift between selectors at one gene count, in gene-count slots, so their SD bars sit side by side.
DODGE = 0.09


def render_method_comparison(tables: dict) -> None:
    """Compare every selector and gene count on one CV metric, then list the configurations."""
    validation = tables["experiment_validation"]
    cv_summary = clarify_eligibility_reasons(tables["cv_summary"], tables["cv_folds"])
    cv_summary["eligibility_reason"] = cv_summary["eligibility_reason"].map(readable_eligibility_reason)
    cv_summary["experiment_type"] = cv_summary["experiment_type"].map({"feature_selection": "Seleksi fitur", "baseline": "Baseline"})

    heading, controls = st.columns([3, 1], vertical_alignment="bottom")
    heading.markdown('<h2 class="overview-section-title">Perbandingan metode</h2>', unsafe_allow_html=True)
    with controls:
        metric_label = st.selectbox("Metrik", list(METRICS.values()), key="comparison_metric")
    zoom_scores = st.checkbox("Perbesar rentang skor pada grafik", key="comparison_zoom", value=False)

    # The chosen metric for the chart and tables.
    metric_key = next(key for key, label in METRICS.items() if label == metric_label)
    mean_column, std_column = f"mean_{metric_key}", f"std_{metric_key}"
    selector_data = cv_summary.loc[cv_summary["experiment_type"].eq("Seleksi fitur")].copy()
    baseline_data = cv_summary.loc[cv_summary["experiment_type"].eq("Baseline")].copy()
    # ±1 SD, kept within 0–100%: a score cannot go past either end.
    selector_data["lower"] = (selector_data[mean_column] - selector_data[std_column]).clip(lower=0)
    selector_data["upper"] = (selector_data[mean_column] + selector_data[std_column]).clip(upper=1)
    # Selectors keep the notebook's order and their colour, whatever the metric.
    summary = tables["run_summary"]
    names = selector_data.drop_duplicates("selector_id").set_index("selector_id")["selector_label"]
    order = [names[key] for key in summary["selector_ids"] if key in names.index]
    colors = dict(zip(order, SERIES_COLORS))
    gene_counts = sorted(selector_data["genes"].astype(int).unique())
    slot = {genes: index for index, genes in enumerate(gene_counts)}
    shift = {label: (index - (len(order) - 1) / 2) * DODGE for index, label in enumerate(order)}
    selector_data["x"] = selector_data["genes"].map(slot) + selector_data["selector_label"].map(shift)
    selector_data["eligible"] = selector_data["eligible_for_selector_ranking"].map({True: "Ya", False: "Tidak"})
    selector_data["genes_label"] = selector_data["genes"].map(count)
    baseline_data["label"] = "Baseline " + baseline_data[mean_column].map(percent)

    # The legend sits outside the plot, so the plot keeps the chart's full height at any width.
    _legend(colors)
    color = alt.Color("selector_label:N", scale=alt.Scale(domain=order, range=list(colors.values())), legend=None)
    labels = "[" + ", ".join(f"'{count(genes)}'" for genes in gene_counts) + "]"
    x = alt.X(
        "x:Q",
        title="Top-k gen",
        scale=alt.Scale(domain=[-0.5, len(gene_counts) - 0.5], nice=False, zero=False),
        axis=alt.Axis(values=list(range(len(gene_counts))), labelExpr=f"{labels}[datum.value]", grid=False, ticks=False),
    )
    tooltip = [
        alt.Tooltip("selector_label:N", title="Selektor"),
        alt.Tooltip("genes_label:N", title="Top-k gen"),
        alt.Tooltip(f"{mean_column}:Q", title=f"Rerata {metric_label}", format=".2%"),
        alt.Tooltip(f"{std_column}:Q", title="SD antar-fold", format=".2%"),
        alt.Tooltip("eligible:N", title="Memenuhi syarat seleksi"),
    ]
    base = alt.Chart(selector_data)
    # Thin ±1 SD bars in the series colour: no translucent bands stacking into a wash.
    spread = base.mark_rule(strokeWidth=2, opacity=0.55).encode(
        x=x, y=alt.Y("lower:Q"), y2="upper:Q", color=color, tooltip=tooltip
    )
    line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        x=x,
        y=alt.Y(
            f"{mean_column}:Q",
            title=f"Rata-rata {metric_label}" + (" - rentang diperbesar" if zoom_scores else ""),
            scale=alt.Scale(zero=False, nice=True) if zoom_scores else alt.Scale(domain=[0, 1], nice=False),
            # Whole percentages only, so a zoomed axis never repeats a label.
            axis=alt.Axis(format=".0%", gridColor=GRID, domain=False, ticks=False, tickCount=5, tickMinStep=0.01),
        ),
        color=color,
        tooltip=tooltip,
    )
    # Filled where the configuration met the selection rules, hollow where it did not; a white ring
    # keeps neighbouring markers apart.
    markers = base.mark_point(size=80, filled=True, opacity=1, strokeWidth=2).encode(
        x=x,
        y=f"{mean_column}:Q",
        fill=alt.condition(alt.datum.eligible_for_selector_ranking, color, alt.value("#ffffff")),
        stroke=alt.condition(alt.datum.eligible_for_selector_ranking, alt.value("#ffffff"), color),
        tooltip=tooltip,
    )
    baseline_rule = (
        alt.Chart(baseline_data)
        .mark_rule(color=INK, strokeDash=[6, 4], strokeWidth=1.5)
        .encode(
            y=alt.Y(f"{mean_column}:Q"),
            tooltip=[
                alt.Tooltip("configuration_label:N", title="Baseline"),
                alt.Tooltip(f"{mean_column}:Q", title=f"Rerata {metric_label}", format=".2%"),
            ],
        )
    )
    baseline_label = (
        alt.Chart(baseline_data)
        .mark_text(align="right", baseline="bottom", dy=-4, fontSize=12, color=INK)
        .encode(x=alt.XDatum(len(gene_counts) - 0.5), y=alt.Y(f"{mean_column}:Q"), text="label:N")
    )
    st.altair_chart(
        alt.layer(spread, line, markers, baseline_rule, baseline_label)
        .properties(height=320)
        .configure(background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            labelColor=MUTED, titleColor=MUTED, labelFont="Segoe UI", titleFont="Segoe UI", labelFontSize=12, titleFontWeight=500
        ),
        width="stretch",
        theme=None,
    )

    # The notebook's locked choice, with ANOVA and MI at the same gene count and the baseline.
    best_row = selector_data.loc[
        selector_data["selector_id"].eq(summary["best_selector_id"])
        & selector_data["genes"].eq(summary["best_feature_count"])
    ].iloc[0]
    baseline_row = baseline_data.iloc[0]
    preview = cv_summary.loc[
        cv_summary["configuration_id"].eq(best_row["configuration_id"])
        | (
            cv_summary["feature_selection"].isin(["anova", "mutual_information"])
            & cv_summary["genes"].eq(best_row["genes"])
        )
        | cv_summary["experiment_type"].eq("Baseline")
    ].copy()
    preview["preview_order"] = preview["feature_selection"].map({"l1": 0, "anova": 1, "mutual_information": 2}).fillna(3)
    preview = preview.sort_values("preview_order")
    preview["selection_status"] = (
        preview["configuration_id"].eq(best_row["configuration_id"]).map({True: "Terpilih", False: "Pembanding"})
    )
    # The gene count has its own column here.
    preview["configuration_label"] = preview["selector_label"].where(
        preview["experiment_type"].ne("Baseline"), METHODS["baseline"]
    )

    # Short summary table, with every configuration in an expander.
    st.markdown('<p class="overview-table-title">Ringkasan konfigurasi</p>', unsafe_allow_html=True)
    st.dataframe(
        preview[["configuration_label", "genes", mean_column, std_column, "selection_status"]]
        .style.apply(
            lambda row: [
                "background-color: #edf7f7; color: #087278; font-weight: 600" if row["selection_status"] == "Terpilih" else ""
            ]
            * len(row),
            axis=1,
        )
        .format({"genes": count, mean_column: percent, std_column: points}),
        width="stretch",
        hide_index=True,
        column_config={
            "configuration_label": "Metode",
            "genes": "Gen",
            mean_column: f"Rerata {metric_label} CV (%)",
            std_column: "SD (poin persentase)",
            "selection_status": "Status",
        },
    )
    with st.expander(f"Lihat {len(cv_summary)} konfigurasi"):
        st.dataframe(
            cv_summary[
                [
                    "performance_rank",
                    "configuration_label",
                    "experiment_type",
                    "genes",
                    mean_column,
                    std_column,
                    "eligible_for_selector_ranking",
                    "eligibility_reason",
                ]
            ].style.format({"genes": count, mean_column: percent, std_column: points}),
            width="stretch",
            hide_index=True,
            column_config={
                "performance_rank": "Peringkat Macro-F1",
                "configuration_label": st.column_config.TextColumn("Konfigurasi", width="large"),
                "experiment_type": "Jenis eksperimen",
                "genes": "Gen",
                mean_column: f"Rerata {metric_label} CV (%)",
                std_column: "SD (poin persentase)",
                "eligible_for_selector_ranking": st.column_config.CheckboxColumn("Memenuhi syarat seleksi"),
                "eligibility_reason": st.column_config.TextColumn("Keterangan", width="large"),
            },
        )
        st.caption(
            "Peringkat memakai Macro-F1. Konfigurasi terpilih tetap mengikuti notebook dan harus memenuhi syarat "
            "seleksi. Baseline bukan kandidat selector."
        )
        st.caption(
            f"Validasi pipeline: {validation['configuration_count']} konfigurasi, "
            f"{validation['folds_per_configuration']} fold per konfigurasi, {validation['fold_level_results']} hasil fold-level."
        )
        st.caption(f"Baseline memakai seluruh {count(baseline_row['genes'])} gen kandidat tanpa seleksi fitur.")


def _legend(colors: dict) -> None:
    """Name each selector by its colour, then the marks that are the same for every selector."""
    swatch = '<span class="method-legend-item"><span class="method-legend-line" style="background:{}"></span>{}</span>'
    items = [swatch.format(color, html.escape(label)) for label, color in colors.items()]
    items += [
        '<span class="method-legend-item"><span class="method-legend-dash"></span>Baseline</span>',
        '<span class="method-legend-item"><span class="method-legend-hollow"></span>Tidak memenuhi syarat seleksi</span>',
        '<span class="method-legend-item"><span class="method-legend-spread"></span>±1 SD antar-fold</span>',
    ]
    st.markdown(
        f'<div class="method-legend" aria-label="Keterangan grafik">{"".join(items)}</div>', unsafe_allow_html=True
    )
