"""Perbandingan Metode page."""
from __future__ import annotations

import altair as alt
import streamlit as st

from gtex_explorer.experiments.gene_panels import clarify_eligibility_reasons


def render_comparison_page(tables: dict) -> None:
    st.markdown('''<style>
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) { background: #fff; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stMain"] .block-container {
        padding: 1.4rem 2rem 1.5rem;
    }
    [data-testid="stElementContainer"]:has(.comparison-page-marker),
    [data-testid="stElementContainer"]:has([data-testid="stMarkdownContainer"] style) { display: none; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stDivider"] { margin: 1.4rem 0 .7rem; }
    [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stSubheader"] h3 { font-size: 1.4rem; }
    @media (max-width: 640px) {
        [data-testid="stAppViewContainer"]:has(.comparison-page-marker) [data-testid="stMain"] .block-container { padding: 1rem; }
    }
    </style><div class="comparison-page-marker"></div>''', unsafe_allow_html=True)
    st.title("Perbandingan metode")
    st.caption("CV Folding untuk memilih konfigurasi, holdout untuk evaluasi akhir.")
    summary = tables["run_summary"]
    validation = tables["experiment_validation"]
    cv_summary = clarify_eligibility_reasons(tables["cv_summary"], tables["cv_folds"])
    cv_summary["eligibility_reason"] = cv_summary["eligibility_reason"].str.replace(" | ", ", ", regex=False)
    heading, controls = st.columns([3, 1])
    with heading:
        st.subheader("1. Hasil CV Folding")
        st.markdown(f"**Data development: {summary['development_samples']:,} sampel - {summary['development_donors']:,} donor**".replace(",", "."))
        st.caption(
            f"{summary['selector_configurations']} konfigurasi seleksi fitur + {summary['baseline_configurations']} baseline - "
            f"{summary['cv_folds']} fold - {len(tables['cv_folds'])} hasil evaluasi"
        )
    with controls:
        metric_label = st.selectbox(
            "Metrik",
            ["Macro-F1", "Macro precision", "Macro recall", "Balanced accuracy", "Accuracy"],
        )
        zoom_scores = st.checkbox("Perbesar rentang skor pada grafik", key="comparison_zoom", value=False)
    metric_lookup = {
        "Macro-F1": ("mean_macro_f1", "std_macro_f1"),
        "Macro precision": ("mean_macro_precision", "std_macro_precision"),
        "Macro recall": ("mean_macro_recall", "std_macro_recall"),
        "Balanced accuracy": ("mean_balanced_accuracy", "std_balanced_accuracy"),
        "Accuracy": ("mean_accuracy", "std_accuracy"),
    }
    mean_column, std_column = metric_lookup[metric_label]
    selector_data = cv_summary.loc[
        cv_summary["experiment_type"].eq("feature_selection")
    ].copy()
    baseline_data = cv_summary.loc[
        cv_summary["experiment_type"].eq("baseline")
    ].copy()
    selector_data["lower"] = selector_data[mean_column] - selector_data[std_column]
    selector_data["upper"] = selector_data[mean_column] + selector_data[std_column]
    selector_legend = alt.Legend(
        orient="bottom", columns=3, symbolOpacity=1,
        symbolType="stroke", symbolStrokeWidth=3, labelColor="#344054", labelFontSize=13,
    )
    selector_colors = alt.Scale(range=["#4477aa", "#0f8b8d", "#9c6b21", "#5b5f97", "#587d71"])

    line = (
        alt.Chart(selector_data)
        .mark_line(strokeWidth=2.2)
        .encode(
            x=alt.X("genes:O", title="Jumlah gen terpilih"),
            y=alt.Y(
                f"{mean_column}:Q",
                title=f"Rata-rata {metric_label}" + (" - rentang diperbesar" if zoom_scores else ""),
                scale=alt.Scale(zero=False) if zoom_scores else alt.Scale(domain=[0, 1], nice=False),
                axis=alt.Axis(format=".0%"),
            ),
            color=alt.Color("selector_label:N", title="Selektor", scale=selector_colors, legend=selector_legend),
            tooltip=[
                alt.Tooltip("selector_label:N", title="Selektor"),
                alt.Tooltip("genes:O", title="Gen"),
                alt.Tooltip(f"{mean_column}:Q", title=metric_label, format=".2%"),
                alt.Tooltip(
                    "eligible_for_selector_ranking:N",
                    title="Eligible sebagai selector",
                ),
            ],
        )
    )
    
    points = line.mark_point(size=70, filled=True, strokeWidth=2).encode(
        stroke=alt.Color("selector_label:N", scale=selector_colors, legend=None),
        fill=alt.condition(
            alt.datum.eligible_for_selector_ranking,
            alt.Color("selector_label:N", scale=selector_colors, legend=None),
            alt.value("#ffffff"),
        ),
    )
    band = (
        alt.Chart(selector_data)
        .mark_errorband(opacity=0.10, clip=True)
        .encode(
            x=alt.X("genes:O"),
            y=alt.Y("lower:Q", title=None, axis=alt.Axis(format=".0%")),
            y2="upper:Q",
            color=alt.Color("selector_label:N", title="Selektor", scale=selector_colors, legend=selector_legend),
        )
    )
    baseline_rule = (
        alt.Chart(baseline_data)
        .mark_rule(color="#172033", strokeDash=[7, 5], strokeWidth=2)
        .encode(
            y=alt.Y(f"{mean_column}:Q", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("configuration_label:N", title="Baseline"),
                alt.Tooltip(f"{mean_column}:Q", title=metric_label, format=".2%"),
                alt.Tooltip("genes:Q", title="Gen"),
            ],
        )
    )
    st.altair_chart(
        (band + line + baseline_rule + points).resolve_legend(color="shared").properties(height=280),
        width="stretch",
    )
    st.caption(
        "Titik berongga: tidak memenuhi syarat seleksi. Garis hitam: baseline. Pita: ±1 SD antar-fold."
    )

    # Display the notebook's locked choice instead of choosing again in the UI.
    saved_choice = tables["run_summary"]
    best_row = selector_data.loc[
        selector_data["selector_id"].eq(saved_choice["best_selector_id"])
        & selector_data["genes"].eq(saved_choice["best_feature_count"])
    ].iloc[0]
    baseline_row = baseline_data.iloc[0]
    st.markdown(
        f"**Konfigurasi terpilih pada notebook:** {best_row['selector_label']} - "
        f"{int(best_row['genes']):,}".replace(",", ".")
        + f" gen - Macro-F1 CV {best_row['mean_macro_f1']:.2%}".replace(".", ",")
    )
    
    # Show the locked configuration and ANOVA/MI at the same k; keep the full grid below.
    preview = cv_summary.loc[
        cv_summary["configuration_id"].eq(best_row["configuration_id"])
        | (cv_summary["feature_selection"].isin(["anova", "mutual_information"])
           & cv_summary["genes"].eq(best_row["genes"]))
        | cv_summary["experiment_type"].eq("baseline")
    ].copy()
    
    preview["preview_order"] = preview["feature_selection"].map({"l1": 0, "anova": 1, "mutual_information": 2}).fillna(3)
    preview = preview.sort_values("preview_order")
    preview["selection_status"] = preview["configuration_id"].eq(best_row["configuration_id"]).map({True: "Terpilih", False: "Pembanding"})
    preview["configuration_label"] = preview["selector_label"].where(preview["experiment_type"].ne("baseline"), "Baseline")
    st.markdown("#### Ringkasan konfigurasi")
    st.dataframe(
        preview[["configuration_label", "genes", mean_column, std_column, "selection_status"]].style.apply(
            lambda row: ["background-color: #edf7f7; color: #087278; font-weight: 600" if row["selection_status"] == "Terpilih" else ""] * len(row), axis=1,
        ).format(
            {
                "genes": lambda value: f"{int(value):,}".replace(",", "."),
                mean_column: lambda value: f"{value:.2%}".replace(".", ","),
                std_column: lambda value: f"{value * 100:.2f}".replace(".", ","),
            }
        ),
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
            cv_summary[["performance_rank", "configuration_label", "experiment_type", "genes", mean_column, std_column,
                        "eligible_for_selector_ranking", "eligibility_reason"]].style.format({
                "genes": lambda value: f"{int(value):,}".replace(",", "."),
                mean_column: lambda value: f"{value:.2%}".replace(".", ","),
                std_column: lambda value: f"{value * 100:.2f}".replace(".", ","),
            }), width="stretch", hide_index=True,
            column_config={
                "performance_rank": "Peringkat Macro-F1",
                "configuration_label": st.column_config.TextColumn("Konfigurasi", width="large"),
                "experiment_type": "Jenis eksperimen", "genes": "Gen",
                mean_column: f"Rerata {metric_label} CV (%)", std_column: "SD (poin persentase)",
                "eligible_for_selector_ranking": st.column_config.CheckboxColumn("Memenuhi syarat seleksi"),
                "eligibility_reason": st.column_config.TextColumn("Keterangan", width="large"),
            },
        )
        st.caption("Peringkat memakai Macro-F1. Konfigurasi terpilih tetap mengikuti notebook dan harus memenuhi syarat seleksi. Baseline bukan kandidat selector.")
        st.caption(
            f"Validasi pipeline: {validation['configuration_count']} konfigurasi, "
            f"{validation['folds_per_configuration']} fold per konfigurasi, {validation['fold_level_results']} hasil fold-level."
        )
        st.caption(f"Baseline memakai seluruh {int(baseline_row['genes']):,} gen kandidat tanpa seleksi fitur.".replace(",", "."))

    st.divider()
    st.subheader("2. Hasil akhir - holdout")
    st.markdown(f"**{summary['test_samples']:,} sampel - {summary['test_donors']:,} donor**".replace(",", "."))
    st.caption(
        "Konfigurasi terpilih CV dibandingkan dengan baseline. Keduanya dilatih pada data development yang sama; holdout tidak digunakan untuk memilih ulang model."
    )

    holdout = tables["final_holdout"].copy()
    if holdout.empty:
        st.info(
            "Hasil perbandingan holdout belum tersedia. Jalankan LANGKAH 37A sampai "
            "39C pada notebook Part A, lalu salin ekspor terbaru ke folder data aplikasi."
        )
    else:
        selected_label = f"{best_row['selector_label']}, " + f"{int(best_row['genes']):,} gen".replace(",", ".")
        baseline_label = "Baseline, " + f"{int(baseline_row['genes']):,} gen".replace(",", ".")
        model_order = [selected_label, baseline_label]
        holdout["model"] = holdout["experiment_type"].map(
            {
                "feature_selection": selected_label,
                "baseline": baseline_label,
            }
        ).fillna(holdout["configuration_label"])

        selected_holdout = holdout.loc[
            holdout["experiment_type"].eq("feature_selection")
        ].iloc[0]
        baseline_holdout = holdout.loc[
            holdout["experiment_type"].eq("baseline")
        ].iloc[0]

        st.markdown(
            "**Selisih Macro-F1: "
            + f"{(selected_holdout['macro_f1'] - baseline_holdout['macro_f1']) * 100:+.2f}".replace(".", ",")
            + " poin persentase**"
        )

        holdout_metric_labels = {
            "macro_f1": "Macro-F1",
            "macro_precision": "Macro precision",
            "macro_recall": "Macro recall",
            "balanced_accuracy": "Balanced accuracy",
            "accuracy": "Accuracy",
        }
        holdout_long = holdout.melt(
            id_vars=["model"],
            value_vars=list(holdout_metric_labels),
            var_name="metric",
            value_name="score",
        )
        holdout_long["metric"] = holdout_long["metric"].map(
            holdout_metric_labels
        )
        holdout_long["display_score"] = holdout_long["score"].map(lambda value: f"{value:.2%}".replace(".", ","))
        holdout_base = (
            alt.Chart(holdout_long)
            .encode(
                x=alt.X("metric:N", title=None, sort=list(holdout_metric_labels.values())),
                xOffset=alt.XOffset("model:N", sort=model_order),
                y=alt.Y("score:Q", title="Skor holdout", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                color=alt.Color(
                    "model:N",
                    title=None, legend=alt.Legend(orient="bottom"),
                    scale=alt.Scale(domain=model_order, range=["#078c92", "#667085"]),
                ),
                tooltip=[
                    alt.Tooltip("model:N", title="Model"),
                    alt.Tooltip("metric:N", title="Metrik"),
                    alt.Tooltip("score:Q", title="Skor", format=".2%"),
                ],
            )
        )
        holdout_chart = (
            holdout_base.mark_bar()
            + holdout_base.mark_text(dy=-8, fontSize=12, color="#344054").encode(text="display_score:N")
        ).properties(height=240, padding={"top": 18})
        st.altair_chart(holdout_chart, width="stretch")

        holdout_table = holdout[
            [
                "configuration_label",
                "genes",
                "macro_f1",
                "macro_precision",
                "macro_recall",
                "balanced_accuracy",
                "accuracy",
            ]
        ].copy()
        holdout_table["configuration_label"] = holdout["model"]
        with st.expander("Metrik lengkap holdout"):
            st.dataframe(
                holdout_table.style.format(
                    {
                        "genes": lambda value: f"{int(value):,}".replace(",", "."),
                        "macro_f1": lambda value: f"{value:.2%}".replace(".", ","),
                        "macro_precision": lambda value: f"{value:.2%}".replace(".", ","),
                        "macro_recall": lambda value: f"{value:.2%}".replace(".", ","),
                        "balanced_accuracy": lambda value: f"{value:.2%}".replace(".", ","),
                        "accuracy": lambda value: f"{value:.2%}".replace(".", ","),
                    }
                ),
                width="stretch", hide_index=True,
                column_config={
                    "configuration_label": "Konfigurasi final", "genes": "Jumlah gen",
                    "macro_f1": "Macro-F1", "macro_precision": "Macro-precision",
                    "macro_recall": "Macro-recall", "balanced_accuracy": "Balanced accuracy", "accuracy": "Accuracy",
                },
            )

        per_tissue_holdout = tables["final_holdout_per_tissue"].copy()
        if not per_tissue_holdout.empty:
            per_tissue_holdout["model"] = per_tissue_holdout[
                "configuration_id"
            ].map(
                {
                    str(selected_holdout["configuration_id"]): selected_label,
                    str(baseline_holdout["configuration_id"]): baseline_label,
                }
            )
            tissue_chart = (
                alt.Chart(per_tissue_holdout)
                .mark_bar()
                .encode(
                    y=alt.Y("tissue:N", title=None, axis=alt.Axis(labelLimit=0, labelOverlap=False, labelFontSize=13)),
                    x=alt.X(
                        "f1-score:Q",
                        title="F1-score per jaringan",
                        scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format=".0%"),
                    ),
                    yOffset=alt.YOffset("model:N", sort=model_order),
                    color=alt.Color(
                        "model:N",
                        title="Model final",
                        scale=alt.Scale(domain=model_order, range=["#078c92", "#667085"]),
                    ),
                    tooltip=[
                        alt.Tooltip("tissue:N", title="Jaringan"),
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("f1-score:Q", title="F1-score", format=".2%"),
                    ],
                )
                .properties(height=max(430, per_tissue_holdout["tissue"].nunique() * 40))
            )
            with st.expander("F1-score per jaringan"):
                st.altair_chart(tissue_chart, width="stretch")
