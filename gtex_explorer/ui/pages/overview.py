"""Saved experiment overview: cohort, model selection, and held-out evaluation."""

from __future__ import annotations

import html

import altair as alt
import streamlit as st

from gtex_explorer.ui.shell.theme import page_style


def render_overview_page(tables: dict) -> None:
    """Show the summary of the thesis experiment."""

    # Find the final configuration and prepare the numbers.
    summary = tables["run_summary"]
    final = tables["final_test"].iloc[0]
    configurations = tables["cv_summary"]
    # Match both selector and gene count. Never reselect a winner from holdout scores.
    chosen = configurations.loc[
        configurations["selector_id"].eq(final["selector_id"]) & configurations["genes"].eq(final["genes"])
    ]
    if len(chosen) != 1:
        raise ValueError("Konfigurasi final tidak mempunyai satu ringkasan CV yang sesuai.")
    chosen = chosen.iloc[0]
    selector_label = str(chosen["selector_label"])
    if chosen["feature_selection"] == "l1":
        selector_label = f"L1 - C = {float(chosen['l1_c']):g}"

    selector_count = configurations.loc[
        configurations["experiment_type"].eq("feature_selection"), "configuration_id"
    ].nunique()
    baseline_count = configurations.loc[configurations["experiment_type"].eq("baseline"), "configuration_id"].nunique()
    configuration_count = selector_count + baseline_count
    development_samples = f"{int(summary['development_samples']):,}".replace(",", ".")
    test_samples = f"{int(summary['test_samples']):,}".replace(",", ".")
    development_donors = f"{int(summary['development_donors']):,}".replace(",", ".")
    test_donors = f"{int(summary['test_donors']):,}".replace(",", ".")

    st.markdown(page_style("overview"), unsafe_allow_html=True)
    st.markdown('<div class="overview-page-marker"></div>', unsafe_allow_html=True)

    # Cohort summary.
    st.title("Ringkasan eksperimen")

    cohort_stats = [
        ("Sampel terpilih", summary["samples_loaded"]),
        ("Donor unik", summary["development_donors"] + summary["test_donors"]),
        ("Kelas jaringan", tables["labels"]["tissue"].nunique()),
    ]
    cohort_html = "".join(
        '<div class="overview-stat">'
        f'<div class="overview-stat-value">{int(value):,}</div>'
        f'<div class="overview-stat-label">{label}</div></div>'.replace(",", ".")
        for label, value in cohort_stats
    )
    st.markdown(
        '<section class="overview-cohort" aria-label="Kohort GTEx">'
        f'<div class="overview-cohort-stats">{cohort_html}</div>'
        '<div class="overview-split-grid">'
        '<section aria-label="Data development"><p class="overview-split-title">Data development</p>'
        f'<p class="overview-split-count">{development_samples} sampel - {development_donors} donor</p>'
        "</section>"
        '<section aria-label="Data holdout"><p class="overview-split-title">Data holdout</p>'
        f'<p class="overview-split-count">{test_samples} sampel - {test_donors} donor</p>'
        "</section></div></section>",
        unsafe_allow_html=True,
    )

    # Contents of the two result tabs.
    gene_count = f"{int(final['genes']):,}".replace(",", ".")
    config_values = [
        ("Seleksi fitur", selector_label),
        ("Gen terpilih", gene_count),
        ("Klasifikasi", f"XGBoost - {summary['xgboost_parameters']['n_estimators']} putaran boosting"),
    ]
    config_html = "".join(
        '<div class="overview-config-row">' f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in config_values
    )
    cv_mean = f"{float(chosen['mean_macro_f1']):.2%}".replace(".", ",")
    cv_std = f"{float(chosen['std_macro_f1']) * 100:.2f}".replace(".", ",")
    metric_html = "".join(
        '<div class="overview-stat">'
        f'<div class="overview-stat-value">{float(final[field]):.2%}</div>'
        f'<div class="overview-stat-label">{label}</div></div>'.replace(".", ",")
        for label, field in [
            ("Macro-F1", "macro_f1"),
            ("Balanced accuracy", "balanced_accuracy"),
            ("Accuracy", "accuracy"),
        ]
    )
    cv_tab, holdout_tab = st.tabs(
        ["CV Folding", "Hasil akhir - holdout"],
        default="Hasil akhir - holdout",
    )
    # How the configuration was chosen.
    with cv_tab:
        st.markdown(
            '<section class="overview-search" aria-label="Hasil CV Folding">'
            '<h2 class="overview-section-title">Pemilihan konfigurasi dengan CV Folding</h2>'
            f'<p class="overview-note">Data development: {development_samples} sampel - {development_donors} donor</p>'
            f'<p class="overview-note">{selector_count} konfigurasi seleksi fitur + {baseline_count} baseline - '
            f'<strong>{configuration_count} konfigurasi × {summary["cv_folds"]} fold = '
            f'{len(tables["cv_folds"])} hasil evaluasi</strong></p></section>'
            '<div class="overview-model-grid">'
            '<section class="overview-config" aria-label="Konfigurasi terpilih">'
            '<h2 class="overview-section-title">Konfigurasi terpilih</h2>'
            f"<dl>{config_html}</dl></section>"
            '<section class="overview-cv-summary" aria-label="Skor CV">'
            f'<div class="overview-stat-value">{cv_mean}</div>'
            '<div class="overview-stat-label">Rerata Macro-F1 CV</div>'
            f'<p class="overview-note">SD {cv_std} poin persentase</p>'
            f'<p class="overview-note">Rerata dan SD dari {summary["cv_folds"]} fold</p></section></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Data training digunakan untuk melatih selector dan XGBoost. Skor dihitung pada data validation tiap fold."
        )
        if st.button("Lihat perbandingan metode", key="overview_cv_comparison", width="stretch"):
            st.session_state.page = "Perbandingan Metode"
            st.rerun()
        with st.expander("Metrik lengkap CV"):
            for label, field in [
                ("Macro-F1", "macro_f1"),
                ("Macro precision", "macro_precision"),
                ("Macro recall", "macro_recall"),
                ("Balanced accuracy", "balanced_accuracy"),
                ("Accuracy", "accuracy"),
            ]:
                mean = f"{float(chosen[f'mean_{field}']):.2%}".replace(".", ",")
                sd = f"{float(chosen[f'std_{field}']) * 100:.2f}".replace(".", ",")
                st.markdown(f"**{label}:** {mean} ± {sd} poin persentase")
            st.caption("Nilai ± adalah simpangan baku antar-fold, bukan interval kepercayaan atau skor holdout.")
            st.caption(
                "Konfigurasi dipilih dari hasil CV yang memenuhi syarat seleksi, lalu dilatih ulang pada seluruh data development."
            )

    # Final result on the holdout data.
    with holdout_tab:
        st.markdown(
            '<section class="overview-evaluation" aria-label="Evaluasi akhir pada holdout">'
            '<h2 class="overview-section-title">Evaluasi akhir pada holdout</h2>'
            f'<p class="overview-note">{test_samples} sampel - {test_donors} donor</p>'
            f'<p class="overview-model-identity">{html.escape(selector_label)}, {gene_count} gen, XGBoost</p>'
            '<p class="overview-note">Konfigurasi terpilih dari CV</p>'
            f'<div class="overview-primary-metrics">{metric_html}</div></section>',
            unsafe_allow_html=True,
        )
        _render_holdout_tissue_chart(tables)
        with st.expander("Metrik lengkap holdout"):
            st.markdown(
                f"**Macro-precision holdout:** {float(final['macro_precision']):.2%} - "
                f"**Macro-recall holdout:** {float(final['macro_recall']):.2%}".replace(".", ",")
            )
            st.caption(
                "Macro-F1 merangkum F1 seluruh kelas. Balanced accuracy adalah rerata recall, sedangkan accuracy adalah proporsi prediksi benar."
            )
            st.caption("Holdout dipisahkan sebelum validasi silang dan tidak digunakan untuk memilih konfigurasi.")

    # Where the numbers come from.
    with st.expander("Asal angka dan rincian eksperimen"):
        st.markdown(
            "Sampel dihitung dari SAMPID unik, donor dari SUBJID unik, dan kelas dari label SMTSD. "
            "Kohort dipilih secara berstrata menurut jaringan setelah pencocokan ID metadata dengan matriks TPM. "
            "Pembagian donor diaudit melalui sample_split.csv. Sampel adalah profil yang diklasifikasikan, "
            "sedangkan gen merupakan variabel masukan model."
        )
        selector_rows = configurations.loc[configurations["experiment_type"].eq("feature_selection")]
        labels = selector_rows.drop_duplicates("selector_id").set_index("selector_id")["selector_label"]
        selector_names = ", ".join(str(labels.loc[key]) for key in summary["selector_ids"] if key in labels.index)
        counts = ", ".join(
            f"{value:,}".replace(",", ".") for value in sorted(selector_rows["genes"].astype(int).unique())
        )
        baseline_genes = f"{int(summary['baseline_genes']):,}".replace(",", ".")
        st.markdown(
            f"**{selector_count} konfigurasi seleksi fitur:** {selector_names}, masing-masing pada **{counts} gen**. "
            f"Baseline memakai **{baseline_genes} gen** tanpa seleksi fitur. Satu hasil validasi adalah satu "
            "pasangan konfigurasi–fold, bukan satu sampel atau satu jaringan, dan tidak mencakup evaluasi akhir pada holdout."
        )
        source_rows = [
            ("Kohort dan split", "run_summary.json", "Jumlah sampel serta donor pada setiap kelompok."),
            ("Kelas jaringan", "label_mapping.csv", "Label jaringan yang digunakan model."),
            ("Pemilihan dan skor CV", "experiment_cv_summary.csv", "Ringkasan setiap konfigurasi pada fold validasi."),
            ("Hasil per fold", "experiment_cv_folds.csv", "Satu baris per pasangan konfigurasi–fold."),
            ("Evaluasi akhir", "final_test_summary.csv", "Skor model final pada holdout."),
            ("F1 dan sampel tiap kelas", "test_per_tissue_xgboost.csv", "Skor per jaringan dan jumlah sampel uji."),
        ]
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row) + "</tr>" for row in source_rows
        )
        st.markdown(
            '<table class="overview-source-table"><thead><tr><th>Hasil</th><th>Berkas</th><th>Isi</th></tr>'
            f"</thead><tbody>{rows_html}</tbody></table>",
            unsafe_allow_html=True,
        )

    # Links to the other experiment pages.
    actions = st.columns(3)
    for column, label, destination, key in [
        (actions[0], "Bandingkan metode", "Perbandingan Metode", "overview_open_comparison"),
        (actions[1], "Telusuri panel gen", "Panel Gen", "overview_open_genes"),
        (actions[2], "Prediksi jaringan", "Prediksi Jaringan", "overview_open_prediction"),
    ]:
        if column.button(label, key=key, width="stretch"):
            st.session_state.page = destination
            st.rerun()


def _render_holdout_tissue_chart(tables: dict) -> None:
    """Show the F1-score of each tissue on the holdout data."""
    st.markdown(
        '<div class="overview-chart-heading"><h2 class="overview-section-title">F1-score per jaringan</h2>'
        '<p class="overview-note">Data holdout</p></div>',
        unsafe_allow_html=True,
    )
    order = st.selectbox(
        "Urutkan jaringan",
        ["F1-score terendah", "F1-score tertinggi", "Nama jaringan"],
        key="overview_tissue_order",
    )
    per_tissue = tables["per_tissue"].copy()
    per_tissue = per_tissue.sort_values(
        ["tissue"] if order == "Nama jaringan" else ["f1-score", "tissue"],
        ascending=True if order != "F1-score tertinggi" else [False, True],
        kind="stable",
    )
    per_tissue["display_f1"] = per_tissue["f1-score"].map(lambda value: f"{float(value):.2%}".replace(".", ","))
    per_tissue["display_support"] = per_tissue["support"].map(lambda value: f"n = {int(value)}")
    base = alt.Chart(per_tissue).encode(
        y=alt.Y(
            "tissue:N",
            sort=per_tissue["tissue"].tolist(),
            title=None,
            axis=alt.Axis(
                labelLimit=0, labelOverlap=False, labelFontSize=13, labelPadding=10, ticks=False, domain=False
            ),
        ),
        tooltip=[
            alt.Tooltip("tissue:N", title="Jaringan"),
            alt.Tooltip("f1-score:Q", title="F1-score", format=".2%"),
            alt.Tooltip("support:Q", title="Sampel uji", format=".0f"),
        ],
    )
    bars = base.mark_bar(color="#0f8b8d", size=17).encode(
        x=alt.X(
            "f1-score:Q",
            title="F1-score",
            scale=alt.Scale(domain=[0, 1], nice=False),
            axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".0%", gridColor="#e2e8ec"),
        )
    )
    scores = base.mark_text(align="left", dx=10, fontSize=13, color="#344054", clip=False).encode(
        x=alt.XDatum(1),
        text="display_f1:N",
    )
    supports = base.mark_text(align="left", dx=90, fontSize=12, color="#657184", clip=False).encode(
        x=alt.XDatum(1),
        text="display_support:N",
    )
    chart = (
        (bars + scores + supports)
        .properties(
            height=max(160, len(per_tissue) * 30),
            padding={"right": 150},
        )
        .configure_view(stroke=None)
        .configure_axis(labelColor="#657184", labelFont="Segoe UI", titleFont="Segoe UI")
    )
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption("n = jumlah sampel holdout pada setiap kelas jaringan.")
