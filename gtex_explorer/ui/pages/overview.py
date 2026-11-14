"""Saved experiment overview: cohort, model selection, and held-out evaluation."""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from gtex_explorer.display import METRICS, count, decimal, method_name, percent, points
from gtex_explorer.training.data import TECHNICAL_FILTER_RULE
from gtex_explorer.ui.components.method_comparison import render_method_comparison
from gtex_explorer.ui.shell.theme import page_style


# Where these results come from, shown under the page title.
THESIS_NOTICE = (
    "Hasil di bawah ini berasal dari eksperimen perbandingan metode seleksi fitur untuk skripsi "
    "Program Studi Teknik Informatika, Universitas Tarumanagara. Hasilnya tetap dan tidak berubah "
    "oleh pelatihan di aplikasi."
)
# Where the experiment's data came from, cited as in the thesis (IEEE style).
DATA_FILES = [
    ("matriks gene TPM non-LCM", "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm_non_lcm.gct"),
    ("atribut sampel", "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"),
]
GTEX_PORTAL = "https://gtexportal.org"
GTEX_CITATION = (
    "GTEx Consortium, “The GTEx Consortium atlas of genetic regulatory effects across human tissues,” "
    "<em>Science</em>, vol. 369, no. 6509, pp. 1318–1330, 2020, doi: "
    '<a href="https://doi.org/10.1126/science.aaz1776" target="_blank">10.1126/science.aaz1776</a>.'
)


def _join(items: list) -> str:
    """Join a list as 'a dan b' or 'a, b, dan c'."""
    if len(items) < 3:
        return " dan ".join(items)
    return ", ".join(items[:-1]) + ", dan " + items[-1]


def _pipeline_steps(tables: dict, chosen, selector_label: str) -> tuple:
    """The steps of the experiment, from the exported results.

    Returns the main steps as (title, detail HTML, where its results are) and the steps repeated
    in every cross-validation fold as (title, detail).
    """
    summary = tables["run_summary"]
    configurations = tables["cv_summary"]
    selectors = configurations.loc[configurations["experiment_type"].eq("feature_selection")]
    eligible = int(selectors["eligible_for_selector_ranking"].sum())
    counts = _join([count(value) for value in summary["feature_counts"]])
    c_values = " / ".join(decimal(value) for value in summary["l1_c_values"])
    candidates = count(summary["variance_prefilter_genes"])
    folds = int(summary["cv_folds"])
    development = f"{count(summary['development_samples'])} sampel ({count(summary['development_donors'])} donor)"
    holdout = f"{count(summary['test_samples'])} sampel ({count(summary['test_donors'])} donor)"
    # The same sizes for use inside parentheses.
    development_size = f"{count(summary['development_samples'])} sampel dari {count(summary['development_donors'])} donor"
    holdout_size = f"{count(summary['test_samples'])} sampel dari {count(summary['test_donors'])} donor"
    total = len(selectors["configuration_id"].unique()) + int(summary["baseline_configurations"])
    escape = html.escape
    return [
        (
            "Data dan kohort",
            "Matriks TPM dan metadata sampel GTEx V11 dicocokkan menurut ID sampel; hanya sampel yang terdapat pada "
            f"keduanya yang dipakai. Dari {tables['labels']['tissue'].nunique()} jaringan target, "
            f"<strong>{count(summary['samples_loaded'])} sampel</strong> dipilih secara berstrata, yaitu diambil per "
            "jaringan agar setiap jaringan terwakili.",
            "",
        ),
        (
            "Pembagian data utama",
            f"Sampel dipartisi per donor menjadi <strong>data development {development}</strong> dan "
            f"<strong>data holdout {holdout}</strong>, sehingga seluruh sampel dari satu donor berada pada satu "
            "partisi saja. Data holdout disisihkan dan baru dipakai pada langkah 6 (Evaluasi akhir pada holdout).",
            "",
        ),
        (
            f"Cross-validation {folds} fold pada data development",
            f"Data development dibagi menjadi {folds} fold per donor. Setiap fold bergantian menjadi data validation, "
            "sedangkan fold lainnya menjadi data training. Langkah 3.1 sampai 3.4 dijalankan pada setiap fold: "
            "nilainya dihitung dari data training fold saja, lalu diterapkan pada data validation fold untuk mencegah "
            "kebocoran data.",
            "Hasil: tab Cross-validation di bawah dan panel CV pada Panel Gen Eksperimen",
        ),
        (
            "Pemilihan konfigurasi",
            f"Konfigurasi dengan rerata Macro-F1 CV tertinggi dipilih dari {eligible} konfigurasi seleksi fitur yang "
            f"memenuhi syarat: <strong>{escape(selector_label)}, {count(chosen['genes'])} gen</strong>. Konfigurasi L1 "
            "tidak memenuhi syarat bila pada suatu fold jumlah gen berkoefisien bukan nol lebih sedikit daripada jumlah "
            "gen yang diminta. Bila rerata Macro-F1 sama, dipilih balanced accuracy yang lebih tinggi, lalu jumlah gen "
            "yang lebih sedikit, lalu urutan ID metode. Baseline tidak ikut dipilih karena hanya menjadi pembanding.",
            "Hasil: tab Cross-validation di bawah",
        ),
        (
            "Pelatihan ulang pada data development",
            f"Proses pra-pemrosesan dilakukan satu kali pada seluruh data development ({development_size}), sehingga daftar "
            f"{candidates} gen kandidat model final dapat sedikit berbeda dari daftar pada fold. Seleksi fitur terpilih "
            "dan XGBoost lalu dilatih pada hasilnya. Baseline dilatih ulang dengan cara yang sama sebagai pembanding.",
            "Hasil: panel Model final pada Panel Gen Eksperimen",
        ),
        (
            "Evaluasi akhir pada holdout",
            f"Data holdout ({holdout_size}) diproses dengan nilai pra-pemrosesan dari data development, lalu model final "
            "dan baseline dievaluasi satu kali tanpa diubah. Skor holdout tidak dipakai untuk memilih ulang "
            "konfigurasi, sehingga menjadi perkiraan kinerja pada data yang belum pernah dilihat model.",
            "Hasil: tab Hasil akhir - holdout di bawah",
        ),
    ], [
        (
            "Pra-pemrosesan",
            "Nilai kosong diisi median gen; filter teknis membuang gen yang terlalu jarang terekspresi atau tidak "
            f"bervariasi ({TECHNICAL_FILTER_RULE}); nilai ditransformasi log2(TPM+1); {candidates} gen kandidat "
            "dengan varians tertinggi dipertahankan; lalu setiap gen distandardisasi (z-score).",
        ),
        (
            "Seleksi fitur",
            f"ANOVA, Mutual Information (n_neighbors = {summary['mi_primary_n_neighbors']}), dan regresi logistik L1 "
            f"(C = {c_values}) memeringkat gen kandidat. Setiap metode mempertahankan {counts} gen teratas, sehingga "
            f"terbentuk {int(summary['selector_configurations'])} konfigurasi. Baseline memakai seluruh "
            f"{count(summary['baseline_genes'])} gen kandidat tanpa seleksi fitur.",
        ),
        (
            "XGBoost",
            f"XGBoost dilatih pada gen terpilih dengan {summary['xgboost_parameters']['n_estimators']} putaran boosting "
            "dan hiperparameter yang sama untuk setiap konfigurasi.",
        ),
        (
            "Skor validation",
            "Model dievaluasi pada data validation fold dengan Macro-F1, balanced accuracy, dan accuracy: "
            f"{total} konfigurasi × {folds} fold = {len(tables['cv_folds'])} hasil evaluasi.",
        ),
    ]


def _pipeline_overview(tables: dict, chosen, selector_label: str, final) -> list:
    """Each step's short name and its one key fact (HTML), for the compact row; the details hold the rest."""
    summary = tables["run_summary"]
    configurations = tables["cv_summary"]
    total = configurations["configuration_id"].nunique()
    return [
        ("Data dan kohort", f"{count(summary['samples_loaded'])} sampel - {tables['labels']['tissue'].nunique()} jaringan"),
        (
            "Pembagian data utama",
            f"{count(summary['development_samples'])} development - {count(summary['test_samples'])} holdout, "
            "donor terpisah",
        ),
        ("Cross-validation", f"{int(summary['cv_folds'])} fold - {total} konfigurasi"),
        (
            "Pemilihan konfigurasi",
            f"<strong>{html.escape(selector_label)}, {count(chosen['genes'])} gen</strong>",
        ),
        ("Pelatihan ulang", f"Seluruh {count(summary['development_samples'])} sampel development"),
        ("Evaluasi akhir", f"Macro-F1 holdout <strong>{percent(final['macro_f1'])}</strong>"),
    ]


def _render_pipeline(tables: dict, chosen, selector_label: str, final) -> None:
    """Show the experiment as a row of six steps, with every step's details in a closed expander."""
    row = "".join(
        f'<li class="pipeline-row-step"><span class="pipeline-number" aria-hidden="true">{number}</span>'
        f'<p class="pipeline-row-title">{html.escape(title)}</p><p class="pipeline-row-fact">{fact}</p></li>'
        for number, (title, fact) in enumerate(_pipeline_overview(tables, chosen, selector_label, final), 1)
    )
    st.markdown(
        '<section class="overview-pipeline overview-flow" aria-label="Alur eksperimen">'
        '<h2 class="overview-section-title">Alur eksperimen</h2>'
        f'<ol class="pipeline-row">{row}</ol></section>',
        unsafe_allow_html=True,
    )
    with st.expander("Rincian setiap langkah"):
        _render_pipeline_details(tables, chosen, selector_label)


def _render_pipeline_details(tables: dict, chosen, selector_label: str) -> None:
    """Every step in full, the cross-validation loop as a box inside its step."""
    steps, loop = _pipeline_steps(tables, chosen, selector_label)
    loop_html = "".join(
        f'<li class="pipeline-substep"><p class="pipeline-substep-title">3.{number} {title}</p><p>{detail}</p></li>'
        for number, (title, detail) in enumerate(loop, 1)
    )
    items = []
    for number, (title, detail, where) in enumerate(steps, 1):
        body = f'<p class="pipeline-title">{html.escape(title)}</p><p class="pipeline-detail">{detail}</p>'
        if number == 3:
            body += (
                '<div class="pipeline-loop"><p class="pipeline-loop-label">Diulang pada setiap fold</p>'
                f'<ol class="pipeline-substeps">{loop_html}</ol></div>'
            )
        if where:
            body += f'<p class="pipeline-where">{html.escape(where)}</p>'
        items.append(
            f'<li class="pipeline-step"><span class="pipeline-number" aria-hidden="true">{number}</span>'
            f"<div>{body}</div></li>"
        )
    st.markdown(
        '<section class="overview-flow" aria-label="Rincian alur eksperimen">'
        f'<ol class="pipeline-steps">{"".join(items)}</ol>'
        '<p class="overview-note">Latih Model menjalankan urutan yang sama untuk satu konfigurasi per pelatihan, '
        "dengan nama tahap seperti pada panduan Cara kerja pelatihan. Beberapa pelatihan dapat dibandingkan pada "
        "Perbandingan Model.</p></section>",
        unsafe_allow_html=True,
    )


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
    # Method names as on the training pages.
    selector_label = method_name(chosen["feature_selection"], l1_c=chosen["l1_c"])

    selector_count = configurations.loc[
        configurations["experiment_type"].eq("feature_selection"), "configuration_id"
    ].nunique()
    baseline_count = configurations.loc[configurations["experiment_type"].eq("baseline"), "configuration_id"].nunique()
    configuration_count = selector_count + baseline_count
    development_samples, test_samples = count(summary["development_samples"]), count(summary["test_samples"])
    development_donors, test_donors = count(summary["development_donors"]), count(summary["test_donors"])

    st.markdown(page_style("overview"), unsafe_allow_html=True)
    st.markdown('<div class="overview-page-marker"></div>', unsafe_allow_html=True)

    # Cohort summary.
    st.title("Ringkasan eksperimen")
    st.caption(THESIS_NOTICE)

    cohort_stats = [
        ("Sampel terpilih", summary["samples_loaded"]),
        ("Donor unik", summary["development_donors"] + summary["test_donors"]),
        ("Kelas jaringan", tables["labels"]["tissue"].nunique()),
    ]
    cohort_html = "".join(
        '<div class="overview-stat">'
        f'<div class="overview-stat-value">{count(value)}</div>'
        f'<div class="overview-stat-label">{label}</div></div>'
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

    # How the experiment ran, step by step; the tabs below hold the results of steps 3, 4, and 6.
    _render_pipeline(tables, chosen, selector_label, final)

    # Contents of the two result tabs.
    gene_count = count(final["genes"])
    config_values = [
        ("Seleksi fitur", selector_label),
        ("Gen terpilih", gene_count),
        ("Klasifikasi", f"XGBoost - {summary['xgboost_parameters']['n_estimators']} putaran boosting"),
    ]
    config_html = "".join(
        '<div class="overview-config-row">' f"<dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in config_values
    )
    cv_mean, cv_std = percent(chosen["mean_macro_f1"]), points(chosen["std_macro_f1"])
    # The baseline was trained on the same development data and tested on the same holdout.
    holdout = tables["final_holdout"]
    baseline = holdout.loc[holdout["experiment_type"].eq("baseline")].iloc[0] if not holdout.empty else None
    baseline_name = configurations.loc[configurations["experiment_type"].eq("baseline"), "configuration_label"].iloc[0]
    metric_html = "".join(
        '<div class="overview-stat">'
        f'<div class="overview-stat-value">{percent(final[field])}</div>'
        f'<div class="overview-stat-label">{METRICS[field]}</div>'
        + (
            f'<div class="overview-stat-delta">{points(final[field] - baseline[field], sign=True)} poin vs baseline</div>'
            if baseline is not None
            else ""
        )
        + "</div>"
        for field in ["macro_f1", "balanced_accuracy", "accuracy"]
    )
    cv_tab, holdout_tab = st.tabs(
        ["Cross-validation", "Hasil akhir - holdout"],
        default="Hasil akhir - holdout",
    )
    # Every method and gene count first, then the configuration chosen from them.
    with cv_tab:
        render_method_comparison(tables)
        st.markdown('<div class="overview-rule"></div>', unsafe_allow_html=True)
        st.markdown(
            '<section class="overview-search" aria-label="Hasil cross-validation">'
            '<h2 class="overview-section-title">Pemilihan konfigurasi dengan cross-validation</h2>'
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
        with st.expander("Metrik lengkap konfigurasi terpilih CV"):
            for field, label in METRICS.items():
                st.markdown(
                    f"**{label}:** {percent(chosen[f'mean_{field}'])} ± {points(chosen[f'std_{field}'])} poin persentase"
                )
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
        if baseline is not None:
            _render_baseline_comparison(final, baseline, baseline_name)
        _render_holdout_tissue_chart(tables)
        with st.expander("Metrik lengkap holdout"):
            if baseline is not None:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Metrik": list(METRICS.values()),
                            "Model final": [percent(final[field]) for field in METRICS],
                            "Baseline": [percent(baseline[field]) for field in METRICS],
                            "Selisih (poin persentase)": [
                                points(final[field] - baseline[field], sign=True) for field in METRICS
                            ],
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.markdown(" - ".join(f"**{METRICS[field]}:** {percent(final[field])}" for field in METRICS))
            st.caption(
                "Macro-F1 merangkum F1 seluruh kelas. Balanced accuracy adalah rerata recall, sedangkan accuracy adalah proporsi prediksi benar."
            )
            st.caption("Holdout dipisahkan sebelum cross-validation dan tidak digunakan untuk memilih konfigurasi.")

    # Links to the other pages.
    actions = st.columns(2)
    for column, label, destination, key in [
        (actions[0], "Telusuri panel gen", "Panel Gen", "overview_open_genes"),
        (actions[1], "Prediksi jaringan", "Prediksi Jaringan", "overview_open_prediction"),
    ]:
        if column.button(label, key=key, width="stretch"):
            st.session_state.page = destination
            st.rerun()

    # The data source, last, as a reference.
    files = " dan ".join(f"{name} (<code>{html.escape(file)}</code>)" for name, file in DATA_FILES)
    st.markdown(
        '<section class="overview-source" aria-label="Sumber data">'
        '<h2 class="overview-section-title">Sumber data</h2>'
        '<p class="overview-note">Data ekspresi gen dan metadata sampel berasal dari Genotype-Tissue Expression '
        f'(GTEx) V11, yang diunduh dari <a href="{GTEX_PORTAL}" target="_blank">GTEx Portal</a>: {files}.</p>'
        f'<p class="overview-reference">{GTEX_CITATION}</p></section>',
        unsafe_allow_html=True,
    )


BASELINE_MODELS = ["Model final", "Baseline XGBoost"]


def _render_baseline_comparison(final, baseline, baseline_name: str) -> None:
    """The final model next to the XGBoost baseline on every holdout metric, as on the tissue chart below."""
    gap = points(final["macro_f1"] - baseline["macro_f1"], sign=True)
    st.markdown(
        '<div class="overview-chart-heading"><h2 class="overview-section-title">Perbandingan dengan baseline XGBoost</h2>'
        f'<p class="overview-note">Data holdout - selisih Macro-F1 <strong>{gap} poin persentase</strong></p></div>',
        unsafe_allow_html=True,
    )
    # The bars start at zero, so the final model's label also carries its difference from the baseline.
    frame = pd.DataFrame(
        [
            {
                "metric": label,
                "model": model,
                "score": float(source[field]),
                "display": percent(source[field])
                + (f" ({points(final[field] - baseline[field], sign=True)})" if source is final else ""),
            }
            for field, label in METRICS.items()
            for model, source in zip(BASELINE_MODELS, [final, baseline])
        ]
    )
    base = alt.Chart(frame).encode(
        y=alt.Y(
            "metric:N",
            sort=list(METRICS.values()),
            title=None,
            scale=alt.Scale(paddingInner=0.3),
            axis=alt.Axis(labelLimit=0, labelFontSize=13, labelPadding=10, ticks=False, domain=False),
        ),
        yOffset=alt.YOffset("model:N", sort=BASELINE_MODELS),
        tooltip=[
            alt.Tooltip("metric:N", title="Metrik"),
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("score:Q", title="Skor", format=".2%"),
        ],
    )
    bars = base.mark_bar(size=15).encode(
        x=alt.X(
            "score:Q",
            title="Skor holdout",
            scale=alt.Scale(domain=[0, 1], nice=False),
            axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1], format=".0%", gridColor="#e2e8ec"),
        ),
        color=alt.Color(
            "model:N",
            title=None,
            scale=alt.Scale(domain=BASELINE_MODELS, range=["#0f8b8d", "#7d8797"]),
            legend=alt.Legend(orient="top", labelFontSize=12, symbolType="square"),
        ),
    )
    scores = base.mark_text(align="left", dx=10, fontSize=12, color="#344054", clip=False).encode(
        x=alt.XDatum(1), text="display:N"
    )
    chart = (
        (bars + scores)
        .properties(height=len(METRICS) * 62, padding={"right": 120})
        .configure_view(stroke=None)
        .configure_axis(labelColor="#657184", labelFont="Segoe UI", titleFont="Segoe UI")
        .configure_legend(labelColor="#344054", labelFont="Segoe UI")
    )
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption(
        f"Pembanding: {baseline_name}. Keduanya dilatih pada data development yang sama dan diuji "
        "pada holdout yang sama; holdout tidak dipakai untuk memilih ulang model."
    )


def _render_holdout_tissue_chart(tables: dict) -> None:
    """Show the F1-score of each tissue on the holdout data: the final model as bars, the baseline as a mark."""
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
    per_tissue["display_f1"] = per_tissue["f1-score"].map(percent)
    per_tissue["display_support"] = per_tissue["support"].map(lambda value: f"n = {int(value)}")
    # The baseline's score per tissue, when the export has it.
    holdout = tables["final_holdout"]
    both = tables["final_holdout_per_tissue"]
    baseline_ids = holdout.loc[holdout["experiment_type"].eq("baseline"), "configuration_id"] if not holdout.empty else []
    has_baseline = len(baseline_ids) == 1 and not both.empty
    if has_baseline:
        baseline_scores = both.loc[both["configuration_id"].eq(baseline_ids.iloc[0])].set_index("tissue")["f1-score"]
        per_tissue["baseline_f1"] = per_tissue["tissue"].map(baseline_scores)
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
            alt.Tooltip("f1-score:Q", title="F1-score model final", format=".2%"),
            *([alt.Tooltip("baseline_f1:Q", title="F1-score baseline", format=".2%")] if has_baseline else []),
            alt.Tooltip("support:Q", title="Sampel holdout", format=".0f"),
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
    layers = bars + scores + supports
    if has_baseline:
        layers += base.mark_tick(color="#172033", thickness=2.5, size=24).encode(x="baseline_f1:Q")
    chart = (
        layers
        .properties(
            height=max(160, len(per_tissue) * 30),
            padding={"right": 150},
        )
        .configure_view(stroke=None)
        .configure_axis(labelColor="#657184", labelFont="Segoe UI", titleFont="Segoe UI")
    )
    st.altair_chart(chart, width="stretch", theme=None)
    st.caption(
        ("Batang: model final. Garis hitam: baseline. " if has_baseline else "")
        + "n = jumlah sampel holdout pada setiap kelas jaringan."
    )
