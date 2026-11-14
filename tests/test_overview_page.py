"""Check that the redesign renders exports, not numbers copied from the mockup."""
from pathlib import Path
import json
import sys

import pytest
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_DIR = Path(__file__).resolve().parents[1]


def overview_app(changes: str = "") -> AppTest:
    source = (
        f"import sys\nsys.path.insert(0, {str(APP_DIR)!r})\n"
        "from gtex_explorer.experiments.reference_tables import load_reference_tables\n"
        "from gtex_explorer.ui.pages.overview import render_overview_page\n"
        "tables = load_reference_tables()\n"
        f"{changes}\n"
        "render_overview_page(tables)\n"
    )
    app = AppTest.from_string(source, default_timeout=60).run()
    assert not app.exception
    return app


def visible_html(app: AppTest) -> str:
    return "\n".join(item.value for item in [*app.markdown, *app.caption] if "<style>" not in item.value)


def test_overview_shows_saved_results_and_readable_labels():
    app = overview_app()
    text = visible_html(app)
    assert "L1 - C = 1" in text
    assert "1.000" in text and "494" in text
    assert "15 konfigurasi seleksi fitur + 1 baseline" in text
    assert "16 konfigurasi × 5 fold = 80 hasil evaluasi" in text
    assert "807 sampel - 494 donor" in text
    assert "193 sampel - 124 donor" in text
    assert "Donor unik" in text and ">618</div>" in text
    for value in ["96,25%", "95,49%", "96,37%"]:
        assert value in text
    # Macro-precision now sits in the holdout metrics table, beside the baseline.
    holdout = next(table.value for table in app.dataframe if "Model final" in table.value.columns)
    assert holdout.set_index("Metrik").loc["Macro-precision", "Model final"] == "97,55%"
    assert "Part A" not in text and "l1_c_1" not in text
    # The page says where the experiment comes from, right under its title.
    assert "skripsi Program Studi Teknik Informatika, Universitas Tarumanagara." in app.caption[0].value
    # The only tables are the method comparison in the Cross-validation tab and the holdout metrics.
    assert [len(table.value) for table in app.dataframe] == [4, 16, 5]


def test_overview_keeps_scope_notes_and_leaves_out_source_files():
    app = overview_app()
    text = visible_html(app)
    assert "Holdout dipisahkan sebelum cross-validation" in text
    assert "tidak digunakan untuk memilih konfigurasi" in text
    assert "dilatih ulang pada seluruh data development" in text
    assert "**Macro-F1:** 99,27% ± 0,49 poin persentase" in text
    assert "bukan interval kepercayaan atau skor holdout" in text
    # The file-by-file source notes are not part of the page.
    assert not any(item.label == "Asal angka dan rincian eksperimen" for item in app.expander)
    assert "run_summary.json" not in text


def test_cohort_numbers_reconcile_to_original_split_and_saved_cv_results():
    source = APP_DIR.parent / "results/notebook_run_500_1000_2000_l1_grid/part_a_only"
    if not (source / "sample_split.csv").is_file():
        pytest.skip("Original notebook export is not distributed with the portable app.")
    split = pd.read_csv(source / "sample_split.csv")
    summary = json.loads((APP_DIR / "data/run_summary.json").read_text())
    original = json.loads((source / "run_summary.json").read_text())
    assert original["experiment_fingerprint"] == summary["experiment_fingerprint"]
    assert len(split) == split.SAMPID.nunique() == summary["samples_loaded"]
    groups = {}
    for split_name, prefix in [("development", "development"), ("test", "test")]:
        rows = split.loc[split.split.eq(split_name)]
        assert len(rows) == summary[f"{prefix}_samples"]
        assert rows.SUBJID.nunique() == summary[f"{prefix}_donors"]
        groups[split_name] = set(rows.SUBJID)
    assert not groups["development"] & groups["test"]
    assert split.SUBJID.nunique() == summary["development_donors"] + summary["test_donors"]
    labels = pd.read_csv(APP_DIR / "data/label_mapping.csv")
    assert set(split.SMTSD) == set(labels.tissue)
    folds = pd.read_csv(APP_DIR / "data/experiment_cv_folds.csv")
    configurations = pd.read_csv(APP_DIR / "data/experiment_cv_summary.csv")
    assert not folds.duplicated(["configuration_id", "fold"]).any()
    assert folds.groupby("configuration_id").size().eq(summary["cv_folds"]).all()
    assert set(folds.configuration_id) == set(configurations.configuration_id)
    assert len(folds) == len(configurations) * summary["cv_folds"]
    holdout = pd.read_csv(APP_DIR / "data/final_holdout_predictions.csv")
    assert set(holdout.SAMPID) == set(split.loc[split.split.eq("test"), "SAMPID"])


def test_split_and_cv_labels_use_current_result_values():
    text = visible_html(overview_app(
        "tables['run_summary']['development_samples'] = 777\n"
        "tables['run_summary']['test_samples'] = 223\n"
        "tables['run_summary']['test_donors'] = 90\n"
        "tables['run_summary']['cv_folds'] = 4\n"
        "tables['cv_folds'] = tables['cv_folds'].iloc[:64]\n"
    ))
    assert "777 sampel - 494 donor" in text
    assert "223 sampel - 90 donor" in text
    assert ">584</div>" in text
    assert "16 konfigurasi × 4 fold = 64 hasil evaluasi" in text


@pytest.mark.parametrize("changes, expected", [
    ("tables['run_summary']['samples_loaded'] = 2345", "2.345"),
    ("tables['run_summary']['development_donors'] = 87", "807 sampel - 87 donor"),
    ("tables['final_test'].loc[0, 'macro_f1'] = 0.8123", "81,23%"),
    ("tables['labels'] = tables['labels'].iloc[:3]", ">3</div>"),
    ("tables['run_summary']['xgboost_parameters']['n_estimators'] = 42", "42 putaran boosting"),
])
def test_overview_changes_when_export_values_change(changes, expected):
    assert expected in visible_html(overview_app(changes))


def test_selector_label_is_not_fixed_to_l1():
    app = overview_app("tables['final_test'].loc[0, 'selector_id'] = 'anova'")
    text = visible_html(app)
    # The final model is named from the export; L1 still appears among the methods that were tested.
    assert '<p class="overview-model-identity">ANOVA, 1.000 gen, XGBoost</p>' in text
    assert "<dd>ANOVA</dd>" in text and "L1 - C = 1</dd>" not in text


def test_chart_keeps_full_names_zero_based_axis_and_saved_f1():
    app = overview_app()
    chart = tissue_chart(app)
    spec = json.loads(chart.proto.spec)
    encoding = {**spec.get("encoding", {}), **spec["layer"][0].get("encoding", {})}
    assert encoding["x"]["field"] == "f1-score"
    assert encoding["x"]["scale"]["domain"] == [0, 1]
    assert encoding["y"]["axis"]["labelLimit"] == 0
    assert len(encoding["y"]["sort"]) == 13
    assert "Adipose - Visceral (Omentum)" in encoding["y"]["sort"]
    assert "Heart - Left Ventricle" in encoding["y"]["sort"]
    assert spec["height"] == 13 * 30
    assert encoding["y"]["sort"][0] == "Artery - Coronary"
    assert any(layer.get("encoding", {}).get("text", {}).get("field") == "display_support" for layer in spec["layer"])
    assert any(layer["mark"]["type"] == "text" for layer in spec["layer"])

    changed = overview_app(
        "tables['per_tissue'].loc[tables['per_tissue']['tissue'].eq('Thyroid'), 'f1-score'] = 0.5"
    )
    changed_spec = json.loads(tissue_chart(changed).proto.spec)
    changed_encoding = {**changed_spec.get("encoding", {}), **changed_spec["layer"][0].get("encoding", {})}
    assert changed_encoding["y"]["sort"][0] == "Thyroid"
    changed.selectbox(key="overview_tissue_order").set_value("F1-score tertinggi").run()
    reordered = json.loads(tissue_chart(changed).proto.spec)
    reordered_encoding = {**reordered.get("encoding", {}), **reordered["layer"][0].get("encoding", {})}
    assert reordered_encoding["y"]["sort"][-1] == "Thyroid"
    changed.selectbox(key="overview_tissue_order").set_value("Nama jaringan").run()
    alphabetical = json.loads(tissue_chart(changed).proto.spec)
    alphabetical_encoding = {**alphabetical.get("encoding", {}), **alphabetical["layer"][0].get("encoding", {})}
    assert alphabetical_encoding["y"]["sort"] == sorted(encoding["y"]["sort"])


def tissue_chart(app: AppTest):
    """The per-tissue F1 chart of the holdout tab."""
    return next(chart for chart in app.get("vega_lite_chart") if "display_support" in chart.proto.spec)


def pipeline_html(app: AppTest, label: str = "Rincian alur eksperimen") -> str:
    return next(item.value for item in app.markdown if f'aria-label="{label}"' in item.value)


def test_pipeline_row_shows_each_step_with_one_key_fact_and_keeps_details_closed():
    app = overview_app()
    row = pipeline_html(app, "Alur eksperimen")
    steps = ["Data dan kohort", "Pembagian data utama", "Cross-validation", "Pemilihan konfigurasi",
             "Pelatihan ulang", "Evaluasi akhir"]
    positions = [row.index(f'<p class="pipeline-row-title">{step}</p>') for step in steps]
    assert positions == sorted(positions) and row.count('class="pipeline-row-step"') == 6
    for fact in ["1.000 sampel - 13 jaringan", "807 development - 193 holdout", "5 fold - 16 konfigurasi",
                 "Regresi logistik L1 - C = 1, 1.000 gen", "Seluruh 807 sampel development",
                 "Macro-F1 holdout <strong>96,25%</strong>"]:
        assert fact in row
    # The full text of every step sits in a closed expander, so the results stay near the top.
    details = next(item for item in app.expander if item.label == "Rincian setiap langkah")
    assert not details.proto.expanded
    assert any('aria-label="Rincian alur eksperimen"' in item.value for item in details.markdown)


def test_pipeline_lists_the_experiment_steps_in_order_from_the_exports():
    text = pipeline_html(overview_app())
    steps = ["Data dan kohort", "Pembagian data utama", "Cross-validation 5 fold pada data development",
             "Pemilihan konfigurasi", "Pelatihan ulang pada data development", "Evaluasi akhir pada holdout"]
    # Steps refer to each other by name, so find each step by its title.
    positions = [text.index(f'<p class="pipeline-title">{step}</p>') for step in steps]
    assert positions == sorted(positions)
    # The steps repeated in every fold sit inside step 3, in pipeline order.
    loop = [text.index(f"3.{number} {title}") for number, title in
            enumerate(["Pra-pemrosesan", "Seleksi fitur", "XGBoost", "Skor validation"], 1)]
    assert positions[2] < loop[0] < loop[-1] < positions[3] and loop == sorted(loop)
    for expected in ["1.000 sampel", "data development 807 sampel (494 donor)", "data holdout 193 sampel (124 donor)",
                     "TPM ≥ 1 pada sedikitnya 20% sampel training dan varians lebih dari 0",
                     "5.000 gen kandidat dengan varians tertinggi", "distandardisasi (z-score)",
                     "n_neighbors = 3", "C = 1 / 0,1 / 0,01", "500, 1.000, dan 2.000 gen teratas",
                     "terbentuk 15 konfigurasi", "16 konfigurasi × 5 fold = 80 hasil evaluasi",
                     "dari 8 konfigurasi seleksi fitur yang memenuhi syarat",
                     "Regresi logistik L1 - C = 1, 1.000 gen", "300 putaran boosting"]:
        assert expected in text
    # Plain wording, the same as the training guide.
    assert "seluruh sampel dari satu donor berada pada satu partisi saja" in text
    assert "diambil per jaringan agar setiap jaringan terwakili" in text
    assert "proses pra-pemrosesan dilakukan satu kali pada seluruh data development".lower() in text.lower()
    assert "tumpang tindih" not in text


@pytest.mark.parametrize("changes, expected", [
    ("tables['run_summary']['variance_prefilter_genes'] = 4000", "4.000 gen kandidat"),
    ("tables['run_summary']['l1_c_values'] = [1.0, 0.5]", "C = 1 / 0,5"),
    ("tables['run_summary']['feature_counts'] = [250, 750]", "250 dan 750 gen"),
    ("tables['run_summary']['test_donors'] = 99", "data holdout 193 sampel (99 donor)"),
])
def test_pipeline_follows_export_values(changes, expected):
    assert expected in pipeline_html(overview_app(changes))


def test_overview_says_the_results_come_from_the_thesis_experiment():
    app = overview_app()
    assert app.caption[0].value.startswith("Hasil di bawah ini berasal dari eksperimen perbandingan metode seleksi fitur")
    assert "tidak berubah oleh pelatihan di aplikasi" in app.caption[0].value


def test_holdout_tab_compares_the_final_model_with_the_xgboost_baseline_on_every_metric():
    from gtex_explorer.display import METRICS
    from gtex_explorer.experiments.reference_tables import load_reference_tables
    app = overview_app()
    text = visible_html(app)
    holdout = load_reference_tables()["final_holdout"]
    final = holdout.loc[holdout["experiment_type"].eq("feature_selection")].iloc[0]
    baseline = holdout.loc[holdout["experiment_type"].eq("baseline")].iloc[0]
    gap = f"{(final['macro_f1'] - baseline['macro_f1']) * 100:+.2f}".replace(".", ",")
    assert "Perbandingan dengan baseline XGBoost" in text
    assert f"selisih Macro-F1 <strong>{gap} poin persentase</strong>" in text
    chart = next(item for item in app.get("vega_lite_chart") if "Baseline XGBoost" in item.proto.spec)
    import pyarrow as pa
    data = pa.ipc.open_stream(chart.proto.datasets[0].data.data).read_all().to_pandas()
    values = {(row.metric, row.model): row.score for row in data.itertuples()}
    assert len(values) == 2 * len(METRICS)
    for field, label in METRICS.items():
        assert values[(label, "Model final")] == pytest.approx(float(final[field]))
        assert values[(label, "Baseline XGBoost")] == pytest.approx(float(baseline[field]))


def test_ringkasan_ends_with_the_gtex_data_source_and_its_reference():
    app = overview_app()
    source = next(item.value for item in app.markdown if 'aria-label="Sumber data"' in item.value)
    for expected in ["Genotype-Tissue Expression (GTEx) V11", 'href="https://gtexportal.org"',
                     "GTEx_Analysis_2025-08-22_v11_RNASeQCv2.4.3_gene_tpm_non_lcm.gct",
                     "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt",
                     "The GTEx Consortium atlas of genetic regulatory effects across human tissues",
                     'href="https://doi.org/10.1126/science.aaz1776"']:
        assert expected in source
    # It is the last thing on the page.
    assert 'aria-label="Sumber data"' in app.markdown[-1].value
