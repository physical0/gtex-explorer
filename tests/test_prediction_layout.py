"""Behavior of the approved compact prediction layout and existing page groups."""
import io
from pathlib import Path
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.analysis.core import load_model_bundle

APP = Path(__file__).resolve().parents[1] / "main.py"


class Upload(io.BytesIO):
    name = "two_samples.csv"
    # Streamlit's UploadedFile gets a new file_id for every upload.
    file_id = "two-samples"

    @property
    def size(self):
        return len(self.getvalue())


def new_app():
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    app = enter_application(app)
    assert not app.exception
    return app


def summary_table(app):
    markup = next(item.value for item in app.markdown if '<table' in item.value and 'prediction-audit-summary' in item.value)
    root = ET.fromstring(markup)
    columns = [cell.text for cell in root.findall('./thead/tr/th')]
    rows = [[cell.text for cell in row] for row in root.findall('./tbody/tr')]
    return pd.DataFrame(rows, columns=columns)


def test_navigation_groups_existing_notebook_and_training_routes():
    app = new_app()
    groups = {item.label: item for item in app.sidebar.expander}
    navigation = [item for item in app.sidebar.button if item.key and item.key.startswith('nav_')]
    assert [item.label for item in navigation][:2] == ["Latih Model", "Prediksi Jaringan"]
    assert app.button(key='workspace_change')
    assert set(groups) == {"Referensi Eksperimen", "Gallery Hasil Pelatihan"}
    assert not groups["Referensi Eksperimen"].proto.expanded
    assert not groups["Gallery Hasil Pelatihan"].proto.expanded
    assert [item.label for item in groups["Referensi Eksperimen"].button] == [
        "Ringkasan", "Perbandingan Metode", "Panel Gen Eksperimen",
    ]
    assert [item.label for item in groups["Gallery Hasil Pelatihan"].button] == [
        "Hasil Pelatihan", "Panel Gen",
    ]
    app.button(key="nav_Ringkasan").click().run()
    assert not app.exception
    assert next(item for item in app.sidebar.expander if item.label == "Referensi Eksperimen").proto.expanded
    app.button(key="nav_Prediksi Jaringan").click().run()
    assert app.title[0].value == "Prediksi jaringan"


def test_three_audit_measures_follow_selected_sample_without_conflating_zeros():
    pre, _ = load_model_bundle()
    genes = list(pre["selected_genes"][:3]) + ["ENSG_TEST_OTHER"]
    raw = pd.DataFrame([[5., 0., float('nan'), 2.], [0., 0., 3., 0.]],
                       index=pd.Index(["S1", "S2"], name="SAMPID"), columns=genes).to_csv().encode()
    app = new_app()
    with patch("streamlit.file_uploader", return_value=Upload(raw)):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        summary = summary_table(app)
        assert summary["Pemeriksaan"].tolist() == [
            "Kecocokan ID gen model", "Kelengkapan nilai gen model", "Gen input dengan TPM > 0",
        ]
        assert summary["Hasil"].tolist() == ["3 / 1.000 (0,3%)", "2 / 1.000 (0,2%)", "2 / 4 (50,0%)"]
        app.selectbox(key="active_sample_selector").set_value("S2").run()
        assert not app.exception
        assert summary_table(app)["Hasil"].tolist() == [
            "3 / 1.000 (0,3%)", "3 / 1.000 (0,3%)", "1 / 4 (25,0%)",
        ]
        assert not app.button(key="run_prediction").disabled


def test_compact_info_replaces_repetitive_success_and_coverage_notices():
    app = new_app()
    with patch("streamlit.file_uploader", return_value=Upload(b'SAMPID,ENSG_TEST_OTHER\nS1,0\n')):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        assert not app.exception
        info = "\n".join(item.value for item in app.info)
        assert "Informasi gen model" in info
        assert "median" in info
        assert not any("Lolos pemeriksaan" in item.value for item in app.markdown)
        assert any(item.label == "Detail pemeriksaan input" for item in app.expander)
        assert app.number_input(key="prediction_positive_percent").value == 90
        assert app.number_input(key="prediction_completeness_percent").value == 90
        assert not app.button(key="run_prediction").disabled
        assert any("Input siap diprediksi" in item.value for item in app.markdown)


def test_reminder_controls_exist_even_before_an_input_is_audited():
    app = new_app()
    assert app.number_input(key="prediction_positive_percent").min == 0
    assert app.number_input(key="prediction_positive_percent").max == 100
    assert app.number_input(key="prediction_completeness_percent").max == 100
    assert app.button(key="run_prediction_disabled").disabled


def test_threshold_chosen_before_first_audit_is_used_and_not_reset():
    app = new_app()
    with patch("streamlit.file_uploader", return_value=Upload(b'SAMPID,ENSG_TEST_A,ENSG_TEST_B\nS1,5,0\n')):
        app.run()
        app.number_input(key="prediction_positive_percent").set_value(50).run()
        app.button(key="inspect_uploaded_input").click().run()
        assert not app.exception
        audit = app.session_state["input_audit"]
        assert audit.positive_expression_threshold == .5
        assert audit.low_positive_sample_count == 0
        assert app.number_input(key="prediction_positive_percent").value == 50


@pytest.mark.parametrize("key", ["prediction_positive_percent", "prediction_completeness_percent"])
def test_empty_reminder_field_uses_last_valid_setting_without_breaking_prediction(key):
    app = new_app()
    app.button(key="load_local_demo").click().run()
    app.number_input(key=key).set_value(50).run()
    app.number_input(key=key).set_value(None).run()
    assert not app.exception
    assert not app.button(key="run_prediction").disabled
    assert any("0–100%" in item.value for item in app.caption)
    audit = app.session_state["input_audit"]
    threshold = audit.positive_expression_threshold if key == "prediction_positive_percent" else audit.completeness_threshold
    assert threshold == .5


def test_reminder_preferences_survive_navigation_without_changing_selected_model():
    app = new_app()
    app.selectbox(key="prediction_model").set_value("original_baseline_5000").run()
    app.number_input(key="prediction_positive_percent").set_value(40).run()
    app.number_input(key="prediction_completeness_percent").set_value(70).run()
    app.button(key="nav_Ringkasan").click().run()
    app.button(key="nav_Prediksi Jaringan").click().run()
    assert not app.exception
    assert app.number_input(key="prediction_positive_percent").value == 40
    assert app.number_input(key="prediction_completeness_percent").value == 70
    assert app.selectbox(key="prediction_model").value == "original_baseline_5000"
