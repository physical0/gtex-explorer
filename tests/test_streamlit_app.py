from __future__ import annotations

from pathlib import Path
import io
import json
import uuid
from unittest.mock import patch

import pandas as pd

import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.display import SERIES_COLORS


APP_FILE = Path(__file__).resolve().parents[1] / "main.py"


def load_app() -> AppTest:
    app = AppTest.from_file(str(APP_FILE), default_timeout=60)
    app.run()
    return enter_application(app)


def test_prediction_opens_after_entry_and_reference_navigation() -> None:
    app = load_app()
    assert app.title[0].value == "Prediksi jaringan"
    assert app.button(key='workspace_change')
    app.button(key="nav_Ringkasan").click().run()
    app.button(key='nav_Prediksi Jaringan').click().run()
    assert not app.exception
    assert app.title[0].value == "Prediksi jaringan"


def test_prediction_page_audits_demo_before_running_model() -> None:
    app = load_app()
    assert app.selectbox(key="prediction_boosting_rounds").value == 300
    assert "predictions" not in app.session_state

    app.button(key="load_local_demo").click().run()
    assert not app.exception
    assert app.session_state["input_audit"].ready
    assert app.selectbox(key="active_sample_selector").value
    assert "predictions" not in app.session_state

    app.button(key="run_prediction").click().run()
    assert not app.exception
    assert app.session_state["predictions"] is not None
    assert len(app.get("vega_lite_chart")) == 1
    assert any("result-name" in markdown.value for markdown in app.markdown)


@pytest.mark.parametrize(
    ("navigation_key", "expected_title"),
    [
        ("nav_Ringkasan", "Ringkasan eksperimen"),
        ("nav_Panel Gen", "Panel gen eksperimen"),
    ],
)
def test_navigation_pages_load_without_exceptions(
    navigation_key: str,
    expected_title: str,
) -> None:
    app = load_app()
    app.button(key=navigation_key).click().run()
    assert not app.exception
    assert expected_title in [title.value for title in app.title]
    sidebar_text = "\n".join(
        element.value for element in [*app.sidebar.markdown, *app.sidebar.caption]
    )
    for removed_text in [
        "Berjalan sepenuhnya di komputer ini",
        "Tidak mengirim matriks TPM ke layanan eksternal",
        "Model lokal siap",
        "model lokal siap",
    ]:
        assert removed_text not in sidebar_text


def test_ringkasan_displays_the_method_comparison_and_all_gene_baseline() -> None:
    app = load_app()
    app.button(key="nav_Ringkasan").click().run()
    assert not app.exception
    visible_text = "\n".join(
        element.value
        for element in [*app.markdown, *app.info, *app.caption]
        if isinstance(element.value, str)
    )
    assert "Baseline memakai seluruh 5.000 gen kandidat tanpa seleksi fitur" in visible_text
    assert "80 hasil fold-level" in visible_text
    assert [tab.label for tab in app.tabs] == ["Cross-validation", "Hasil akhir - holdout"]


def test_old_links_to_perbandingan_metode_open_ringkasan() -> None:
    app = load_app()
    app.session_state["page"] = "Perbandingan Metode"
    app.run()
    assert not app.exception
    assert app.session_state["page"] == "Ringkasan"
    assert "Ringkasan eksperimen" in [title.value for title in app.title]


class Upload(io.BytesIO):
    def __init__(self, content: bytes, name: str = "sample.csv"):
        super().__init__(content)
        self.name = name
        self.size = len(content)
        # Streamlit's UploadedFile gets a new file_id for every upload.
        self.file_id = uuid.uuid4().hex


def demo_bytes() -> bytes:
    return (APP_FILE.parent / "data/demo_tpm.csv").read_bytes()


@pytest.mark.parametrize("query", ["[", "(", "*", ".", "Thyroid"])
def test_gene_search_is_literal_and_does_not_crash(query) -> None:
    app = load_app()
    app.button(key="nav_Panel Gen").click().run()
    app.text_input[0].set_value(query).run()
    assert not app.exception
    displayed = app.dataframe[0].value
    if not displayed.empty:
        assert (displayed["gene"].str.contains(query, case=False, regex=False) | displayed["dominant_tissue"].fillna("").str.contains(query, case=False, regex=False)).all()


def test_changing_or_removing_upload_invalidates_old_results() -> None:
    app = load_app()
    original = demo_bytes()
    # A new pending upload must invalidate even an active demo prediction.
    app.button(key="load_local_demo").click().run()
    app.button(key="run_prediction").click().run()
    with patch("streamlit.file_uploader", return_value=Upload(original)):
        app.run()
        assert "predictions" not in app.session_state
        assert "input_audit" not in app.session_state
        app.button(key="inspect_uploaded_input").click().run()
        assert app.session_state["input_audit"].ready
        # A second audit must work even if a file cursor is already at EOF.
        app.button(key="inspect_uploaded_input").click().run()
        assert not app.exception
        app.button(key="run_prediction").click().run()
        assert "predictions" in app.session_state
    changed = original.replace(b"GTEX", b"TEST", 1)
    assert len(changed) == len(original) and changed != original
    with patch("streamlit.file_uploader", return_value=Upload(changed)):
        app.run()
        assert "predictions" not in app.session_state
        assert "input_audit" not in app.session_state
        app.button(key="inspect_uploaded_input").click().run()
        app.button(key="run_prediction").click().run()
        assert "predictions" in app.session_state
    with patch("streamlit.file_uploader", return_value=None):
        app.run()
        assert not app.exception
        for key in ["predictions", "probabilities", "audited_tpm", "input_audit"]:
            assert key not in app.session_state
        assert app.button(key="run_prediction_disabled").disabled
        assert len(app.get("download_button")) == 0


@pytest.mark.parametrize("value", [0., None, "not_a_number"])
def test_only_invalid_numeric_upload_is_blocked(value) -> None:
    app = load_app()
    matrix = pd.read_csv(io.BytesIO(demo_bytes()), index_col=0).iloc[:1].copy().astype(object)
    matrix.loc[:, :] = value
    with patch("streamlit.file_uploader", return_value=Upload(matrix.to_csv().encode())):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        assert not app.exception
        assert "predictions" not in app.session_state
        if value == "not_a_number":
            assert app.error and "non-numerik" in app.error[0].value
            assert app.button(key="run_prediction_disabled").disabled
        else:
            assert app.session_state["input_audit"].ready
            app.button(key="run_prediction").click().run()
            assert not app.exception
            assert len(app.session_state["predictions"]) == 1


def test_reminder_threshold_reaudits_original_values_without_discarding_predictions() -> None:
    app = load_app()
    matrix = pd.read_csv(io.BytesIO(demo_bytes()), index_col=0).iloc[:1].copy()
    matrix.iloc[:, :200] = float('nan')
    with patch("streamlit.file_uploader", return_value=Upload(matrix.to_csv().encode())):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        assert app.session_state["input_audit"].below_threshold_sample_count == 1
        before = app.session_state["audited_tpm"].copy()
        app.button(key="run_prediction").click().run()
        predictions = app.session_state["predictions"].copy()
        app.number_input(key="prediction_completeness_percent").set_value(80).run()
        assert not app.exception
        assert app.session_state["input_audit"].below_threshold_sample_count == 0
        assert app.session_state["input_audit"].sample_quality.iloc[0]["observed_fraction"] == .8
        pd.testing.assert_frame_equal(before, app.session_state["audited_tpm"])
        pd.testing.assert_frame_equal(predictions, app.session_state["predictions"])
        app.number_input(key="prediction_completeness_percent").set_value(100).run()
        assert app.session_state["input_audit"].below_threshold_sample_count == 1


def test_zero_overlap_upload_can_predict_with_explicit_imputation_reminder() -> None:
    app = load_app()
    with patch("streamlit.file_uploader", return_value=Upload(b'SAMPID,ENSG_OTHER\nS1,5\n')):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        assert app.session_state["input_audit"].ready
        assert app.session_state["input_audit"].matched_gene_count == 0
        assert any("median" in info.value for info in app.info)
        app.button(key="run_prediction").click().run()
        assert not app.exception
        assert len(app.session_state["predictions"]) == 1
        assert abs(app.session_state["probabilities"].iloc[0].sum() - 1) < 1e-6


def test_nonzero_threshold_does_not_use_model_gene_match_percentage() -> None:
    app = load_app()
    raw = b'SAMPID,ENSG_OTHER1,ENSG_OTHER2\nS1,5,0\n'
    with patch("streamlit.file_uploader", return_value=Upload(raw)):
        app.run()
        app.button(key="inspect_uploaded_input").click().run()
        assert app.session_state["input_audit"].matched_fraction == 0
        assert app.session_state["input_audit"].sample_quality.iloc[0]["uploaded_positive_fraction"] == .5
        assert app.session_state["input_audit"].low_positive_sample_count == 1
        app.number_input(key="prediction_positive_percent").set_value(50).run()
        assert not app.exception
        assert app.session_state["input_audit"].low_positive_sample_count == 0
        assert app.session_state["input_audit"].matched_fraction == 0
        assert not app.button(key="run_prediction").disabled


def test_overview_has_no_dummy_and_chart_has_selector_legend() -> None:
    app = load_app()
    app.button(key="nav_Ringkasan").click().run()
    overview_text = "\n".join(markdown.value for markdown in app.markdown)
    assert "XGBoost" in overview_text
    assert "DummyClassifier" not in overview_text
    for metric in ["Macro-F1", "Macro-precision", "Macro-recall", "Balanced accuracy", "Accuracy"]:
        app.selectbox(key="comparison_metric").set_value(metric).run()
        assert not app.exception
        spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
        color_encodings = [layer.get("encoding", {}).get("color") for layer in spec["layer"]]
        selector_encodings = [color for color in color_encodings if color and color.get("field") == "selector_label"]
        assert selector_encodings
        # The key is drawn outside the plot, in the selectors' fixed order and colours.
        assert all(color["legend"] is None for color in selector_encodings)
        assert all(color["scale"]["range"] == SERIES_COLORS for color in selector_encodings)
        legend = next(markdown.value for markdown in app.markdown if 'class="method-legend"' in markdown.value)
        order = selector_encodings[0]["scale"]["domain"]
        assert order == ["ANOVA", "Mutual Information", "Regresi logistik L1 - C = 1",
                         "Regresi logistik L1 - C = 0,1", "Regresi logistik L1 - C = 0,01"]
        assert [legend.index(label) for label in order] == sorted(legend.index(label) for label in order)
        # No translucent bands: the spread is drawn as thin SD bars.
        assert not any(layer.get("mark", {}).get("type") == "errorband" for layer in spec["layer"])


def test_app_is_named_htx_tissue_classifier():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py"), default_timeout=60).run()
    # The workspace screen and, once a workspace is open, the menu show the name.
    assert any("HTx Tissue Classifier" in item.value for item in app.markdown)
    assert not any("GTEx Tissue Classifier" in item.value for item in app.markdown)
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert 'page_title="HTx Tissue Classifier"' in source
