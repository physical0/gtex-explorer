from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.analysis.core import load_model_bundle, parse_expression_file, audit_and_align_input, predict_tissues
from gtex_explorer.ui.prediction_page import _probability_chart, _probability_percent


APP_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value, expected", [(0, "0,00%"), (1, "100,00%"), (0.0053, "0,53%"), (0.00001, "<0,01%")])
def test_probability_formats_small_values_without_claiming_zero(value, expected):
    assert _probability_percent(value) == expected


def test_chart_retains_all_classes_sorted_with_zero_based_scale_and_visible_values():
    labels = [f"Class {index}" for index in range(13)]
    values = pd.Series([0.988] + [0.001] * 12, index=labels)
    chart = _probability_chart(values, labels[0]).to_dict()
    rows = next(iter(chart["datasets"].values()))
    assert len(rows) == 13
    assert sum(row["probability"] for row in rows) == pytest.approx(1)
    assert all(row["percentage_label"] for row in rows)
    bar, text = chart["layer"]
    assert bar["encoding"]["x"]["scale"]["domain"] == [0, 1]
    assert text["encoding"]["text"]["field"] == "percentage_label"
    assert bar["encoding"]["y"]["axis"]["labelOverlap"] is False
    assert bar["encoding"]["y"]["axis"]["labelLimit"] == 0
    assert chart["height"] >= 13 * 30
    assert bar["encoding"]["y"]["sort"] == labels


def test_real_demo_summary_matches_chart_and_sample_selection_changes_both():
    app = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60).run()
    app = enter_application(app)
    app.button(key="load_local_demo").click().run()
    app.button(key="run_prediction").click().run()
    assert not app.exception
    probabilities = app.session_state["probabilities"].copy()
    for sample in probabilities.index:
        app.selectbox(key="active_sample_selector").set_value(sample).run()
        assert not app.exception
        values = probabilities.loc[sample].sort_values(ascending=False, kind="stable")
        text = "\n".join(item.value for item in [*app.markdown, *app.caption])
        for value in values.iloc[:3]:
            assert _probability_percent(value) in text
        assert str(values.index[2]) in text and "Peringkat ketiga" in text
        assert "kelas lainnya" not in text and "Probabilitas gabungan" not in text
        assert "total sekitar 100%" in text
        assert "poin persentase." in text and sample in text
        assert "belum dikalibrasi untuk penggunaan klinis" not in text
        assert "Bukan bukti hubungan sebab-akibat biologis" not in text
        assert "tidak ditujukan untuk keputusan klinis" not in text
        assert "Referensi lokasi jaringan" not in [item.label for item in app.expander]
        assert len(app.get("vega_lite_chart")) == 1
        spec = json.loads(app.get("vega_lite_chart")[0].proto.spec)
        dataset = app.get("vega_lite_chart")[0].proto.datasets[0]
        rows = pa.ipc.open_stream(dataset.data.data).read_all().to_pandas().to_dict("records")
        assert len(rows) == 13
        assert [row["tissue"] for row in rows] == values.index.tolist()
        np.testing.assert_allclose([row["probability"] for row in rows], values.values)
        assert len(app.dataframe[-1].value) == 5


def test_pending_upload_signature_does_not_read_the_file():
    # Reading and hashing a 200 MB upload on every rerun copied it each time.
    import streamlit as st
    from gtex_explorer.ui.prediction_page import _sync_uploaded_input

    class Upload:
        name, file_id, size = 'big.csv', 'id-1', 10

        def getvalue(self):
            raise AssertionError('upload content read during rerun')

    st.session_state.clear()
    _sync_uploaded_input(Upload())
    assert st.session_state.pending_upload_signature == ('big.csv', 'id-1', 10)
