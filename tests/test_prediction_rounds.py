"""Use one boosting-round range for probabilities, explanations, and exports."""
from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from gtex_explorer.analysis.core import (
    DATA_DIR,
    audit_and_align_input,
    explain_prediction,
    load_model_bundle,
    parse_expression_file,
    predict_tissues,
)
from test_streamlit_app import load_app


@pytest.fixture(scope="module")
def prediction_inputs():
    preprocessor, classifier = load_model_bundle()
    with (DATA_DIR / "demo_tpm.csv").open("rb") as handle:
        matrix = parse_expression_file(handle, "demo_tpm.csv")
    aligned, audit = audit_and_align_input(matrix, preprocessor)
    assert audit.ready
    values = (
        np.log2(aligned.to_numpy(dtype=float) + 1.0)
        - np.asarray(preprocessor["scaler_mean"])
    ) / np.asarray(preprocessor["scaler_scale"])
    return preprocessor, classifier, aligned, values


@pytest.mark.parametrize("rounds", [100, 200, 300])
def test_round_range_matches_native_probabilities_and_contributions(prediction_inputs, rounds):
    preprocessor, classifier, aligned, values = prediction_inputs
    expected = classifier.get_booster().predict(
        xgb.DMatrix(values), iteration_range=(0, rounds),
    )
    predictions, probabilities = predict_tissues(
        aligned, preprocessor, classifier, boosting_rounds=rounds,
    )
    np.testing.assert_allclose(probabilities.to_numpy(), expected, rtol=1e-6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert predictions.boosting_rounds.tolist() == [rounds] * 3
    assert predictions.predicted_tissue.tolist() == probabilities.idxmax(axis=1).tolist()
    sample = str(aligned.index[0])
    explanation = explain_prediction(
        aligned, sample, preprocessor, classifier, limit=5, boosting_rounds=rounds,
    )
    native = classifier.get_booster().predict(
        xgb.DMatrix(values[:1]), pred_contribs=True, iteration_range=(0, rounds),
    )[0, int(expected[0].argmax()), :-1]
    genes = np.asarray(preprocessor["selected_genes"], dtype=str)
    order = np.lexsort((genes, -np.abs(native)))[:5]
    assert explanation.gene.tolist() == genes[order].tolist()
    np.testing.assert_allclose(explanation.contribution.to_numpy(), native[order], rtol=1e-6)


def test_default_prediction_is_the_unchanged_full_model(prediction_inputs):
    preprocessor, classifier, aligned, values = prediction_inputs
    expected = classifier.predict_proba(values)
    before = bytes(classifier.get_booster().save_raw(raw_format="json"))
    predictions, probabilities = predict_tissues(aligned, preprocessor, classifier)
    np.testing.assert_array_equal(probabilities.to_numpy(), expected)
    assert predictions.boosting_rounds.eq(300).all()
    for rounds in [100, 200]:
        predict_tissues(aligned, preprocessor, classifier, boosting_rounds=rounds)
        explain_prediction(
            aligned, str(aligned.index[0]), preprocessor, classifier,
            boosting_rounds=rounds,
        )
    assert bytes(classifier.get_booster().save_raw(raw_format="json")) == before


@pytest.mark.parametrize("rounds", [0, -1, 301, 1.5, True, "100"])
@pytest.mark.parametrize("operation", ["predict", "explain"])
def test_invalid_rounds_are_rejected_before_inference(prediction_inputs, rounds, operation):
    preprocessor, classifier, aligned, _ = prediction_inputs
    with pytest.raises(ValueError, match="putaran"):
        if operation == "predict":
            predict_tissues(aligned, preprocessor, classifier, boosting_rounds=rounds)
        else:
            explain_prediction(
                aligned, str(aligned.index[0]), preprocessor, classifier,
                boosting_rounds=rounds,
            )


def test_round_selector_defaults_to_full_model_without_bypassing_input_audit():
    app = load_app()
    control = app.selectbox(key="prediction_boosting_rounds")
    assert control.value == 300
    assert control.options == ["100", "200", "300"]
    control.set_value(100).run()
    assert app.button(key="run_prediction_disabled").disabled
    assert "predictions" not in app.session_state


def test_changing_rounds_invalidates_results_and_updates_explanation_and_csv(prediction_inputs):
    preprocessor, classifier, _, _ = prediction_inputs
    app = load_app()
    app.button(key="load_local_demo").click().run()
    app.button(key="run_prediction").click().run()
    assert app.session_state["predictions"].boosting_rounds.eq(300).all()
    full_probabilities = app.session_state["probabilities"].copy()
    for rounds in [100, 200, 300]:
        app.selectbox(key="prediction_boosting_rounds").set_value(rounds).run()
        assert not app.exception
        assert "predictions" not in app.session_state
        assert "probabilities" not in app.session_state
        assert app.session_state["input_audit"].ready
        assert not app.get("download_button")
        app.button(key="run_prediction").click().run()
        assert not app.exception
        assert app.session_state["predictions"].boosting_rounds.eq(rounds).all()
        text = "\n".join(item.value for item in [*app.markdown, *app.caption])
        assert f"{rounds} dari 300 putaran" in text
        explanation = explain_prediction(
            app.session_state["audited_tpm"], app.session_state["active_sample"],
            preprocessor, classifier, limit=5, boosting_rounds=rounds,
        )
        shown = app.dataframe[-1].value
        assert shown["ID Gen"].tolist() == explanation.gene.tolist()
        np.testing.assert_allclose(shown["Kontribusi model"], explanation.contribution)
        with patch("streamlit.download_button") as download:
            app.run()
            payload = next(
                call.kwargs["data"] for call in download.call_args_list
                if call.kwargs.get("key") == "download_predictions"
            )
        exported = pd.read_csv(io.BytesIO(payload))
        assert exported.boosting_rounds.tolist() == [rounds] * len(app.session_state['audited_tpm'])
        probabilities = app.session_state["probabilities"]
        np.testing.assert_allclose(exported[probabilities.columns], probabilities.to_numpy())
        if rounds < 300:
            assert not np.allclose(probabilities, full_probabilities)
        else:
            pd.testing.assert_frame_equal(probabilities, full_probabilities)


def test_returning_to_prediction_cannot_mix_cached_results_with_another_round_count():
    app = load_app()
    app.button(key="load_local_demo").click().run()
    app.selectbox(key="prediction_boosting_rounds").set_value(100).run()
    app.button(key="run_prediction").click().run()
    app.button(key="nav_Ringkasan").click().run()
    app.button(key="nav_Prediksi Jaringan").click().run()
    assert not app.exception
    if "predictions" in app.session_state:
        assert app.session_state["predictions"].boosting_rounds.eq(
            app.selectbox(key="prediction_boosting_rounds").value
        ).all()
    else:
        assert not app.get("download_button")
