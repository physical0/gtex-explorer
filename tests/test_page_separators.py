"""Reader-facing separators must not change algorithms or glossary matching."""
import pytest
from streamlit.testing.v1 import AppTest
from ui_helpers import enter_application

from gtex_explorer.platform.paths import APP_ROOT as APP_DIR


@pytest.mark.parametrize("page", ["Prediksi Jaringan", "Ringkasan", "Perbandingan Metode", "Panel Gen", "Kamus Biologi"])
def test_pages_do_not_use_middle_dot_or_pipe_separators(page):
    app = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60).run()
    app = enter_application(app)
    app.button(key=f"nav_{page}").click().run()
    assert not app.exception
    text = "\n".join(str(item.value) for item in [*app.markdown, *app.caption, *app.info])
    assert "·" not in text
    assert "|" not in text
    for table in app.dataframe:
        if "eligibility_reason" in table.value:
            assert not table.value.eligibility_reason.astype(str).str.contains("|", regex=False).any()


def test_cv_diagnostic_copy_and_glossary_aliases_remain_readable():
    app = AppTest.from_file(str(APP_DIR / "main.py"), default_timeout=60).run()
    app = enter_application(app)
    app.button(key="nav_Panel Gen").click().run()
    app.radio(key="gene_panel_source").set_value("Cross-validation").run()
    for method in ["anova", "mutual_information", "l1_c_1"]:
        app.selectbox(key="gene_cv_method").set_value(method).run()
        assert not app.exception
        text = "\n".join(str(item.value) for item in [*app.markdown, *app.caption])
        assert "·" not in text and "|" not in text
    app.button(key="nav_Kamus Biologi").click().run()
    assert not app.exception
    text = "\n".join(item.value for item in app.caption)
    assert "|" not in text and "·" not in text
