"""Application skeleton: joins the parts in the order the user meets them.

1. Page setup and theme
2. Workspace step (gtex_explorer.ui.workspace)
3. Main app (gtex_explorer.ui.app)
"""

import streamlit as st

from gtex_explorer.platform.paths import APP_ROOT
from gtex_explorer.ui.shell.theme import apply_theme
from gtex_explorer.ui.workspace import enter_workspace

st.set_page_config(
    page_title="GTEx Tissue Classifier",
    page_icon=str(APP_ROOT / "assets" / "brand_dna.png"),
    layout="wide",
    initial_sidebar_state="auto",
)
apply_theme()
enter_workspace()

# The main app pulls in the heavy libraries, so it is imported only after a workspace is open.
from gtex_explorer.ui.app import run_app

run_app()
