"""Application styling."""

from pathlib import Path

import streamlit as st

STYLES = Path(__file__).resolve().parents[1] / "styles"


def apply_theme() -> None:
    """Add the application styles to the page."""
    # workspace.css overrides base.css, so the order matters.
    for name in ("base.css", "workspace.css"):
        st.markdown(f"<style>\n{(STYLES / name).read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
