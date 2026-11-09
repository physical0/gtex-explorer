"""Application styling."""

from pathlib import Path

import streamlit as st

STYLES = Path(__file__).resolve().parents[1] / "styles"


def page_style(name: str) -> str:
    """Return one stylesheet from ui/styles, ready for st.markdown or st.html."""
    return f"<style>\n{(STYLES / f'{name}.css').read_text(encoding='utf-8')}</style>"


def apply_theme() -> None:
    """Add the application styles to the page."""
    # workspace.css overrides base.css, so the order matters.
    for name in ("base", "workspace"):
        st.markdown(page_style(name), unsafe_allow_html=True)
