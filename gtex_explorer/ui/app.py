"""The main app: the frame every page sits in once a workspace is open.

This file only decides which page to show. What a page draws is in gtex_explorer.ui.pages,
and the work behind it is in the inference, training, and experiments packages.
"""

from __future__ import annotations

from gtex_explorer.platform.desktop_runtime import state_directory
from gtex_explorer.platform.paths import TRAINING_DATA_DIR
from gtex_explorer.ui.pages.comparison import render_comparison_page
from gtex_explorer.ui.pages.genes import render_gene_page
from gtex_explorer.ui.pages.glossary import render_glossary_page
from gtex_explorer.ui.pages.overview import render_overview_page
from gtex_explorer.ui.pages.prediction import render_prediction_page
from gtex_explorer.ui.pages.training import render_training_page
from gtex_explorer.ui.pages.training_genes import render_training_gene_page
from gtex_explorer.ui.pages.training_results import render_training_results_page
from gtex_explorer.ui.shell.form_state import keep_form_values
from gtex_explorer.ui.shell.navigation import navigation
from gtex_explorer.ui.shell.reference_data import load_tables_or_stop
from gtex_explorer.ui.workspace import current_workspace_root, remember_location

DEFAULT_PAGE = "Prediksi Jaringan"


def run_app() -> None:
    """Draw the navigation and the current page."""
    tables = load_tables_or_stop()
    # Must run before navigation(), which reruns early.
    keep_form_values()
    page = navigation()
    remember_location()
    workspace = current_workspace_root(state_directory())

    routes = {
        DEFAULT_PAGE: lambda: render_prediction_page(tables),
        "Pelatihan Model": lambda: render_training_page(TRAINING_DATA_DIR, workspace),
        "Hasil Pelatihan": lambda: render_training_results_page(TRAINING_DATA_DIR, workspace),
        "Panel Gen Pelatihan": lambda: render_training_gene_page(workspace),
        "Ringkasan": lambda: render_overview_page(tables),
        "Perbandingan Metode": lambda: render_comparison_page(tables),
        "Panel Gen": lambda: render_gene_page(tables),
        "Kamus Biologi": render_glossary_page,
    }
    routes.get(page, routes[DEFAULT_PAGE])()
