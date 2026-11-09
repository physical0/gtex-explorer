"""The main app: what the user sees once a workspace is open. Each page is in gtex_explorer.ui.pages."""
from __future__ import annotations

from gtex_explorer.desktop_runtime import state_directory
from gtex_explorer.paths import APP_ROOT
from gtex_explorer.ui.gene_page import render_gene_page
from gtex_explorer.ui.glossary import render_glossary_page
from gtex_explorer.ui.overview_page import render_overview_page
from gtex_explorer.ui.pages.comparison import render_comparison_page
from gtex_explorer.ui.prediction_page import render_prediction_page
from gtex_explorer.ui.shell.navigation import navigation
from gtex_explorer.ui.shell.reference_data import load_tables_or_stop
from gtex_explorer.ui.training_gene_page import render_training_gene_page
from gtex_explorer.ui.training_page import preserve_training_form, render_training_page, render_training_results_page
from gtex_explorer.ui.workspace_ui import current_workspace_root, remember_location


def run_app() -> None:
    tables = load_tables_or_stop()
    # Must run before navigation(), which reruns early.
    preserve_training_form()
    page = navigation()
    remember_location()
    workspace = current_workspace_root(state_directory())
    training_data = APP_ROOT / 'training_data'

    routes = {
        "Pelatihan Model": lambda: render_training_page(training_data, workspace),
        "Hasil Pelatihan": lambda: render_training_results_page(training_data, workspace),
        "Panel Gen Pelatihan": lambda: render_training_gene_page(workspace),
        "Ringkasan": lambda: render_overview_page(tables),
        "Perbandingan Metode": lambda: render_comparison_page(tables),
        "Panel Gen": lambda: render_gene_page(tables),
        "Kamus Biologi": render_glossary_page,
        "Kamus Istilah": render_glossary_page,
    }
    # Prediksi Jaringan is the default page. It loads the model chosen in its own selector.
    routes.get(page, lambda: render_prediction_page(None, None, tables))()
