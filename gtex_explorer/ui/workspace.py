"""Workspace selection and session boundaries for the local application."""

import html
import importlib
import json
import sys
import threading
from pathlib import Path

import streamlit as st

from gtex_explorer.platform.cloud.media import image
from gtex_explorer.platform.paths import APP_ROOT
from gtex_explorer.platform.workspaces import (
    create_workspace,
    delete_workspace,
    last_workspace,
    list_workspaces,
    remember_workspace,
    rename_workspace,
    workspace_directory,
    workspace_roots,
)
from gtex_explorer.ui.shell.theme import page_style

_preload_started = False


# --- Library preload ---


def preload_libraries():
    """Load the heavy libraries in the background, so opening a workspace does not wait for them."""
    global _preload_started
    if _preload_started:
        return
    _preload_started = True

    def load():
        """Import each heavy library, leaving any failure for its page to report."""
        for name in [
            "numpy",
            "pandas",
            "altair",
            "sklearn.linear_model",
            "sklearn.feature_selection",
            "sklearn.metrics",
            "xgboost",
        ]:
            try:
                importlib.import_module(name)
            except ImportError:
                pass  # The page that needs the library reports the failure.

    threading.Thread(target=load, name="library-preload", daemon=True).start()


# --- Session and URL state ---


def upload_widget_key(name, state=None):
    """Return an uploader key that resets when the workspace changes."""
    state = st.session_state if state is None else state
    return f"{name}__{state.get('active_workspace_id', 'default')}__{state.get('workspace_generation', 0)}"


def activate_workspace(state, root, workspace_id):
    """Switch the session to a workspace and clear the previous one's state."""
    # Resolve/validate before changing any current session state.
    workspace_directory(root, workspace_id)
    if state.get("active_workspace_id") != workspace_id:
        for key in list(state):
            if key.startswith(("training_", "prediction_", "input_")) or key in {
                "audited_tpm",
                "probabilities",
                "predictions",
                "active_model_id",
                "active_run_id",
                "active_sample",
                "active_sample_selector",
                "offline_tpm_upload",
                "tpm_upload",
            }:
                state.pop(key, None)
        state["workspace_generation"] = state.get("workspace_generation", 0) + 1
    state["active_workspace_id"] = workspace_id
    remember_workspace(root, workspace_id)


PAGES = {
    "Pelatihan Model",
    "Prediksi Jaringan",
    "Hasil Pelatihan",
    "Perbandingan Pelatihan",
    "Panel Gen Pelatihan",
    "Kamus Biologi",
    "Ringkasan",
    "Panel Gen",
}
# Pages that were merged into another, so saved links still open the right place.
MERGED_PAGES = {"Perbandingan Metode": "Ringkasan"}


def restore_location(root):
    """Re-enter the workspace and page named in the URL once per new session.

    Session state lives in one server's memory. Hosted connections are cut at
    the platform's maximum duration and may reconnect to a fresh session, so
    the user's place is also kept in the URL, which the browser preserves.
    """
    state = st.session_state
    if state.get("location_restored"):
        return
    state.location_restored = True
    if state.get("workspace_entered") or not st.query_params.get("ws"):
        return
    try:
        activate_workspace(state, root, st.query_params["ws"])
    except (ValueError, OSError):
        st.query_params.clear()
        return
    state.workspace_entered = True
    page = MERGED_PAGES.get(st.query_params.get("page"), st.query_params.get("page"))
    state.page = page if page in PAGES else "Pelatihan Model"
    if st.query_params.get("model"):
        state.active_model_id = st.query_params["model"]


def remember_location():
    """Keep the URL in step with the current workspace, page and model."""
    location = {"ws": st.session_state.active_workspace_id, "page": st.session_state.page}
    # Right after a reconnect only the restored active_model_id is known.
    model = st.session_state.get("prediction_model") or st.session_state.get("active_model_id")
    if model:
        location["model"] = model
    if st.query_params.to_dict() != location:
        st.query_params.from_dict(location)


def current_workspace_root(root=None):
    """Return the open workspace's folder."""
    if root is None:
        from gtex_explorer.platform.desktop_runtime import state_directory

        root = state_directory()
    identity = st.session_state.get("active_workspace_id", last_workspace(root))
    return workspace_directory(root, identity)


# --- Dialogs ---


def _dismiss_dialog():
    """Forget which dialog was open."""
    st.session_state.pop("workspace_dialog", None)


@st.dialog("Workspace baru", on_dismiss=_dismiss_dialog)
def _new_workspace(root):
    """Show the dialog for creating a workspace."""
    with st.form("workspace_create_form"):
        name = st.text_input("Nama workspace", key="workspace_new_name", max_chars=80)
        include = st.checkbox(
            "Sertakan model bawaan",
            value=True,
            key="workspace_include_originals",
            help="Menyediakan model asli dan baseline. Dataset serta model Anda tetap terpisah.",
        )
        submitted = st.form_submit_button("Buat workspace", key="workspace_create", type="primary")
    if submitted:
        try:
            entry = create_workspace(root, name, include_bundled_models=include)
            st.session_state.workspace_pending = entry["id"]
            _dismiss_dialog()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


@st.dialog("Hapus workspace", on_dismiss=_dismiss_dialog)
def _delete_workspace(root, entry):
    """Show the dialog for deleting a workspace."""
    directory = workspace_directory(root, entry["id"])
    runs = len(list((directory / "runs").glob("run_*/status.json")))
    st.warning(
        f"**{entry['name']}** akan dihapus permanen, termasuk {runs} hasil pelatihan beserta modelnya "
        "dan semua dataset unggahannya. Tindakan ini tidak dapat dibatalkan."
    )
    with st.form("workspace_delete_form"):
        typed = st.text_input(
            "Ketik nama workspace untuk mengonfirmasi", max_chars=80, key="workspace_delete_name_" + entry["id"]
        )
        submitted = st.form_submit_button(
            "Hapus permanen", key="workspace_delete_confirm", type="primary", icon=":material/delete:"
        )
    if submitted:
        if typed.strip() != entry["name"]:
            st.error("Nama yang diketik tidak sama. Workspace tidak dihapus.")
            return
        try:
            delete_workspace(root, entry["id"])
            st.session_state.workspace_notice = f"Workspace {entry['name']} telah dihapus."
            _dismiss_dialog()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


@st.dialog("Ubah nama workspace", on_dismiss=_dismiss_dialog)
def _rename_workspace(root, entry):
    """Show the dialog for renaming a workspace."""
    with st.form("workspace_rename_form"):
        name = st.text_input(
            "Nama workspace", value=entry["name"], max_chars=80, key="workspace_rename_name_" + entry["id"]
        )
        submitted = st.form_submit_button("Simpan nama", key="workspace_save_name", type="primary")
    if submitted:
        try:
            rename_workspace(root, entry["id"], name)
            st.session_state.workspace_pending = entry["id"]
            _dismiss_dialog()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


# --- Training notice ---


def _open_process(root, workspace_id, run_id):
    """Jump to the running training process."""
    activate_workspace(st.session_state, root, workspace_id)
    st.session_state.workspace_entered = True
    st.session_state.training_selected_run = run_id
    st.session_state.training_run = run_id
    st.session_state.training_results_view = "Status proses"
    st.session_state.page = "Hasil Pelatihan"


@st.fragment(run_every=3)
def _training_notice(root):
    """Show a shortcut while training runs in any workspace."""
    for workspace, directory in workspace_roots(root):
        for path in (directory / "runs").glob("run_*/status.json"):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("status") not in {"running", "queued", "paused", "cancelling"}:
                    continue
                # Imported here because it loads the training libraries.
                from gtex_explorer.training.jobs import read_job

                record = read_job(path.parent.name, directory)
                if record.status in {"running", "queued", "paused", "cancelling"}:
                    st.caption(f"Proses training: {workspace['name']}")
                    if st.button(
                        "Buka proses",
                        key="workspace_active_process",
                        icon=":material/monitoring:",
                        on_click=_open_process,
                        args=(root, workspace["id"], record.run_id),
                    ):
                        st.rerun(scope="app")
                    return
            except (OSError, ValueError, KeyError):
                continue


# --- Chooser actions ---


def _enter_workspace(root, workspace_id):
    """Open a workspace from the chooser."""
    activate_workspace(st.session_state, root, workspace_id)
    st.session_state.workspace_entered = True
    st.session_state.page = "Pelatihan Model"


def _return_to_launcher():
    """Go back to the workspace chooser."""
    st.session_state.workspace_selection = st.session_state.active_workspace_id
    st.session_state.workspace_entered = False
    st.session_state.pop("workspace_search", None)
    st.query_params.clear()


def _select_workspace(workspace_id):
    """Highlight a workspace in the chooser."""
    st.session_state.workspace_selection = workspace_id


def _request_rename(workspace_id):
    """Open the rename dialog for a workspace."""
    st.session_state.workspace_rename_target = workspace_id
    st.session_state.workspace_dialog = "rename"


def _request_delete(workspace_id):
    """Open the delete dialog for a workspace."""
    st.session_state.workspace_delete_target = workspace_id
    st.session_state.workspace_dialog = "delete"


def leave_removed_workspace(root):
    """Send a session back to the launcher when another session removed its workspace."""
    state = st.session_state
    if state.get("workspace_entered") and state.get("active_workspace_id") not in {
        entry["id"] for entry in list_workspaces(root)
    }:
        state.workspace_entered = False
        state.pop("active_workspace_id", None)
        state.workspace_notice = "Workspace yang sedang dibuka telah dihapus."
        st.query_params.clear()


# --- Chooser screen ---


def render_workspace_launcher(root, brand_path):
    """Choose a project before rendering or loading the main application."""

    # Work out which workspace is selected.
    root = Path(root).resolve()
    st.session_state.setdefault("workspace_entered", False)
    entries = {entry["id"]: entry for entry in list_workspaces(root)}
    pending = st.session_state.pop("workspace_pending", None)
    if pending:
        st.session_state.workspace_selection = pending
        st.session_state.workspace_search = ""
    if st.session_state.get("workspace_selection") not in entries:
        st.session_state.workspace_selection = last_workspace(root)
    selected = st.session_state.workspace_selection

    st.html(page_style("workspace_launcher"))
    with st.container(key="workspace_launcher"):
        with st.container(
            horizontal=True, horizontal_alignment="center", vertical_alignment="center", key="workspace_brand"
        ):
            image(brand_path, width=44)
            st.markdown(
                '<div class="brand-title">GTEx Tissue Classifier</div>'
                '<div class="brand-subtitle">Analisis ekspresi gen lokal</div>',
                unsafe_allow_html=True,
            )

        # Title and the button for a new workspace.
        with st.container(key="workspace_heading"):
            title, create = st.columns([3, 1], vertical_alignment="center")
            with title:
                st.title("Pilih workspace")
                st.caption("Buka workspace yang tersedia atau buat workspace baru.")
                notice = st.session_state.pop("workspace_notice", None)
                if notice:
                    st.success(notice)
            with create:
                if st.button("Buat workspace", icon=":material/add:", key="workspace_new", width="stretch"):
                    st.session_state.workspace_dialog = "new"

        # Search box and result count.
        search, count = st.columns([4, 1], vertical_alignment="center")
        with search:
            query = st.text_input(
                "Cari workspace",
                key="workspace_search",
                placeholder="Cari nama workspace",
                icon=":material/search:",
                label_visibility="collapsed",
            )
        matching = [entry for entry in entries.values() if (query or "").strip().casefold() in entry["name"].casefold()]
        with count:
            st.caption(f"{len(matching)} dari {len(entries)} workspace")
        with st.container(key="workspace_table_header"):
            name, policy, actions = st.columns([3, 1.8, 2.6])
            name.caption("Workspace")
            policy.caption("Model bawaan")
            actions.caption("Tindakan")

        # Workspace list.
        with st.container(height=min(420, max(80, 68 * len(matching))), border=False, key="workspace_list"):
            if not matching:
                st.info("Tidak ada workspace yang cocok dengan pencarian.")
            for entry in matching:
                identity, full_name = entry["id"], entry["name"]
                short_name = full_name if len(full_name) <= 42 else full_name[:41].rstrip() + "…"
                row_key = "workspace_row_selected" if identity == selected else "workspace_row_" + identity
                with st.container(key=row_key):
                    name, policy, actions = st.columns([3, 1.8, 2.6], vertical_alignment="center")
                    with name:
                        st.button(
                            short_name,
                            icon=":material/check_circle:" if identity == selected else ":material/folder:",
                            type="tertiary",
                            help=full_name,
                            width="stretch",
                            key="workspace_select_" + identity,
                            on_click=_select_workspace,
                            args=(identity,),
                        )
                    policy.caption("Disertakan" if entry["include_bundled_models"] else "Tidak disertakan")
                    with actions, st.container(horizontal=True, gap="small"):
                        st.button(
                            "Ubah nama",
                            icon=":material/edit:",
                            type="tertiary",
                            key="workspace_rename_" + identity,
                            on_click=_request_rename,
                            args=(identity,),
                        )
                        if identity != "default":
                            st.button(
                                "Hapus",
                                icon=":material/delete:",
                                type="tertiary",
                                key="workspace_delete_" + identity,
                                on_click=_request_delete,
                                args=(identity,),
                            )

        # The selected workspace and the button that opens it.
        with st.container(key="workspace_footer"):
            summary, enter = st.columns([3, 1], vertical_alignment="center")
            with summary:
                st.markdown(
                    '<div class="workspace-selected-name">Workspace dipilih: <strong>'
                    + html.escape(entries[selected]["name"])
                    + "</strong></div>",
                    unsafe_allow_html=True,
                )
                st.caption("Dataset, model, dan hasil training disimpan terpisah.")
            with enter:
                st.button(
                    "Buka workspace",
                    type="primary",
                    icon=":material/arrow_forward:",
                    icon_position="right",
                    key="workspace_open",
                    width="stretch",
                    on_click=_enter_workspace,
                    args=(root, selected),
                )
        _training_notice(root)

    # Whichever dialog was requested.
    if st.session_state.get("workspace_dialog") == "new":
        _new_workspace(root)
    elif st.session_state.get("workspace_dialog") == "rename":
        # Another session may have removed the workspace while this dialog was open.
        if st.session_state.workspace_rename_target in entries:
            _rename_workspace(root, entries[st.session_state.workspace_rename_target])
        else:
            _dismiss_dialog()
    elif st.session_state.get("workspace_dialog") == "delete":
        if st.session_state.workspace_delete_target in entries:
            _delete_workspace(root, entries[st.session_state.workspace_delete_target])
        else:
            _dismiss_dialog()


# --- Sidebar controls ---


def render_workspace_controls(root):
    """Compact project context inside the application, not another chooser."""
    entries = {entry["id"]: entry for entry in list_workspaces(root)}
    entry = entries[st.session_state.active_workspace_id]
    with st.container(key="workspace_context"):
        context, action = st.columns([3, 1.7], gap="xsmall", vertical_alignment="center", wrap=False)
        with context:
            st.caption("Workspace aktif")
            with st.container(horizontal=True, wrap=False, gap="xsmall", vertical_alignment="center"):
                st.markdown(":material/folder:", width="content")
                st.markdown(
                    '<div class="workspace-context-name" title="'
                    + html.escape(entry["name"], quote=True)
                    + '">'
                    + html.escape(entry["name"])
                    + "</div>",
                    unsafe_allow_html=True,
                    width="stretch",
                )
        with action:
            st.button(
                "Ganti",
                icon=":material/swap_horiz:",
                key="workspace_change",
                help="Kembali ke pemilihan workspace",
                on_click=_return_to_launcher,
                type="tertiary",
            )
    _training_notice(root)


# --- Workspace step ---


def enter_workspace() -> None:
    """Show the chooser until a workspace is open. The main app runs only after this returns."""
    from gtex_explorer.platform.desktop_runtime import state_directory

    root = state_directory()
    restore_location(root)
    leave_removed_workspace(root)
    if not st.session_state.get("workspace_entered", False):
        # The chooser needs none of the heavy libraries, so it appears while they load behind it.
        preload_libraries()
        render_workspace_launcher(root, APP_ROOT / "assets" / "brand_dna.png")
        st.stop()
    # A session restored from its URL skips the chooser and can arrive before the libraries are ready.
    if "xgboost" not in sys.modules:
        with st.spinner("Memuat aplikasi..."):
            importlib.import_module("gtex_explorer.inference.prediction")
