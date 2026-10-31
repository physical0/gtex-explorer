"""Workspace selection and session boundaries for the local application."""
import json
import html
from pathlib import Path

import streamlit as st
from gtex_explorer.cloud.media import image

from gtex_explorer.workspaces import (create_workspace, last_workspace, list_workspaces,
                        remember_workspace, rename_workspace, workspace_directory, workspace_roots)


def upload_widget_key(name, state=None):
    state = st.session_state if state is None else state
    return f"{name}__{state.get('active_workspace_id', 'default')}__{state.get('workspace_generation', 0)}"


def activate_workspace(state, root, workspace_id):
    # Resolve/validate before changing any current session state.
    workspace_directory(root, workspace_id)
    if state.get('active_workspace_id') != workspace_id:
        for key in list(state):
            if (key.startswith(('training_', 'prediction_', 'input_')) or key in {
                    'audited_tpm', 'probabilities', 'predictions', 'active_model_id',
                    'active_run_id', 'active_sample', 'active_sample_selector',
                    'offline_tpm_upload', 'tpm_upload'}):
                state.pop(key, None)
        state['workspace_generation'] = state.get('workspace_generation', 0) + 1
    state['active_workspace_id'] = workspace_id
    remember_workspace(root, workspace_id)


def current_workspace_root(root=None):
    if root is None:
        from gtex_explorer.desktop_runtime import state_directory
        root = state_directory()
    identity = st.session_state.get('active_workspace_id', last_workspace(root))
    return workspace_directory(root, identity)


def _dismiss_dialog():
    st.session_state.pop('workspace_dialog', None)


@st.dialog('Workspace baru', on_dismiss=_dismiss_dialog)
def _new_workspace(root):
    with st.form('workspace_create_form'):
        name = st.text_input('Nama workspace', key='workspace_new_name', max_chars=80)
        include = st.checkbox('Sertakan model bawaan', value=True, key='workspace_include_originals',
                              help='Menyediakan model asli dan baseline. Dataset serta model Anda tetap terpisah.')
        submitted = st.form_submit_button('Buat workspace', key='workspace_create', type='primary')
    if submitted:
        try:
            entry = create_workspace(root, name, include_bundled_models=include)
            st.session_state.workspace_pending = entry['id']
            _dismiss_dialog()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


@st.dialog('Ubah nama workspace', on_dismiss=_dismiss_dialog)
def _rename_workspace(root, entry):
    with st.form('workspace_rename_form'):
        name = st.text_input('Nama workspace', value=entry['name'], max_chars=80,
                             key='workspace_rename_name_' + entry['id'])
        submitted = st.form_submit_button('Simpan nama', key='workspace_save_name', type='primary')
    if submitted:
        try:
            rename_workspace(root, entry['id'], name)
            st.session_state.workspace_pending = entry['id']
            _dismiss_dialog()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(str(error))


def _open_process(root, workspace_id, run_id):
    activate_workspace(st.session_state, root, workspace_id)
    st.session_state.workspace_entered = True
    st.session_state.training_selected_run = run_id
    st.session_state.training_run = run_id
    st.session_state.training_results_view = 'Status proses'
    st.session_state.page = 'Hasil Pelatihan'


@st.fragment(run_every=3)
def _training_notice(root):
    from gtex_explorer.training.jobs import read_job
    for workspace, directory in workspace_roots(root):
        for path in (directory / 'runs').glob('run_*/status.json'):
            try:
                value = json.loads(path.read_text(encoding='utf-8'))
                if value.get('status') not in {'running', 'queued', 'paused', 'cancelling'}:
                    continue
                record = read_job(path.parent.name, directory)
                if record.status in {'running', 'queued', 'paused', 'cancelling'}:
                    st.caption(f"Proses training: {workspace['name']}")
                    if st.button('Buka proses', key='workspace_active_process', icon=':material/monitoring:',
                                 on_click=_open_process, args=(root, workspace['id'], record.run_id)):
                        st.rerun(scope='app')
                    return
            except (OSError, ValueError, KeyError):
                continue


def _enter_workspace(root, workspace_id):
    activate_workspace(st.session_state, root, workspace_id)
    st.session_state.workspace_entered = True
    st.session_state.page = 'Pelatihan Model'


def _return_to_launcher():
    st.session_state.workspace_selection = st.session_state.active_workspace_id
    st.session_state.workspace_entered = False
    st.session_state.pop('workspace_search', None)


def _select_workspace(workspace_id):
    st.session_state.workspace_selection = workspace_id


def _request_rename(workspace_id):
    st.session_state.workspace_rename_target = workspace_id
    st.session_state.workspace_dialog = 'rename'


def render_workspace_launcher(root, brand_path):
    """Choose a project before rendering or loading the main application."""
    root = Path(root).resolve()
    st.session_state.setdefault('workspace_entered', False)
    entries = {entry['id']: entry for entry in list_workspaces(root)}
    pending = st.session_state.pop('workspace_pending', None)
    if pending:
        st.session_state.workspace_selection = pending
        st.session_state.workspace_search = ''
    if st.session_state.get('workspace_selection') not in entries:
        st.session_state.workspace_selection = last_workspace(root)
    selected = st.session_state.workspace_selection

    st.html('''<style>
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {display:none;}
        .st-key-workspace_launcher {max-width:1000px; margin:0 auto; padding-top:8vh;}
        .st-key-workspace_brand {margin-bottom:2.8rem;}
        .st-key-workspace_brand .brand-title {font-size:1.45rem;}
        .st-key-workspace_brand .brand-subtitle {font-size:.9rem;}
        .st-key-workspace_heading {margin-bottom:.8rem;}
        .st-key-workspace_table_header {padding:.45rem .85rem; background:#f7f8f8;}
        .st-key-workspace_list {border:1px solid #e2e8ec; border-radius:8px;}
        .st-key-workspace_list, [data-testid="stLayoutWrapper"]:has(> .st-key-workspace_list) {
            max-height:clamp(120px, calc(100dvh - 545px), 420px);}
        [class*="st-key-workspace_row_"] {padding:.5rem .85rem; border-bottom:1px solid #e2e8ec;}
        .st-key-workspace_row_selected {background:#eef8f8;}
        [class*="st-key-workspace_row_"] button {justify-content:flex-start;}
        [class*="st-key-workspace_select_"] button > div {justify-content:flex-start;}
        [class*="st-key-workspace_select_"] button {font-weight:600;}
        .st-key-workspace_row_selected button[kind="tertiary"] {color:#087173;}
        .st-key-workspace_footer {padding-top:1rem;}
        .workspace-selected-name {font-size:.95rem; color:#657184; overflow-wrap:anywhere;}
        .workspace-selected-name strong {color:#14213d;}
        @media(max-width:700px) {
          .st-key-workspace_launcher {padding-top:1rem;}
          .st-key-workspace_brand {margin-bottom:1.4rem;}
        }
    </style>''')
    with st.container(key='workspace_launcher'):
        with st.container(horizontal=True, horizontal_alignment='center', vertical_alignment='center',
                          key='workspace_brand'):
            image(brand_path, width=44)
            st.markdown('<div class="brand-title">GTEx Tissue Classifier</div>'
                        '<div class="brand-subtitle">Analisis ekspresi gen lokal</div>',
                        unsafe_allow_html=True)
        with st.container(key='workspace_heading'):
            title, create = st.columns([3, 1], vertical_alignment='center')
            with title:
                st.title('Pilih workspace')
                st.caption('Buka workspace yang tersedia atau buat workspace baru.')
            with create:
                if st.button('Buat workspace', icon=':material/add:', key='workspace_new', width='stretch'):
                    st.session_state.workspace_dialog = 'new'
        search, count = st.columns([4, 1], vertical_alignment='center')
        with search:
            query = st.text_input('Cari workspace', key='workspace_search', placeholder='Cari nama workspace',
                                  icon=':material/search:', label_visibility='collapsed')
        matching = [entry for entry in entries.values() if (query or '').strip().casefold() in entry['name'].casefold()]
        with count:
            st.caption(f'{len(matching)} dari {len(entries)} workspace')
        with st.container(key='workspace_table_header'):
            name, policy, actions = st.columns([3.3, 2.7, 1.4])
            name.caption('Workspace')
            policy.caption('Model bawaan')
            actions.caption('Tindakan')
        with st.container(height=min(420, max(80, 68 * len(matching))), border=False,
                          key='workspace_list'):
            if not matching:
                st.info('Tidak ada workspace yang cocok dengan pencarian.')
            for entry in matching:
                identity, full_name = entry['id'], entry['name']
                short_name = full_name if len(full_name) <= 42 else full_name[:41].rstrip() + '…'
                row_key = 'workspace_row_selected' if identity == selected else 'workspace_row_' + identity
                with st.container(key=row_key):
                    name, policy, actions = st.columns([3.3, 2.7, 1.4], vertical_alignment='center')
                    with name:
                        st.button(short_name, icon=':material/check_circle:' if identity == selected else ':material/folder:',
                                  type='tertiary', help=full_name, width='stretch',
                                  key='workspace_select_' + identity, on_click=_select_workspace, args=(identity,))
                    policy.caption('Disertakan' if entry['include_bundled_models'] else 'Tidak disertakan')
                    with actions:
                        st.button('Ubah nama', icon=':material/edit:', type='tertiary',
                                  key='workspace_rename_' + identity, on_click=_request_rename, args=(identity,))
        with st.container(key='workspace_footer'):
            summary, enter = st.columns([3, 1], vertical_alignment='center')
            with summary:
                st.markdown('<div class="workspace-selected-name">Workspace dipilih: <strong>'
                            + html.escape(entries[selected]['name']) + '</strong></div>', unsafe_allow_html=True)
                st.caption('Dataset, model, dan hasil training disimpan terpisah.')
            with enter:
                st.button('Buka workspace', type='primary', icon=':material/arrow_forward:', icon_position='right',
                          key='workspace_open', width='stretch', on_click=_enter_workspace, args=(root, selected))
        _training_notice(root)
    if st.session_state.get('workspace_dialog') == 'new':
        _new_workspace(root)
    elif st.session_state.get('workspace_dialog') == 'rename':
        _rename_workspace(root, entries[st.session_state.workspace_rename_target])


def render_workspace_controls(root):
    """Compact project context inside the application, not another chooser."""
    entries = {entry['id']: entry for entry in list_workspaces(root)}
    entry = entries[st.session_state.active_workspace_id]
    with st.container(key='workspace_context'):
        context, action = st.columns([3, 1.7], gap='xsmall', vertical_alignment='center', wrap=False)
        with context:
            st.caption('Workspace aktif')
            with st.container(horizontal=True, wrap=False, gap='xsmall', vertical_alignment='center'):
                st.markdown(':material/folder:', width='content')
                st.markdown('<div class="workspace-context-name" title="'
                            + html.escape(entry['name'], quote=True) + '">'
                            + html.escape(entry['name']) + '</div>', unsafe_allow_html=True, width='stretch')
        with action:
            st.button('Ganti', icon=':material/swap_horiz:', key='workspace_change',
                      help='Kembali ke pemilihan workspace', on_click=_return_to_launcher, type='tertiary')
    _training_notice(root)
