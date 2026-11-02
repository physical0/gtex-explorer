"""Training setup and recorded results for the user-training workspace."""
from pathlib import Path
import json
import os
import time
import psutil

import pandas as pd
import streamlit as st
from gtex_explorer.cloud.media import download_button

from gtex_explorer.training.config import training_presets, validate_training_config
from gtex_explorer.training.data import load_training_data, make_cv_assignments
from gtex_explorer.training.jobs import start_job, read_job, request_cancel, resume_job, pause_job, unpause_job, can_resume, delete_run, has_active_run, ACTIVE_STATUSES, FINALIZING_STAGES
from gtex_explorer.training.gallery import list_gallery
from gtex_explorer.training.uploads import audit_uploads, read_upload_table, save_training_upload
from gtex_explorer.error_messages import user_error
from gtex_explorer.ui.workspace_ui import upload_widget_key
from gtex_explorer.cloud.uploads import file_uploader

METHODS = {'l1': 'Regresi logistik L1', 'anova': 'ANOVA',
           'mutual_information': 'Mutual Information', 'baseline': 'Baseline - tanpa seleksi supervised'}
METRICS = {'macro_f1': 'Macro-F1', 'macro_precision': 'Macro-precision',
           'macro_recall': 'Macro-recall', 'balanced_accuracy': 'Balanced accuracy', 'accuracy': 'Accuracy'}


@st.cache_resource
def _data(root: str, version: tuple):
    return load_training_data(Path(root))


def cached_data(root):
    root = Path(root)
    version = tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                    for p in sorted(root.iterdir()) if p.is_file())
    return _data(str(root), version)


def _preset_changed():
    preset = training_presets().get(st.session_state.training_preset)
    if preset:
        st.session_state.training_selector = preset['selector']
        st.session_state.training_top_k = preset['top_k']
        st.session_state.training_l1_c = float(preset.get('l1_c', 1))


TRAINING_FORM_KEYS = ('training_preset', 'training_selector', 'training_candidates', 'training_top_k', 'training_l1_c',
                      'training_l1_iterations', 'training_l1_tol', 'training_l1_weight', 'training_mi_neighbors',
                      'training_cv', 'training_folds', 'training_xgb_threads',
                      *('training_xgb_' + name for name in ('n_estimators', 'learning_rate', 'max_depth', 'min_child_weight',
                                                            'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'gamma')))


def _range_help(low, high):
    def number(value):
        return f'{value:,}'.replace(',', '.') if isinstance(value, int) else f'{value:g}'.replace('.', ',')
    return f'Rentang {number(low)}–{number(high)}.'


def _selection_changed():
    st.session_state.training_preset = 'Custom'


def method_description(config):
    text = METHODS.get(config['selector'], config['selector'])
    if config['selector'] == 'l1':
        text += f" - C = {config['l1_c']:g}"
    elif config['selector'] == 'mutual_information':
        text += f" - n_neighbors = {config['mi_neighbors']}"
    return f"{text} - {config['top_k']:,} gen".replace(',', '.')


def run_created(path):
    """Start time saved with the run; older runs fall back to the file time."""
    config = Path(path) / 'config.json'
    try:
        created = json.loads(config.read_text(encoding='utf-8')).get('created')
        if isinstance(created, (int, float)):
            return float(created)
    except (OSError, ValueError):
        pass
    try:
        return config.stat().st_mtime
    except OSError:
        return 0.


def run_label(path):
    try:
        saved = json.loads((path / 'config.json').read_text(encoding='utf-8'))
        dataset = saved.get('dataset', {})
        filename = dataset.get('source', {}).get('expression_file', 'Data notebook')
        stamp = time.strftime('%d/%m %H:%M', time.localtime(run_created(path)))
        return f"{stamp} - {method_description(saved['config'])} - {filename} - {path.name[-6:]}"
    except (OSError, ValueError, KeyError):
        return f"Riwayat tidak dapat dibaca - {path.name[-6:]}"


def choose_run(state_root, key, *, gallery_only=False):
    paths = sorted((Path(state_root) / 'runs').glob('run_*/status.json'),
                   key=lambda p: run_created(p.parent), reverse=True)
    if gallery_only:
        issues = []
        ready = {entry['run_id'] for entry in list_gallery(state_root, issues=issues)}
        paths = [path for path in paths if path.parent.name in ready]
        for issue in issues:
            st.warning(f"Hasil {issue['run_id'][-6:]} belum lolos verifikasi: {user_error(issue['reason'])}")
    if not paths:
        st.info('Belum ada model terverifikasi di gallery. Periksa Status proses.' if gallery_only
                else 'Belum ada pelatihan pengguna. Mulai dari halaman Latih Model.')
        return None
    roots = {p.parent.name: p.parent for p in paths}
    remembered = st.session_state.get('training_selected_run')
    if st.session_state.get(key) not in roots:
        st.session_state[key] = remembered if remembered in roots else next(iter(roots))
    selected = st.selectbox('Pilih hasil pelatihan', list(roots),
                            format_func=lambda x: run_label(roots[x]), key=key)
    st.session_state.training_selected_run = selected
    return selected


def show_diagnostics(info):
    labels = {'selector': 'Metode', 'candidate_count': 'Gen kandidat', 'requested_k': 'Gen dipilih',
              'n_neighbors': 'Jumlah tetangga MI', 'positive_score_gene_count': 'Gen dengan skor MI positif',
              'converged': 'Solver konvergen', 'iterations': 'Iterasi solver', 'l1_c': 'C',
              'nonzero_gene_count': 'Gen berkoefisien nonzero', 'sparsity': 'Sparsitas gen',
              'training_samples': 'Sampel training', 'validation_samples': 'Sampel validation'}
    rows = []
    for key, label in labels.items():
        if key not in info:
            continue
        value = info[key]
        if key == 'selector':
            value = METHODS.get(value, value)
        elif key == 'sparsity':
            value = f'{value:.2%}'
        elif type(value) is bool:
            value = 'Ya' if value else 'Tidak'
        rows.append({'Diagnostik': label, 'Nilai': str(value)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    if info.get('legacy_scores_unavailable'):
        st.caption('Riwayat ini hanya menyimpan daftar gen. Skor dan diagnostik fold tidak tersedia.')
    if info.get('selector') == 'baseline':
        st.caption('Baseline menggunakan seluruh gen kandidat. Urutan daftar bukan peringkat kepentingan.')


def _dataset_strip(dataset):
    columns = st.columns([3, 1, 1, 1, 1.3], vertical_alignment='center')
    with columns[0]:
        st.markdown('**' + ('Data eksperimen notebook' if dataset['name'] == 'Data eksperimen notebook' else 'Data pengguna') + '**')
        st.caption(dataset['name'] + '  \n' + dataset.get('metadata_name', ''))
    for column, label, value in zip(columns[1:], ['Sampel', 'Donor', 'Jaringan', 'Gen input'],
                                     [dataset['samples'], dataset['donors'], dataset['classes'], dataset['genes']]):
        with column:
            st.markdown('**' + f'{value:,}'.replace(',', '.') + '**')
            st.caption(label)


def render_training_page(data_root: Path, state_root: Path) -> None:
    st.html('''<style>
    .st-key-training_review {border-left:1px solid #dce3eb; padding-left:24px;}
    .st-key-training_review h3 {font-size:1.12rem;}
    .st-key-training_setup h1 {font-size:2.4rem !important; line-height:1.15;}
    .st-key-training_setup h3 {font-size:1.3rem !important;}
    .st-key-training_setup hr {margin:1rem 0 !important;}
    .st-key-training_setup [data-testid="stVerticalBlock"] {gap:.7rem;}
    @media(max-width:800px) {.st-key-training_review {border-left:0; padding-left:0;}}
    </style>''')
    with st.container(key='training_setup'):
        _render_training_setup(data_root, state_root)


TRAINING_STEPS = ['1. Data training', '2. Konfigurasi', '3. Tinjau dan latih']


def _set_training_step(step):
    st.session_state.training_step = step


def _invalidate_training_upload():
    # File changes/removals invalidate the audited matrix, not only its local view.
    st.session_state.pop('training_upload_audit', None)
    st.session_state.pop('training_upload_signature', None)
    st.session_state.training_data_ready = False


def _render_training_setup(data_root: Path, state_root: Path) -> None:
    st.title('Latih model')
    st.caption('Bangun model dari data ekspresi Anda.')
    data_tab, configuration_tab, review_tab = st.tabs(TRAINING_STEPS,
        key='training_step', on_change='rerun')
    audit = st.session_state.get('training_upload_audit')
    data = None
    dataset = None
    with data_tab:
        source = st.radio('Sumber data', ['Data pengguna', 'Data eksperimen notebook'],
                          horizontal=True, key='training_source')
        if source == 'Data pengguna':
            st.subheader('Unggah data training')
            st.caption('Matriks TPM dan metadata berlabel. ID sampel harus sama pada kedua berkas.')
            first, second = st.columns(2)
            with first:
                expression = file_uploader('Matriks ekspresi TPM', type=['csv', 'tsv', 'txt', 'gct'],
                    key=upload_widget_key('training_expression_upload'), on_change=_invalidate_training_upload)
            with second:
                metadata = file_uploader('Metadata sampel', type=['csv', 'tsv', 'txt'],
                    key=upload_widget_key('training_metadata_upload'), on_change=_invalidate_training_upload)
            if expression is not None and metadata is not None:
                # Upload IDs change with every new file; hashing both files each rerun copied them.
                signature = f'{expression.file_id}:{metadata.file_id}'
                try:
                    columns = list(read_upload_table(metadata.getvalue(), metadata.name).columns)
                    st.caption('Pilihan pada setiap kolom di bawah diambil dari judul kolom pada berkas metadata.')
                    mapping = st.columns(3)
                    sample = mapping[0].selectbox('Kolom ID sampel', columns,
                        index=columns.index('SAMPID') if 'SAMPID' in columns else 0, key='training_sample_column')
                    mapping[0].caption('ID Sampel menghubungkan metadata dengan matriks TPM. ID harus sama persis.')
                    donor = mapping[1].selectbox('Kolom donor', columns,
                        index=columns.index('SUBJID') if 'SUBJID' in columns else min(1, len(columns)-1), key='training_donor_column')
                    mapping[1].caption('Sampel dari donor yang sama tidak dipisah antara data pelatihan dan validasi.')
                    tissue = mapping[2].selectbox('Kolom jaringan', columns,
                        index=columns.index('SMTSD') if 'SMTSD' in columns else min(2, len(columns)-1), key='training_tissue_column')
                    mapping[2].caption('Label jaringan yang dipelajari dan diprediksi model.')
                    st.caption('Periksa kembali kolom donor dan jaringan. Pilihan yang keliru tidak selalu terdeteksi aplikasi.')
                    signature += f':{sample}:{donor}:{tissue}'
                    if st.session_state.get('training_upload_signature') != signature:
                        audit = None
                        st.session_state.pop('training_upload_audit', None)
                    if st.button('Periksa data training', type='primary', key='training_audit'):
                        st.session_state.pop('training_upload_audit', None)
                        try:
                            with st.spinner('Memeriksa matriks dan mencocokkan metadata...'):
                                audit = audit_uploads(expression.getvalue(), expression.name,
                                    metadata.getvalue(), metadata.name, sample, donor, tissue)
                            st.session_state.training_upload_audit = audit
                            st.session_state.training_upload_signature = signature
                            st.session_state.training_data_ready = True
                            st.success('Data telah diperiksa. Lanjutkan ke konfigurasi.')
                        except (ValueError, OSError, UnicodeError) as error:
                            audit = None
                            st.error(str(error))
                except (ValueError, OSError, UnicodeError) as error:
                    audit = None
                    st.session_state.pop('training_upload_audit', None)
                    st.error(f'Metadata tidak dapat dibaca: {error}')
            elif expression is not None or metadata is not None:
                audit = None
                st.info('Lengkapi kedua berkas sebelum memeriksa data training.')
            if audit is not None:
                dataset = {'name': audit.source['expression_file'], 'metadata_name': audit.source['metadata_file'],
                           'samples': len(audit.sample_ids), 'donors': len(set(audit.donor_ids)),
                           'classes': len(audit.classes), 'genes': len(audit.gene_names)}
                _dataset_strip(dataset)
                with st.expander('Distribusi jaringan dan contoh metadata'):
                    counts = pd.Series([audit.classes[i] for i in audit.labels]).value_counts().rename_axis('Jaringan').reset_index(name='Sampel')
                    st.dataframe(counts, hide_index=True)
                    st.dataframe(pd.DataFrame({'SAMPID': audit.sample_ids, 'Donor': audit.donor_ids,
                                               'Jaringan': [audit.classes[i] for i in audit.labels]}).head(10), hide_index=True)
        else:
            try:
                data = cached_data(data_root)
                dataset = {'name': 'Data eksperimen notebook', 'metadata_name': 'Subset development tersimpan',
                           'samples': len(data.sample_ids), 'donors': len(set(data.donor_ids)),
                           'classes': len(data.classes), 'genes': len(data.raw_gene_names)}
                _dataset_strip(dataset)
                st.caption('Menggunakan subset development eksperimen. Sampel holdout tidak dimasukkan.')
            except (OSError, ValueError, KeyError) as error:
                st.error(f'Data notebook tidak tersedia: {error}')
        st.divider()
        st.button('Lanjut ke konfigurasi', key='training_to_configuration', type='primary',
                  icon=':material/arrow_forward:', on_click=_set_training_step,
                  args=(TRAINING_STEPS[1],), disabled=dataset is None, width='stretch')
    with configuration_tab:
        if dataset:
            _dataset_strip(dataset)
        else:
            st.info('Unggah dan periksa TPM beserta metadata pada tab Data training.')
        st.divider()
        available = min(5000, dataset['genes']) if dataset else 5000
        if st.session_state.get('training_top_k', 0) is not None and st.session_state.get('training_top_k', 0) > available:
            st.session_state.training_top_k = available
            st.session_state.training_preset = 'Custom'
        st.subheader('Seleksi fitur')
        selector_left, selector_right = st.columns(2)
        with selector_left:
            st.selectbox('Preset seleksi fitur', [*training_presets(), 'Custom'],
                         key='training_preset', on_change=_preset_changed)
        with selector_right:
            method = st.selectbox('Metode seleksi fitur', list(METHODS), format_func=METHODS.get,
                                  key='training_selector', on_change=_selection_changed)
        with selector_right:
            st.session_state.setdefault('training_candidates', available)
            if data is not None:
                st.session_state.training_candidates = int(data.manifest['candidate_count'])
            elif st.session_state.training_candidates is not None and st.session_state.training_candidates > available:
                st.session_state.training_candidates = available
            count = st.number_input('Gen kandidat', min_value=1, max_value=available,
                value=None,
                key='training_candidates', disabled=data is not None,
                on_change=_selection_changed,
                help='Jumlah gen setelah filter teknis dan pemeringkatan varians. Dipelajari dari data training setiap fold.')
        candidate_blank = count is None
        if candidate_blank:
            count = available
        if method == 'baseline':
            st.session_state.training_top_k = count
        elif st.session_state.get('training_top_k') is not None and st.session_state.training_top_k > count:
            st.session_state.training_top_k = count
        st.session_state.setdefault('training_top_k', min(1000, count))
        with selector_left:
            k = st.number_input('Top-k gen', value=None, step=1, help=_range_help(1, count),
                key='training_top_k', disabled=method == 'baseline', on_change=_selection_changed)
        values = {'selector': method, 'top_k': k}
        if method == 'l1':
            with selector_right:
                st.session_state.setdefault('training_l1_c', 1.)
                values['l1_c'] = st.number_input('C (regularisasi)', value=None, step=.1,
                    format='%.4f', help=_range_help(.0001, 1000), key='training_l1_c', on_change=_selection_changed)
            with st.expander('Opsi selector'):
                values['l1_max_iter'] = st.number_input('Maksimum iterasi', value=20000, step=1, help=_range_help(100, 50000),
                                                     key='training_l1_iterations')
                values['l1_tol'] = st.number_input('Toleransi', value=.001, step=.0001, format='%.6f',
                                                help=_range_help(.000001, .01), key='training_l1_tol')
                values['l1_class_weight'] = st.selectbox('Bobot kelas', ['balanced', None],
                    format_func=lambda x: x or 'None', key='training_l1_weight')
        elif method == 'mutual_information':
            with selector_right:
                values['mi_neighbors'] = st.number_input('n_neighbors', value=3, step=1, help=_range_help(1, 20), key='training_mi_neighbors')
        st.divider()
        st.subheader('Evaluasi')
        cv_left, cv_right = st.columns([2, 1])
        with cv_left:
            values['evaluate_cv'] = st.checkbox('Evaluasi dengan cross-validation', key='training_cv')
        with cv_right:
            values['cv_folds'] = st.selectbox('Jumlah fold', [5, 3, 2, 4, 6, 7, 8, 9, 10],
                key='training_folds', disabled=not values['evaluate_cv'])
        st.caption('Donor data training dan data validation dipisahkan pada setiap fold.')
        if not values['evaluate_cv']:
            st.caption('Tanpa CV, model disimpan tanpa skor validation baru.')
        st.subheader('Parameter XGBoost')
        columns = st.columns(3)
        xgb = {}
        # Bounds are checked by validate_training_config so an out-of-range entry stays visible as an error
        # instead of the widget silently keeping the previous value.
        specs = [('n_estimators', 'Putaran boosting', 1, 1000, 300, 1, None),
                 ('learning_rate', 'Learning rate', .001, 1, .05, .01, '%.3f'), ('max_depth', 'Max depth', 1, 12, 6, 1, None),
                 ('min_child_weight', 'Min child weight', 0, 100, 1., .1, '%.2f'),
                 ('subsample', 'Subsample', .1, 1, .8, .05, '%.2f'),
                 ('colsample_bytree', 'Colsample (bytree)', .1, 1, .8, .05, '%.2f')]
        for i, (name, label, low, high, default, step, number_format) in enumerate(specs):
            with columns[i % 3]:
                xgb[name] = st.number_input(label, value=default, step=step, format=number_format,
                                            help=_range_help(low, high), key='training_xgb_' + name)
        with st.expander('Parameter lanjutan'):
            advanced = st.columns(2)
            for i, (name, label, high, default) in enumerate([
                ('reg_lambda', 'L2 XGBoost (lambda)', 10000., 1.),
                ('reg_alpha', 'L1 XGBoost (alpha)', 10000., 0.), ('gamma', 'Gamma', 1000., 0.)]):
                with advanced[i % 2]:
                    xgb[name] = st.number_input(label, value=default, step=.1, format='%.2f',
                                                help=_range_help(0, high), key='training_xgb_' + name)
            with advanced[1]:
                xgb['n_jobs'] = st.number_input('Thread CPU', 1, max(1, os.cpu_count() or 1),
                                               min(4, os.cpu_count() or 1), key='training_xgb_threads')
        values['xgboost'] = xgb
        st.divider()
        validation_error = None
        config = None
        try:
            if candidate_blank:
                raise ValueError('Isi jumlah gen kandidat sebelum memulai training.')
            config = validate_training_config(values, candidate_count=count)
            if dataset and values['evaluate_cv']:
                if data is not None:
                    make_cv_assignments(data, values['cv_folds'])
                else:
                    # Splitting requires only sample/label/donor axes, never fitted gene values.
                    from gtex_explorer.training.data import TrainingData
                    preview = TrainingData(Path('.'), {'cached_folds': []}, audit.sample_ids,
                        audit.donor_ids, audit.classes, audit.labels, audit.gene_names, audit.raw_tpm)
                    make_cv_assignments(preview, values['cv_folds'])
        except (ValueError, OSError, TypeError) as error:
            validation_error = str(error)
            st.error(user_error(error))
        st.divider()
        back, forward = st.columns(2)
        back.button('Kembali ke data', key='training_back_to_data', icon=':material/arrow_back:',
                    on_click=_set_training_step, args=(TRAINING_STEPS[0],), width='stretch')
        forward.button('Tinjau dan latih', key='training_to_review', type='primary',
                       icon=':material/arrow_forward:', on_click=_set_training_step,
                       args=(TRAINING_STEPS[2],), disabled=dataset is None or validation_error is not None,
                       width='stretch')
    with review_tab:
        st.subheader('Tinjau data dan konfigurasi')
        if dataset:
            _dataset_strip(dataset)
            st.success('Data telah diperiksa.')
        else:
            st.info('Unggah dan periksa data pada langkah 1 sebelum memulai training.')
        st.divider()
        selection, model = st.columns(2, gap='large')
        with selection:
            st.subheader('Seleksi fitur')
            rows = [('Metode', METHODS[method]), ('Gen kandidat', f'{count:,}'.replace(',', '.')),
                    ('Top-k gen', f'{k or 0:,}'.replace(',', '.'))]
            if method == 'l1':
                rows.append(('C', str(values.get('l1_c'))))
            elif method == 'mutual_information':
                rows.append(('n_neighbors', str(values['mi_neighbors'])))
            st.table(pd.DataFrame(rows, columns=['Pengaturan', 'Nilai']).set_index('Pengaturan'))
        with model:
            st.subheader('Model dan evaluasi')
            rows = [('Classifier', 'XGBoost'), ('Putaran boosting', str(xgb['n_estimators'])),
                    ('Evaluasi', 'Cross-validation' if values['evaluate_cv'] else 'Tanpa CV')]
            if values['evaluate_cv']:
                rows += [('Jumlah fold', str(values['cv_folds'])), ('Pembagian', 'Donor terpisah')]
            st.table(pd.DataFrame(rows, columns=['Pengaturan', 'Nilai']).set_index('Pengaturan'))
        with st.expander('Parameter XGBoost dan opsi selector'):
            st.dataframe(pd.DataFrame([{'Parameter': name, 'Nilai': str(value)}
                for name, value in values.items() if name != 'xgboost'] +
                [{'Parameter': name, 'Nilai': str(value)} for name, value in xgb.items()]), hide_index=True)
        if validation_error:
            st.error(user_error(validation_error))
        elif dataset:
            st.success('Konfigurasi siap dilatih. Model dan panel gen akan disimpan.')
        if not values['evaluate_cv']:
            st.caption('Tanpa CV, model disimpan tanpa skor validation baru.')
        st.divider()
        back, action = st.columns(2)
        back.button('Kembali ke konfigurasi', key='training_review_back', icon=':material/arrow_back:',
                    on_click=_set_training_step, args=(TRAINING_STEPS[1],), width='stretch')
        with action:
            if st.button('Latih dan simpan model', type='primary', key='training_start',
                         width='stretch', disabled=dataset is None or validation_error is not None):
                try:
                    with st.spinner('Menyimpan data dan memulai training...'):
                        actual_root = data_root if data is not None else save_training_upload(audit, state_root / 'datasets', count)
                        record = start_job(config, actual_root, state_root)
                    st.session_state.training_selected_run = record.run_id
                    st.session_state.training_run = record.run_id
                    st.session_state.training_results_view = 'Status proses'
                    st.session_state.page = 'Hasil Pelatihan'
                    st.rerun()
                except (ValueError, OSError, KeyError) as error:
                    st.error(user_error(error))
            st.caption('Model dan panel gen disimpan sebagai hasil pelatihan baru.')


def _follow_active_run(state_root, polling):
    """Redraw the page when a run starts or ends, so polling is on only while one is active."""
    if has_active_run(state_root) != polling:
        st.rerun(scope='app')


def _run_details(data_root: Path, state_root: Path, polling: bool = False):
    _follow_active_run(state_root, polling)
    gallery_only = st.session_state.get('training_results_view') == 'Gallery model'
    run_id = choose_run(state_root, 'training_run', gallery_only=gallery_only)
    if run_id is None:
        return
    try:
        root = state_root / 'runs' / run_id
        saved = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        record = read_job(run_id, state_root)
        config = saved['config']
        dataset = saved.get('dataset')
        st.subheader(method_description(config))
        if dataset:
            source = dataset['source']
            st.caption(f"{source.get('expression_file', 'Data notebook')} - {dataset['sample_count']} sampel - "
                       f"{dataset['donor_count']} donor - {dataset['class_count']} jaringan")
        st.subheader('Status proses')
        labels = {'queued': 'Menunggu worker', 'preparing_fold': 'Persiapan fold',
                  'fitting_selector': 'Seleksi fitur', 'fitting_xgboost': 'Pelatihan XGBoost',
                  'final_selector': 'Seleksi fitur pada seluruh data development',
                  'final_xgboost': 'Pelatihan model final', 'completed': 'Selesai',
                  'saving_results': 'Menyimpan model dan hasil',
                  'verifying_results': 'Memverifikasi model, metrik, dan panel gen',
                  'cancelled': 'Dihentikan', 'paused': 'Dijeda',
                  'cancelling': 'Menghentikan pelatihan', 'failed': 'Gagal'}
        # A resumed run keeps the stage it stopped at until the worker reports a new one.
        display_stage = record.status if record.status in {'paused', 'cancelling', 'queued', 'cancelled'} else record.details.get('stage')
        elapsed = record.details.get('elapsed_seconds', 0)
        if record.status in {'running', 'queued'}:
            elapsed += max(0, time.time() - record.details['updated'])
        stage_column, time_column, rounds_column = st.columns([1, 1, 1.5])
        stage_column.markdown('**' + labels.get(display_stage, record.status) + '**')
        if record.details.get('fold'):
            stage_column.caption(f"Fold {record.details['fold']} dari {config['cv_folds']}")
        time_column.caption(f"Waktu proses: {elapsed:.0f} detik")
        rounds_column.caption(f"Target XGBoost: {config['xgboost']['n_estimators']} putaran per fit.")
        completed_rounds = record.details.get('rounds_completed')
        total_rounds = record.details.get('rounds_total')
        if completed_rounds is not None and total_rounds:
            prefix = 'Putaran terakhir tercatat' if record.status == 'cancelled' else 'Putaran XGBoost pada fit ini'
            st.progress(int(round(min(1.0, max(0.0, completed_rounds / total_rounds)) * 100)),
                        text=f'{prefix}: {completed_rounds} dari {total_rounds}')
        if record.status == 'cancelled':
            st.info('Pelatihan dihentikan. Fold yang sudah selesai tetap tersimpan. '
                    'Jika dilanjutkan, fit yang terhenti akan diulang.')
        if record.status == 'paused':
            st.info('Proses berhenti sementara dan tetap tersimpan di memori. Resume meneruskan fit yang sama.')
            st.caption('Pause aktif selama worker masih hidup. Menutup atau memulai ulang komputer mengakhiri proses.')
        if record.status in {'running', 'queued', 'cancelling', 'paused'}:
            cancelling = (root / 'cancel.json').exists()
            finalizing = record.status == 'running' and record.details.get('stage') in FINALIZING_STAGES
            if finalizing:
                st.info('Hasil sedang disimpan dan diverifikasi. Model akan masuk gallery setelah proses ini selesai.')
            if cancelling:
                st.info('Permintaan berhenti diterima. Menunggu worker menghentikan pelatihan.')
            pause_column, stop_column = st.columns(2)
            with pause_column:
                if record.status == 'paused':
                    if st.button('Resume', key='training_unpause', type='primary', width='stretch', icon=':material/play_arrow:'):
                        try:
                            unpause_job(run_id, state_root)
                            st.rerun()
                        except (ValueError, OSError, psutil.Error) as error:
                            st.error(user_error(error))
                elif st.button('Pause', key='training_pause', type='primary', width='stretch', icon=':material/pause:',
                               disabled=finalizing or record.status != 'running' or not (root / 'worker.json').exists()):
                    try:
                        pause_job(run_id, state_root)
                        st.rerun()
                    except (ValueError, OSError, psutil.Error) as error:
                        st.error(user_error(error))
            with stop_column:
                if st.button('Stop', key='training_cancel', disabled=cancelling or finalizing, width='stretch',
                             icon=':material/stop:', help='Akhiri worker. Fold yang selesai tetap tersimpan, fit yang terputus dibuang.'):
                    try:
                        request_cancel(run_id, state_root)
                        st.rerun()
                    except (ValueError, OSError, psutil.Error) as error:
                        st.error(user_error(error))
        if record.error:
            st.error(user_error(record.error['message']))
            if record.error.get('diagnostics'):
                show_diagnostics(record.error['diagnostics'])
        if record.status == 'failed' and not can_resume(record):
            st.caption('Kegagalan ini berasal dari data atau konfigurasi. Ubah pengaturan lalu latih run baru.')
        if can_resume(record):
            if st.button('Continue from checkpoint', key='training_resume', icon=':material/restart_alt:'):
                try:
                    resume_job(run_id, data_root, state_root)
                    st.rerun()
                except (ValueError, OSError) as error:
                    st.error(user_error(error))
        _show_saved_parameters(saved)
        summary = record.details.get('cv_summary')
        st.divider()
        if summary:
            _show_cv_results(summary, config)
        elif record.status == 'completed':
            st.info('Training selesai tanpa CV. Tidak ada skor validation atau holdout baru.')
        ready = record.status == 'completed' and any(entry['run_id'] == run_id for entry in list_gallery(state_root))
        if record.status == 'completed' and not ready:
            st.warning('Hasil belum lolos verifikasi gallery. Periksa berkas model, metrik, dan panel gen yang tersimpan.')
        if not ready and record.status not in ACTIVE_STATUSES:
            _delete_button(run_id, 'Hapus hasil pelatihan')
        if ready:
            st.divider()
            st.subheader('Model final tersimpan')
            st.markdown('**' + method_description(config) + ' - XGBoost**')
            panel_note = 'Panel final dan panel tiap fold tersedia pada Panel Gen.' if config['evaluate_cv'] else 'Panel final tersedia pada Panel Gen.'
            st.caption('Dilatih pada seluruh dataset training ini. ' + panel_note)
            _show_model_actions(run_id)
    except (ValueError, OSError, KeyError, TypeError) as error:
        st.error(f'Hasil pelatihan tidak dapat dibaca: {user_error(error)}')


def _delete_button(run_id, label):
    if st.button(label, key='training_delete_run', icon=':material/delete:', wrap=True):
        st.session_state.training_delete_target = run_id
        # Dialogs open from a full page run, not from inside these panels.
        st.rerun(scope='app')


def _dismiss_delete():
    st.session_state.pop('training_delete_target', None)


@st.dialog('Hapus hasil pelatihan', on_dismiss=_dismiss_delete)
def _delete_run_dialog(state_root, run_id):
    st.write(run_label(Path(state_root) / 'runs' / run_id))
    st.warning('Hasil pelatihan, model, dan panel gennya akan dihapus permanen. Tindakan ini tidak dapat dibatalkan. '
               'Dataset unggahan tetap tersimpan sampai workspace dihapus.')
    confirm, cancel = st.columns(2)
    if confirm.button('Hapus permanen', key='training_delete_confirm', type='primary',
                      icon=':material/delete:', width='stretch'):
        try:
            delete_run(run_id, state_root)
            _dismiss_delete()
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(user_error(error))
    if cancel.button('Batal', key='training_delete_cancel', width='stretch'):
        _dismiss_delete()
        st.rerun()


def _show_saved_parameters(saved):
    with st.expander('Data dan parameter tersimpan'):
        dataset = saved.get('dataset')
        if dataset:
            st.write('Metadata: ' + dataset.get('source', {}).get('metadata_file', 'Metadata notebook'))
        if saved.get('data_fingerprint'):
            st.caption('Identitas dataset: ' + saved['data_fingerprint'][:16])
        st.dataframe(pd.DataFrame([{'Parameter': key, 'Nilai': str(value)}
                                  for key, value in saved['config']['xgboost'].items()]), hide_index=True)


def _show_model_actions(run_id):
    with st.container(horizontal=True, gap='small'):
        if st.button('Gunakan untuk Prediksi Jaringan', key='training_open_prediction', type='primary',
                     icon=':material/biotech:', wrap=True):
            # The prediction page loads this model and invalidates previous model results.
            st.session_state.prediction_model = run_id
            for key in ['input_audit', 'predictions', 'probabilities', 'prediction_rounds']:
                st.session_state.pop(key, None)
            st.session_state.page = 'Prediksi Jaringan'
            st.rerun()
        if st.button('Buka Panel Gen', key='training_open_genes', icon=':material/table_view:', wrap=True):
            st.session_state.training_panel_run = run_id
            st.session_state.page = 'Panel Gen Pelatihan'
            st.rerun()
        _delete_button(run_id, 'Hapus model')


def _show_cv_results(summary, config):
    st.subheader('Evaluasi CV')
    st.caption(f"Rerata dan simpangan baku dari {summary.get('fold_count', config['cv_folds'])} fold validation.")
    with st.container(key='training_cv_results'):
        metrics_column, stability_column = st.columns([3, 1], gap='medium')
        metrics_column.dataframe(pd.DataFrame({
            'Metrik': list(METRICS.values()),
            'Rerata': [f"{summary['mean_' + key]:.2%}".replace('.', ',') for key in METRICS],
            'SD (poin persentase)': [f"{summary['std_' + key] * 100:.2f}".replace('.', ',') for key in METRICS]}),
            hide_index=True, width='stretch')
        if summary['mean_jaccard'] is not None:
            stability_column.metric('Jaccard antar-fold', f"{summary['mean_jaccard']:.2%}".replace('.', ','))
            stability_column.caption('Kesamaan gen terpilih')
    folds = pd.DataFrame(summary['fold_results']).drop(columns='selected_genes', errors='ignore')
    with st.container(horizontal=True, vertical_alignment='top'):
        with st.expander('Hasil setiap fold'):
            st.dataframe(folds.rename(columns=METRICS).style.format(
                {label: '{:.2%}' for label in METRICS.values()}), hide_index=True)
        download_button('Unduh hasil fold', folds.to_csv(index=False), 'cv_folds.csv',
                           key='training_download_folds', icon=':material/download:')


def _select_gallery_run(run_id):
    st.session_state.training_gallery_run = run_id
    st.session_state.training_selected_run = run_id


def _gallery_results(state_root, polling: bool = False):
    _follow_active_run(state_root, polling)
    issues = []
    entries = list_gallery(state_root, issues=issues)
    for issue in issues:
        st.warning(f"Hasil {issue['run_id'][-6:]} belum lolos verifikasi: {user_error(issue['reason'])}")
    if not entries:
        st.info('Belum ada model terverifikasi di gallery. Periksa Status proses.')
        return
    try:
        records = []
        for entry in entries:
            path = Path(state_root) / 'runs' / entry['run_id'] / 'config.json'
            saved = json.loads(path.read_text(encoding='utf-8'))
            records.append({**entry, 'saved': saved,
                'method': method_description(saved['config']),
                'filename': saved.get('dataset', {}).get('source', {}).get('expression_file', 'Data eksperimen notebook'),
                'date': time.strftime('%d/%m/%Y %H:%M', time.localtime(run_created(path.parent)))})
        library, details = st.columns([1, 2.6], gap='large')
        with library, st.container(key='gallery_models'):
            st.subheader('Model tersimpan')
            query = st.text_input('Cari model atau dataset', placeholder='Cari model atau dataset',
                                  key='training_gallery_search', label_visibility='collapsed').strip().casefold()
            visible = [record for record in records if query in (record['method'] + ' ' + record['filename']).casefold()]
            if not visible:
                st.info('Tidak ada model yang cocok dengan pencarian.')
                return
            selected = st.session_state.get('training_gallery_run', st.session_state.get('training_selected_run'))
            if selected not in {record['run_id'] for record in visible}:
                selected = visible[0]['run_id']
                _select_gallery_run(selected)
            for record in visible:
                label = f"**{record['method']}**  \n{record['filename']}  \n{record['date']}"
                st.button(label, key='gallery_select_' + record['run_id'],
                          type='primary' if record['run_id'] == selected else 'tertiary',
                          width='stretch', wrap=True, on_click=_select_gallery_run, args=(record['run_id'],))
        record = next(record for record in visible if record['run_id'] == selected)
        saved = record['saved']
        config = saved['config']
        with details, st.container(key='gallery_detail'):
            with st.container(horizontal=True, vertical_alignment='center', gap='small'):
                st.subheader(record['method'] + ', XGBoost', width='content')
                st.badge('Tersimpan dan terverifikasi', icon=':material/check:', color='green')
            evaluation = f"{config['cv_folds']} fold" if config['evaluate_cv'] else 'Tanpa CV'
            context = [f"{config['xgboost']['n_estimators']} putaran", evaluation]
            dataset = saved.get('dataset', {})
            for key, label in [('sample_count', 'sampel'), ('donor_count', 'donor'), ('class_count', 'jaringan')]:
                if key in dataset:
                    context.append(f"{dataset[key]:,} {label}".replace(',', '.'))
            st.write(' - '.join(context))
            st.caption('Dataset: ' + record['filename'])
            _show_model_actions(selected)
            st.divider()
            if record['cv_summary']:
                _show_cv_results(record['cv_summary'], config)
            else:
                st.caption('Tanpa CV. Tidak ada skor validation atau holdout baru.')
            _show_saved_parameters(saved)
    except (OSError, ValueError, KeyError, TypeError) as error:
        st.error(f'Hasil pelatihan tidak dapat dibaca: {user_error(error)}')


def render_training_results_page(data_root: Path, state_root: Path):
    st.markdown('''<style>
    .st-key-training_results h3 {font-size:1.3rem !important;}
    .st-key-training_results h1 {font-size:2.4rem !important;}
    .st-key-training_results hr {margin:.6rem 0 !important;}
    .st-key-training_results [data-testid="stVerticalBlock"] {gap:.6rem;}
    .st-key-training_results [data-baseweb="tab-list"] {gap:2rem;}
    .st-key-training_results [data-baseweb="tab-list"] button p {font-size:1rem; font-weight:600;}
    .st-key-gallery_models {border-right:1px solid #dce3e7; padding-right:1.1rem;
        min-height:calc(100dvh - 14rem); padding-top:1rem;}
    .st-key-gallery_detail {padding-top:1rem;}
    .st-key-gallery_models button {text-align:left; justify-content:flex-start; border:0;
        border-left:3px solid transparent; border-radius:5px; padding:.7rem .8rem;}
    .st-key-gallery_models button[kind="primary"] {background:#edf8f8 !important; color:#14213d !important;
        border:0 !important; border-left:3px solid #078c92 !important;}
    .st-key-gallery_models button[kind="primary"] * {color:#14213d !important;}
    .st-key-gallery_models button > div {width:100%; justify-content:flex-start;}
    .st-key-gallery_models button [data-testid="stMarkdownContainer"] {text-align:left;}
    .st-key-gallery_models button p {white-space:normal; overflow-wrap:anywhere; font-size:.9rem; line-height:1.6;}
    .st-key-gallery_models button strong {font-size:1.05rem;}
    .st-key-gallery_detail [data-testid="stMetric"] {padding:.7rem;}
    .st-key-gallery_detail [data-testid="stMetricValue"] {font-size:2rem;}
    @media(max-width:1000px) {
        .st-key-training_cv_results [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
        .st-key-training_cv_results [data-testid="stColumn"] {width:100%; flex:1 1 100%;}
        .st-key-training_results [data-testid="stHorizontalBlock"]:has(> div .st-key-gallery_models) {flex-wrap:wrap;}
        .st-key-training_results [data-testid="stHorizontalBlock"]:has(> div .st-key-gallery_models) > [data-testid="stColumn"] {width:100%; flex:1 1 100%;}
        .st-key-gallery_models {border-right:0; border-bottom:1px solid #dce3e7; padding-right:0; padding-bottom:1rem; min-height:0;}
    }
    </style>''', unsafe_allow_html=True)
    with st.container(key='training_results'):
        st.title('Hasil pelatihan')
        st.caption('Hasil model yang dilatih pengguna, terpisah dari eksperimen notebook.')
        if 'training_results_view' not in st.session_state:
            st.session_state.training_results_view = 'Gallery model' if list_gallery(state_root) else 'Status proses'
        gallery, process = st.tabs(['Gallery model', 'Status proses'], key='training_results_view', on_change='rerun')
        # Saved results do not change, so these panels poll only while a run is active.
        polling = has_active_run(state_root)
        run_every = 2 if polling else None
        if gallery.open:
            with gallery:
                st.fragment(_gallery_results, run_every=run_every)(state_root, polling)
        if process.open:
            with process:
                st.fragment(_run_details, run_every=run_every)(data_root, state_root, polling)
        target = st.session_state.get('training_delete_target')
        if target:
            if (Path(state_root) / 'runs' / target).is_dir():
                _delete_run_dialog(state_root, target)
            else:
                _dismiss_delete()

