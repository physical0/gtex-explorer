"""Private direct-to-Storage uploads avoid Function request-body limits."""
from gtex_explorer.paths import APP_ROOT
import hashlib
import math
import re
import time
import uuid

import streamlit as st

from gtex_explorer.cloud.state import enabled, state, part_key, PART_SIZE


MAX_UPLOAD = 200 * 1024 * 1024
MAX_PARTS = math.ceil(MAX_UPLOAD / PART_SIZE)
# Signed upload URLs expire after two hours; reissue well before that.
TARGET_LIFETIME = 90 * 60
_UPLOAD = None


def upload_key(upload_id):
    return 'incoming/' + upload_id + '/upload'


def validate_reference(value, issued=None):
    if (not isinstance(value, dict) or not isinstance(value.get('id'), str)
            or not re.fullmatch(r'[0-9a-f]{32}', value['id'])
            or (issued is not None and value['id'] not in issued)
            or not isinstance(value.get('name'), str)
            or not re.fullmatch(r'[^/\\]+\.(csv|tsv|txt|gct)', value['name'], re.I)
            or type(value.get('size')) is not int or not 0 <= value['size'] <= MAX_UPLOAD):
        raise ValueError('Referensi berkas unggahan tidak valid. Pilih berkas kembali.')
    return value


def _issue_target():
    upload_id = uuid.uuid4().hex
    store, key = state().store, upload_key(upload_id)
    st.session_state.setdefault('gtex-upload-issued', set()).add(upload_id)
    return {'id': upload_id, 'created': time.time(), 'url': store.signed_upload_url(key),
            'parts': [store.signed_upload_url(part_key(key, i)) for i in range(MAX_PARTS)]}


@st.cache_data(max_entries=4, show_spinner='Membaca berkas dari penyimpanan privat...')
def _download(upload_id, size):
    found = state().store.get(upload_key(upload_id))
    if found is None or len(found[0]) != size:
        raise ValueError('Berkas unggahan tidak lengkap atau tidak tersedia. Unggah kembali.')
    return found[0]


class CloudUpload:
    def __init__(self, value):
        self.reference = validate_reference(value, st.session_state.get('gtex-upload-issued', set()))
        self.name = value['name']
        self.size = value['size']

    def getvalue(self):
        return _download(self.reference['id'], self.size)


def file_uploader(label, *, type, key, on_change=None, disabled=False, label_visibility='visible'):
    if not enabled():
        return st.file_uploader(label, type=type, key=key, on_change=on_change,
                                disabled=disabled, label_visibility=label_visibility)
    global _UPLOAD
    if _UPLOAD is None:
        _UPLOAD = st.components.v2.component('gtex_private_upload',
            html='''<div class="upload"><label><span>↑ Upload</span><input type="file" /></label>
              <div role="status"></div><button type="button" aria-label="Hapus berkas">×</button></div>
              <progress max="100" value="0" hidden></progress>''',
            css='''
              .upload {display:flex;align-items:center;gap:12px;padding:14px;background:var(--st-secondary-background-color,#f7f8f8);border-radius:6px;min-height:42px;font-family:var(--st-font,sans-serif);color:var(--st-text-color,#14213d);font-size:14px}
              label {position:relative;flex:none;padding:8px 12px;border:1px solid #d4dce3;border-radius:5px;background:var(--st-background-color,#fff);cursor:pointer}
              input {position:absolute;inset:0;opacity:0;width:100%;cursor:pointer}
              [role="status"] {overflow-wrap:anywhere;flex:1;color:var(--st-text-color,#14213d)}
              button {border:0;background:transparent;font-size:22px;cursor:pointer;color:inherit}
              progress {width:100%;height:4px;accent-color:var(--st-primary-color,#078c92)}
            ''',
            js='\n' + (APP_ROOT / 'assets/cloud_upload.js').read_text(encoding='utf-8'))
    if label_visibility == 'visible':
        st.markdown(label)
    # CCv2 reserves '__' internally; retain workspace identity through a hash.
    component_key = 'gtex-upload-' + hashlib.sha256(key.encode()).hexdigest()[:32]
    saved = st.session_state.get(component_key)
    current = saved.get('value') if isinstance(saved, dict) else getattr(saved, 'value', None)
    target = st.session_state.get(component_key + '-target')
    # Each target path is writable once, so a used or ageing target is replaced.
    if (target is None or time.time() - target['created'] > TARGET_LIFETIME
            or (isinstance(current, dict) and current.get('id') == target['id'])):
        target = st.session_state[component_key + '-target'] = _issue_target()
    result = _UPLOAD(key=component_key, data={'accept': ','.join('.' + extension for extension in type),
        'current': current, 'disabled': disabled, 'max_size': MAX_UPLOAD, 'part_size': PART_SIZE,
        'target': {field: target[field] for field in ('id', 'url', 'parts')}},
        default={'value': None}, on_value_change=on_change or (lambda: None), height='content')
    value = getattr(result, 'value', None)
    return CloudUpload(value) if value is not None else None
