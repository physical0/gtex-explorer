"""Media and exports must not depend on a particular HTTP instance's memory."""
import base64
import hashlib
import mimetypes
import os
from pathlib import Path

import streamlit as st

from gtex_explorer.cloud.state import enabled, state, CloudConflict


def image(source, **kwargs):
    if enabled() and isinstance(source, (str, Path)) and Path(source).is_file():
        path = Path(source)
        media_type = mimetypes.guess_type(path.name)[0] or 'image/png'
        source = 'data:' + media_type + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
    return st.image(source, **kwargs)


# Exports are deleted after a day, so re-store well within that window.
@st.cache_data(ttl=12 * 60 * 60, max_entries=32, show_spinner=False)
def _store_export(content, prefix):
    name = 'exports/' + prefix + '/' + hashlib.sha256(content).hexdigest()
    store = state().store
    try:
        store.put(name, content, create=True)
    except CloudConflict:
        found = store.get(name)
        if found is None or hashlib.sha256(found[0]).digest() != hashlib.sha256(content).digest():
            raise ValueError('Berkas ekspor gagal diverifikasi. Coba kembali.')
    return name


# Links stay valid for an hour; re-signing every 30 minutes avoids a call per rerun.
@st.cache_data(ttl=30 * 60, max_entries=64, show_spinner=False)
def _signed_url(name, file_name):
    return state().store.signed_download_url(name, file_name, expires=60 * 60)


def download_button(label, data, file_name=None, mime=None, *, key=None, **kwargs):
    if not enabled():
        return st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, **kwargs)
    content = data.encode('utf-8') if isinstance(data, str) else bytes(data)
    name = _store_export(content, os.environ.get('GTEX_CLOUD_PREFIX', 'gtex-production'))
    url = _signed_url(name, file_name or 'export.csv')
    # Browser download uses a private backing object, not Streamlit's in-memory
    # media registry, so it works even when the HTTP request reaches another instance.
    allowed = {k:v for k,v in kwargs.items() if k in {'type', 'help', 'disabled', 'icon', 'width'}}
    st.link_button(label, url, key=key, **allowed)
    return False
