"""Hosted entry point. Desktop app.py continues to use local state unchanged."""
import os
from pathlib import Path

import streamlit as st

from gtex_explorer.cloud.state import state, CloudConflict, CloudIntegrityError

os.environ['GTEX_CLOUD'] = '1'
os.environ['GTEX_CLOUD_ROLE'] = 'web'
os.environ.setdefault('GTEX_DESKTOP_STATE_DIR', '/tmp/gtex-cloud-state')
os.environ.setdefault('GTEX_CLOUD_PREFIX', 'gtex-' + os.environ.get('VERCEL_ENV', 'development'))

try:
    with state().script():
        filename = Path(__file__).with_name('app.py')
        exec(compile(filename.read_text(encoding='utf-8'), str(filename), 'exec'),
             {'__name__': '__main__', '__file__': str(filename)})
except (CloudConflict, CloudIntegrityError, OSError) as error:
    st.error(str(error))
    if st.button('Muat ulang', icon=':material/refresh:'):
        st.rerun()
