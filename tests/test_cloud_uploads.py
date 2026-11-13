import pytest

from gtex_explorer.platform.cloud.uploads import validate_reference


from gtex_explorer.platform.cloud.uploads import sign

ISSUED = '0123456789abcdef0123456789abcdef'


@pytest.mark.parametrize('upload_id', ['../state', 'gtex-production/state.json', ISSUED.upper()])
def test_upload_reference_cannot_read_other_private_objects(upload_id):
    with pytest.raises(ValueError):
        validate_reference({'id': upload_id, 'sig': sign(upload_id), 'name': 'test.csv', 'size': 100})


@pytest.mark.parametrize('sig', [None, '', sign('fedcba9876543210fedcba9876543210')])
def test_upload_reference_requires_server_signature_for_its_id(sig):
    with pytest.raises(ValueError):
        validate_reference({'id': ISSUED, 'sig': sig, 'name': 'test.csv', 'size': 100})


@pytest.mark.parametrize('name', ['file.exe', '../tpm.csv', 'a/b.csv'])
def test_upload_reference_rejects_unsupported_names(name):
    with pytest.raises(ValueError):
        validate_reference({'id': ISSUED, 'sig': sign(ISSUED), 'name': name, 'size': 100})


def test_upload_reference_rejects_oversized_files():
    from gtex_explorer.platform.cloud.uploads import MAX_UPLOAD
    with pytest.raises(ValueError):
        validate_reference({'id': ISSUED, 'sig': sign(ISSUED), 'name': 'tpm.csv', 'size': MAX_UPLOAD + 1})


def test_signed_reference_is_accepted_in_any_session():
    value = {'id': ISSUED, 'sig': sign(ISSUED), 'name': 'Ekspresi gen.csv', 'size': 512}
    assert validate_reference(value) == value


def test_hosted_form_handles_namespaced_upload_keys_without_component_error(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import gtex_explorer.platform.cloud.state as cloud_state
    from test_cloud_state import MemoryStore
    monkeypatch.setenv('GTEX_CLOUD', '1')
    monkeypatch.setenv('GTEX_CLOUD_ROLE', 'web')
    monkeypatch.setenv('GTEX_CLOUD_PREFIX', 'test-hosted-form')
    monkeypatch.setattr(cloud_state, '_instance', cloud_state.CloudState(tmp_path / 'state', MemoryStore()))
    script = Path(__file__).resolve().parents[1] / 'cloud_app.py'
    app = AppTest.from_file(str(script), default_timeout=60).run()
    app.button(key='workspace_open').click().run()
    assert not app.exception
    assert app.title[0].value == 'Latih model'
