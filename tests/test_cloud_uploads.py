import pytest

from gtex_explorer.cloud.uploads import validate_reference


@pytest.mark.parametrize('path', ['gtex-production/state.json', '../state.json',
    'incoming/a/test.csv', 'https://example.com/a.csv', 'incoming/' + 'a' * 36 + '/file.exe'])
def test_upload_reference_cannot_read_other_private_objects(path):
    with pytest.raises(ValueError):
        validate_reference({'pathname': path, 'name': 'test.csv', 'size': 100})


def test_generated_private_upload_reference_is_accepted():
    value = {'pathname': 'incoming/01234567-89ab-cdef-0123-456789abcdef/tpm.csv',
             'name': 'Ekspresi gen.csv', 'size': 512}
    assert validate_reference(value) == value


def test_hosted_form_handles_namespaced_upload_keys_without_component_error(tmp_path, monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    import gtex_explorer.cloud.state as cloud_state
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
