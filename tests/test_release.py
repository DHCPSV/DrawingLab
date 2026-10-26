"""Release boot/configuration regressions; no model loading or downloads."""
import json

from fastapi.testclient import TestClient
from drawinglab.engines import create_engine
from drawinglab.server import create_app
from test_smoke import MissingEngine


def test_fresh_checkout_boots_without_klein_dependencies(tmp_path, monkeypatch):
    monkeypatch.delenv('DRAWINGLAB_ENGINE', raising=False)
    monkeypatch.delenv('DRAWINGLAB_MODEL_DIR', raising=False)
    engine = create_engine(tmp_path)
    assert engine.status()['engine'] == 'klein'
    assert engine.status()['ready'] is False
    with TestClient(create_app(engine, tmp_path)) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/status').json()['release'] == '0.1.0'


def test_windows_bom_backend_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv('DRAWINGLAB_ENGINE', raising=False)
    (tmp_path / 'models').mkdir()
    (tmp_path / 'models/active-engine.json').write_text(
        json.dumps({'engine': 'counterfeit'}), encoding='utf-8-sig')
    assert create_engine(tmp_path).__class__.__name__ == 'ModelEngine'


def test_reusing_models_does_not_relocate_application_data(tmp_path, monkeypatch):
    models = tmp_path / 'existing-assets'
    monkeypatch.setenv('DRAWINGLAB_ENGINE', 'klein')
    monkeypatch.setenv('DRAWINGLAB_MODEL_DIR', str(models))
    engine = create_engine(tmp_path / 'app')
    assert engine.model_dir == models.resolve()
    assert engine.root == (tmp_path / 'app').resolve()
    assert not models.exists()


def test_bad_content_length_returns_controlled_error(tmp_path):
    with TestClient(create_app(MissingEngine(), tmp_path)) as client:
        assert client.post('/api/sessions', headers={'content-length': 'invalid'}).status_code == 400
        assert client.post('/api/sessions').status_code == 200
