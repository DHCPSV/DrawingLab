"""No-model-call API smoke test against a running localhost DrawingLab server.

Checks invalid image input -> healthy session -> save/export the public original
example. It deliberately never requests a model inference.
"""
import argparse
import base64
import json
from pathlib import Path
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    base = f'http://127.0.0.1:{args.port}'

    def call(path, value=None):
        request = urllib.request.Request(base + path,
            data=json.dumps(value).encode() if value is not None else None,
            headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as response:
            return response.code, json.load(response)

    code, status = call('/api/status')
    assert code == 200 and status['release'] == '0.1.0'
    code, session = call('/api/sessions', {})
    assert code == 200
    code, _ = call(f"/api/sessions/{session['id']}/suggest", {
        'revision': 1, 'prompt': 'Original cat sketch', 'ink': 'not a PNG'})
    assert code == 422
    code, result = call(f"/api/sessions/{session['id']}/result")
    assert code == 200 and result['status'] == 'idle'
    document = json.loads((ROOT/'examples/cat-sketch.json').read_text(encoding='utf-8'))
    code, saved = call('/api/projects', document)
    assert code == 200
    code, reopened = call('/api/projects/' + saved['name'])
    assert code == 200 and reopened == document
    png = (ROOT/'docs/images/cat-input.png').read_bytes()
    code, exported = call('/api/exports', {'ink': 'data:image/png;base64,' + base64.b64encode(png).decode()})
    assert code == 200 and exported['name'].endswith('.png')
    print(json.dumps({'passed': 6, 'checks': ['status', 'session', 'bad input rejected',
        'session survives failure', 'project save/reopen', 'PNG export'], 'model_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
