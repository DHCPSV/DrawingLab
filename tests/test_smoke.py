"""Offline contract tests. All generated image fixtures and engine doubles live here.

These tests do not assess model quality and never load or download model weights.
Run from the repository with a project-local --basetemp and pytest cache directory.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
from queue import Queue
import threading
import time

import numpy as np
from PIL import Image, ImageDraw
import pytest
from fastapi.testclient import TestClient

from drawinglab.imaging import alignment, decode_ink, encode_png, extract_lines, update_mask
from drawinglab.jobs import JobManager
from drawinglab.server import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOP = ((90, 90), (550, 90))
BOTTOM = ((110, 600), (630, 600))


def test_pale_contours_survive_without_turning_tinted_paper_into_ink():
    paper = np.tile(np.linspace(225, 250, 768).astype(np.uint8), (768, 1))
    blank = Image.fromarray(paper, "L")
    assert np.count_nonzero(np.asarray(extract_lines(blank)) < 240) < 100
    drawing = blank.copy()
    ImageDraw.Draw(drawing).line([(250, 350), (550, 350)], fill=205, width=2)
    lines = np.asarray(extract_lines(drawing))
    assert np.count_nonzero(lines[349:352, 260:540] < 220) > 500


def ink_image(*paths, size=(768, 768)):
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    for path in paths:
        draw.line(path, fill=0, width=3)
    return image


def request(revision, ink, **changes):
    return dict(
        revision=revision,
        intent_revision=0,
        prompt="an original manga character",
        ink=ink,
        focus={"x": 384, "y": 384},
        eraser=False,
        force=False,
        seed=123,
        **changes,
    )


class MissingEngine:
    """An unavailable backend must never return a pretend suggestion."""

    def status(self):
        return {"ready": False, "message": "测试：模型权重未安装"}

    def generate(self, **kwargs):
        raise AssertionError("The API must not call an unavailable model")


@dataclass
class Response:
    image: Image.Image | None = None
    gate: threading.Event | None = None
    error: Exception | None = None


class ScriptedEngine:
    """A deterministic test-only worker, used solely to exercise scheduling."""

    def __init__(self, *responses):
        self.responses = deque(responses)
        self.started = Queue()
        self.calls = []

    def status(self):
        return {"ready": True, "message": "test-only engine"}

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.popleft()
        self.started.put(kwargs)
        if response.gate is not None and not response.gate.wait(5):
            raise RuntimeError("Test did not release its worker gate")
        if response.error is not None:
            raise response.error
        assert response.image is not None
        return response.image.copy(), {"test_fixture": True}


@pytest.fixture
def project_temp(tmp_path):
    assert tmp_path.resolve().is_relative_to(PROJECT_ROOT), (
        "Use --basetemp=.tmp/test-agent so test artifacts remain in this project"
    )
    return tmp_path


@pytest.fixture
def client(project_temp):
    with TestClient(create_app(engine=MissingEngine(), root=project_temp)) as value:
        yield value


@pytest.fixture
def manager_factory(project_temp):
    managers = []
    gates = []

    def factory(*responses):
        for response in responses:
            if response.gate is not None:
                gates.append(response.gate)
        engine = ScriptedEngine(*responses)
        manager = JobManager(engine, project_temp / f"worker-{len(managers)}")
        managers.append(manager)
        return manager, engine, manager.create()

    yield factory
    for gate in gates:
        gate.set()
    for manager in managers:
        manager.close()
        assert not manager.worker.is_alive(), "A test left a worker running"


def wait_finished(manager, identifier, status, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with manager.condition:
            session = manager.sessions[identifier]
            if session.status == status and session.pending is None and manager.active is None:
                return manager.result(identifier)
        time.sleep(0.01)
    pytest.fail(f"Worker did not reach {status}: {manager.result(identifier)!r}")


def test_missing_model_is_503_but_canvas_and_sessions_still_work(client):
    status = client.get("/api/status")
    assert status.status_code == 200
    assert status.json()["model"]["ready"] is False
    assert status.json()["canvas"] == {"width": 768, "height": 768, "pen_size": 3}
    assert client.get("/").status_code == 200
    session = client.post("/api/sessions").json()["id"]
    response = client.post(
        f"/api/sessions/{session}/suggest",
        json=request(1, encode_png(ink_image(TOP))),
    )
    assert response.status_code == 503
    assert "模型权重未安装" in response.json()["detail"]
    result = client.get(f"/api/sessions/{session}/result").json()
    assert result["ghost"] is None
    assert result["candidate_id"] is None
    assert result["status"] == "idle"


@pytest.mark.parametrize(
    "payload",
    [
        encode_png(ink_image()),
        encode_png(ink_image(TOP, size=(512, 512))),
        encode_png(Image.new("RGBA", (768, 768), (0, 0, 0, 0))),
        "data:image/png;base64,not-valid-base64!",
        "data:image/jpeg;base64,AA==",
    ],
    ids=["blank", "wrong-size", "transparent-blank", "corrupt-png", "non-png"],
)
def test_invalid_or_blank_canvas_is_rejected_before_model_call(client, payload):
    session = client.post("/api/sessions").json()["id"]
    response = client.post(f"/api/sessions/{session}/suggest", json=request(1, payload))
    assert response.status_code == 422
    assert response.json()["detail"]
    assert client.get(f"/api/sessions/{session}/result").json()["ghost"] is None


def test_save_load_and_export_are_confined_and_preserve_ink(client, project_temp):
    document = {
        "version": 1,
        "width": 768,
        "height": 768,
        "prompt": "原创人物，保留自己的线条",
        "strokes": [
            {"tool": "pen", "size": 3, "points": [{"x": 90, "y": 90}, {"x": 550, "y": 90}]},
            {"tool": "eraser", "size": 30, "points": [{"x": 300, "y": 90}]},
        ],
    }
    saved = client.post("/api/projects", json={"document": document})
    assert saved.status_code == 200
    saved_path = Path(saved.json()["path"]).resolve()
    assert saved_path.parent == (project_temp / "data" / "projects").resolve()
    assert json.loads(saved_path.read_text(encoding="utf-8")) == {"document": document}
    listing = client.get("/api/projects").json()["projects"]
    assert [item["name"] for item in listing] == [saved_path.name]
    loaded = client.get(f"/api/projects/{saved_path.name}")
    assert loaded.status_code == 200
    assert loaded.json() == {"document": document}
    assert not list(saved_path.parent.glob("*.tmp"))

    ink = ink_image(TOP)
    exported = client.post("/api/exports", json={"ink": encode_png(ink)})
    assert exported.status_code == 200
    export_path = Path(exported.json()["path"]).resolve()
    assert export_path.parent == (project_temp / "data" / "exports").resolve()
    with Image.open(export_path) as png:
        assert png.format == "PNG"
        assert np.array_equal(np.asarray(png.convert("L")), np.asarray(ink))


def test_transparent_canvas_composites_on_white_without_moving_lines():
    transparent = Image.new("RGBA", (768, 768), (0, 0, 0, 0))
    ImageDraw.Draw(transparent).line(TOP, fill=(0, 0, 0, 255), width=3)
    decoded = decode_ink(encode_png(transparent))
    assert np.array_equal(np.asarray(decoded), np.asarray(ink_image(TOP)))


def test_one_way_alignment_accepts_completion_and_rejects_misalignment():
    partial = ink_image(((90, 90), (230, 90)))
    complete = ink_image(TOP, BOTTOM, ((550, 90), (630, 600)))
    accepted = alignment(partial, complete)
    assert accepted["accepted"] is True
    assert accepted["matched"] == 1.0
    assert alignment(partial, ink_image(BOTTOM))["accepted"] is False
    assert alignment(partial, ink_image())["accepted"] is False
    assert alignment(partial, Image.new("L", (768, 768), 0))["accepted"] is False


def test_eraser_mask_contains_deleted_area_and_movement_contains_both_positions():
    old = ink_image(TOP, BOTTOM)
    erased = ink_image(TOP)
    mask = np.asarray(update_mask(old, erased, {"x": 90, "y": 90}))
    deleted = (np.asarray(old) < 200) & (np.asarray(erased) >= 200)
    assert mask[deleted].min() == 255
    assert mask[0, 0] == 0

    moved = ink_image(((50, 700), (250, 700)))
    mask = np.asarray(update_mask(ink_image(TOP), moved, {"x": 384, "y": 384}))
    changed = np.asarray(ink_image(TOP)) != np.asarray(moved)
    assert mask[changed].min() == 255
    assert mask.shape == (768, 768)


def test_queue_discards_intermediate_pending_requests(manager_factory):
    gate = threading.Event()
    first = ink_image(TOP)
    latest = ink_image(TOP, BOTTOM)
    manager, engine, session = manager_factory(Response(first, gate), Response(latest))
    manager.submit(session, request(20, first))
    engine.started.get(timeout=2)
    for revision in (21, 22):
        manager.submit(session, request(revision, ink_image(TOP, ((110, 500), (500, 500)))))
    manager.submit(session, request(23, latest))
    with manager.condition:
        assert manager.sessions[session].pending["revision"] == 23
        assert list(manager.queue) == [session]
    gate.set()
    result = wait_finished(manager, session, "ready")
    assert result["revision"] == 23
    assert len(engine.calls) == 2
    assert np.array_equal(np.asarray(engine.calls[1]["ink"]), np.asarray(latest))
    assert result["ghost"] is None, "Raster overlays must never be published by V2"
    assert result["paths_version"] == 1 and result["paths"]
    assert result["metrics"]["accepted"] is True


def test_compatible_newer_ink_accepts_inflight_candidate_then_uses_cache(manager_factory):
    gate = threading.Event()
    first = ink_image(((90, 90), (230, 90)))
    latest = ink_image(((90, 90), (420, 90)))
    manager, engine, session = manager_factory(Response(ink_image(TOP, BOTTOM), gate))
    manager.submit(session, request(1, first))
    engine.started.get(timeout=2)
    manager.submit(session, request(2, latest))
    gate.set()
    result = wait_finished(manager, session, "ready")
    assert result["revision"] == 2
    assert len(engine.calls) == 1, "Compatible additions should not discard and regenerate the candidate"
    assert result["metrics"]["cached"] is True
    assert result["metrics"]["accepted"] is True
    assert result["candidate_id"]
    evidence = manager.root / "data" / "sessions" / session
    assert evidence.resolve().is_relative_to(PROJECT_ROOT)
    with Image.open(evidence / "input.png") as saved:
        assert np.array_equal(np.asarray(saved), np.asarray(latest))


def test_invalidation_blocks_inflight_result_and_next_request_recovers(manager_factory):
    gate = threading.Event()
    manager, engine, session = manager_factory(Response(ink_image(TOP), gate), Response(ink_image(BOTTOM)))
    manager.submit(session, request(1, ink_image(TOP)))
    old_call = engine.started.get(timeout=2)
    manager.invalidate(session, revision=2, intent_revision=1, reset=True)
    assert old_call["cancel"]() is True
    gate.set()
    stale = wait_finished(manager, session, "idle")
    assert stale["revision"] == 2
    assert stale["ghost"] is None
    assert stale["candidate_id"] is None
    assert not (manager.root / "data" / "sessions" / session).exists()

    newer = request(3, ink_image(BOTTOM))
    newer["intent_revision"] = 1
    manager.submit(session, newer)
    result = wait_finished(manager, session, "ready")
    assert result["revision"] == 3
    assert result["intent_revision"] == 1
    assert result["ghost"] is None
    assert result["paths"]
    assert len(engine.calls) == 2


def test_worker_exception_is_visible_and_does_not_poison_next_job(manager_factory):
    manager, engine, session = manager_factory(
        Response(error=RuntimeError("test-only simulated inference failure")),
        Response(ink_image(TOP, BOTTOM)),
    )
    manager.submit(session, request(1, ink_image(TOP)))
    failed = wait_finished(manager, session, "error")
    assert "simulated inference failure" in failed["message"]
    assert failed["ghost"] is None
    assert failed["candidate_id"] is None
    manager.submit(session, request(2, ink_image(TOP)))
    result = wait_finished(manager, session, "ready")
    assert result["ghost"] is None
    assert result["paths"]
    assert len(engine.calls) == 2


def test_eraser_submission_alone_blocks_inflight_result_publication(manager_factory):
    old_gate = threading.Event()
    erased_gate = threading.Event()
    original = ink_image(TOP, BOTTOM)
    erased = ink_image(TOP)
    manager, engine, session = manager_factory(
        Response(original, old_gate), Response(erased, erased_gate)
    )
    manager.submit(session, request(1, original))
    old_call = engine.started.get(timeout=2)

    erase_request = request(2, erased)
    erase_request["eraser"] = True
    # API clients need not send a separate invalidate call to make deletion safe.
    manager.submit(session, erase_request)
    assert old_call["cancel"]() is True
    old_gate.set()
    new_call = engine.started.get(timeout=2)

    # The second call is blocked: any published artifact here could only have
    # come from the obsolete pre-erasure generation.
    during_repair = manager.result(session)
    assert during_repair["status"] == "running"
    assert during_repair["revision"] == 2
    assert during_repair["candidate_id"] is None
    assert during_repair["ghost"] is None
    assert new_call["previous"] is None
    evidence = manager.root / "data" / "sessions" / session
    assert not evidence.exists()

    erased_gate.set()
    repaired = wait_finished(manager, session, "ready")
    assert repaired["revision"] == 2
    assert len(engine.calls) == 2
    with Image.open(evidence / "candidate.png") as candidate:
        assert np.array_equal(np.asarray(candidate), np.asarray(erased))


def test_erasure_after_nonreset_invalidation_passes_previous_and_deleted_mask(manager_factory):
    repair_gate = threading.Event()
    original = ink_image(TOP, BOTTOM)
    erased = ink_image(TOP)
    manager, engine, session = manager_factory(
        Response(original), Response(erased, repair_gate)
    )
    manager.submit(session, request(1, original))
    engine.started.get(timeout=2)
    accepted = wait_finished(manager, session, "ready")
    assert accepted["candidate_id"]

    manager.invalidate(session, revision=2, intent_revision=0, reset=False)
    invalidated = manager.result(session)
    assert invalidated["ghost"] is None
    assert invalidated["candidate_id"] == accepted["candidate_id"]
    erase_request = request(2, erased)
    erase_request["eraser"] = True
    manager.submit(session, erase_request)
    repair_call = engine.started.get(timeout=2)

    assert repair_call["previous"] is not None
    assert np.array_equal(np.asarray(repair_call["previous"]), np.asarray(original))
    assert np.array_equal(np.asarray(repair_call["ink"]), np.asarray(erased))
    assert repair_call["mask"] is not None, "Erasure must repair a region rather than regenerate the whole canvas"
    mask = np.asarray(repair_call["mask"])
    deleted = (np.asarray(original) < 200) & (np.asarray(erased) >= 200)
    assert mask[deleted].min() == 255
    assert mask[90, 300] == 0, "The distant unchanged stroke must remain outside the repair mask"
    assert np.count_nonzero(mask) < mask.size

    repair_gate.set()
    repaired = wait_finished(manager, session, "ready")
    assert len(engine.calls) == 2
    assert repaired["revision"] == 2
    assert repaired["metrics"]["cached"] is False
    assert repaired["candidate_id"] != accepted["candidate_id"]


def stroke(identifier, *points):
    return {"id": identifier, "tool": "pen", "size": 3,
            "points": [{"x": x, "y": y} for x, y in points]}


@pytest.mark.parametrize("change", [
    {"points": [{"x": 90}]},
    {"points": [{"x": -1, "y": 90}]},
    {"size": 9},
    {"tool": "brush"},
    {"id": ""},
])
def test_vector_request_rejects_invalid_strokes(client, change):
    session = client.post("/api/sessions").json()["id"]
    payload = request(1, encode_png(ink_image(TOP)))
    payload["strokes"] = [dict(stroke("line", *TOP), **change)]
    response = client.post(f"/api/sessions/{session}/suggest", json=payload)
    assert response.status_code == 422


def test_duplicate_stroke_identity_is_rejected(client):
    session = client.post("/api/sessions").json()["id"]
    payload = request(1, encode_png(ink_image(TOP)))
    payload["strokes"] = [stroke("same", *TOP), stroke("same", *BOTTOM)]
    assert client.post(f"/api/sessions/{session}/suggest", json=payload).status_code == 422


def test_short_continuation_contract_never_returns_the_completion_image(manager_factory):
    partial = ink_image(((90, 90), (230, 90)))
    manager, engine, session = manager_factory(Response(ink_image(TOP, BOTTOM)))
    payload = request(1, partial)
    payload.update(strokes=[stroke("arm", (90, 90), (230, 90))], focus={"x": 230, "y": 90}, lookahead=60)
    manager.submit(session, payload)
    result = wait_finished(manager, session, "ready")
    assert result["ghost"] is None
    assert result["paths_version"] == 1
    assert result["suggestions"], "A correct partial curve must yield a usable continuation"
    suggestion = result["suggestions"][0]
    points = np.asarray(suggestion["points"])
    length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
    assert 10 <= length <= 61
    assert np.linalg.norm(points[0] - [230, 90]) <= 2.5
    assert points[-1, 0] > 260 and np.abs(points[:, 1] - 90).max() <= 2.5
    assert suggestion["path_id"] in {p["id"] for p in result["paths"]}
    assert not any(np.asarray(p["points"])[:, 1].mean() > 500 for p in result["paths"]), "Unmatched remote body parts cannot enter the published path pool"
    assert len(engine.calls) == 1


def test_displaced_short_features_cannot_pass_or_be_reused_from_cache(manager_factory):
    contour = [(160, 80), (600, 80), (600, 680), (160, 680), (160, 80)]
    features = [((280, 270), (310, 270)), ((430, 270), (460, 270)), ((350, 360), (395, 360))]
    original = ink_image(contour, *features)
    wrong = ink_image(contour, *[[(x, y + 30) for x, y in feature] for feature in features])
    assert alignment(original, wrong)["accepted"], "Fixture must reproduce the old global gate's blind spot"
    manager, engine, session = manager_factory(Response(wrong), Response(wrong))
    # A first drawing compatible with this candidate is allowed into cache.
    first = request(1, wrong)
    first["strokes"] = [stroke("outline", *contour)] + [stroke(f"feature-{i}", *[(x, y + 30) for x, y in feature]) for i, feature in enumerate(features)]
    manager.submit(session, first)
    wait_finished(manager, session, "ready")
    changed = request(2, original)
    changed["strokes"] = [stroke("outline", *contour)] + [stroke(f"feature-{i}", *feature) for i, feature in enumerate(features)]
    manager.submit(session, changed)
    rejected = wait_finished(manager, session, "rejected")
    assert len(engine.calls) == 2, "A misplaced short feature must invalidate candidate reuse"
    assert rejected["paths"] == [] and rejected["suggestions"] == []
    assert rejected["ghost"] is None
    assert rejected["metrics"]["accepted"] is False
