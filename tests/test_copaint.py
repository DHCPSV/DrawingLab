"""V3 behaviour tests with explicit image fixtures; not model-quality evidence."""
import threading

import numpy as np
import pytest
from PIL import Image, ImageDraw

from drawinglab.copaint import CopaintJobManager, completion_preview, style_profile, update_plan
from test_smoke import ScriptedEngine, Response, ink_image, request, wait_finished, TOP, BOTTOM


@pytest.fixture
def copaint_factory(project_temp):
    managers, gates = [], []
    def create(*responses):
        for response in responses:
            if response.gate:
                gates.append(response.gate)
        engine = ScriptedEngine(*responses)
        manager = CopaintJobManager(engine, project_temp / str(len(managers)))
        managers.append(manager)
        return manager, engine, manager.create()
    yield create
    for gate in gates:
        gate.set()
    for manager in managers:
        manager.close()
        assert not manager.worker.is_alive()


@pytest.fixture
def project_temp(tmp_path):
    from pathlib import Path
    assert tmp_path.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    return tmp_path


def test_whole_preview_is_black_alpha_and_never_changes_user_ink():
    ink = ink_image(((90, 90), (200, 90)))
    before = ink.tobytes()
    _, ghost, metrics = completion_preview(ink_image(TOP, BOTTOM), ink)
    a = np.asarray(ghost)
    assert not a[:, :, :3].any()
    assert a[600, 400, 3] > 200, "Distant unpainted parts must be present, not just a tip suffix"
    assert not a[90, 150, 3], "Existing ink must not be darkened by a duplicate ghost"
    assert ink.tobytes() == before
    assert metrics["semantic_alignment_verified"] is False
    assert metrics["accepted"]


def test_blank_dense_or_no_new_lines_cannot_be_successful_completion():
    ink = ink_image(TOP)
    for candidate in (Image.new("L", ink.size, 255), Image.new("L", ink.size, 0), ink.copy()):
        _, ghost, metrics = completion_preview(candidate, ink)
        assert not metrics["accepted"]
        assert not np.asarray(ghost)[:, :, 3].any()


def test_matching_next_stroke_still_runs_model_with_new_ink(copaint_factory):
    first = ink_image(((90, 90), (220, 90)))
    latest = ink_image(((90, 90), (420, 90)))
    manager, engine, sid = copaint_factory(Response(ink_image(TOP, BOTTOM)), Response(ink_image(TOP, BOTTOM)))
    manager.submit(sid, request(1, first))
    initial = wait_finished(manager, sid, "ready")
    manager.submit(sid, request(2, latest))
    result = wait_finished(manager, sid, "ready")
    assert len(engine.calls) == 2
    assert engine.calls[1]["ink"].tobytes() == latest.tobytes()
    assert initial["candidate_id"] != result["candidate_id"]
    assert result["source_revision"] == result["revision"] == 2
    assert result["ghost"].startswith("data:image/png;base64,")
    assert not result["metrics"]["cached"]


def test_inflight_old_candidate_not_relabelled_as_new_revision(copaint_factory):
    first_gate, second_gate = threading.Event(), threading.Event()
    first = ink_image(((90, 90), (220, 90)))
    latest = ink_image(((90, 90), (400, 90)))
    manager, engine, sid = copaint_factory(Response(ink_image(TOP, BOTTOM), first_gate), Response(ink_image(TOP, BOTTOM), second_gate))
    manager.submit(sid, request(1, first))
    engine.started.get(timeout=2)
    manager.submit(sid, request(2, latest))
    first_gate.set()
    engine.started.get(timeout=2)
    intermediate = manager.result(sid)
    assert intermediate["ghost"] is None
    assert intermediate["source_revision"] is None
    second_gate.set()
    assert wait_finished(manager, sid, "ready")["source_revision"] == 2


def test_erase_or_intent_invalidation_discards_inflight_output(copaint_factory):
    gate = threading.Event()
    manager, engine, sid = copaint_factory(Response(ink_image(TOP, BOTTOM), gate))
    manager.submit(sid, request(1, ink_image(((90, 90), (220, 90)))))
    engine.started.get(timeout=2)
    manager.invalidate(sid, 2, 1, True)
    gate.set()
    result = wait_finished(manager, sid, "idle")
    assert result["ghost"] is None and result["candidate_id"] is None


def test_correction_expands_to_old_connected_arm_not_just_new_line():
    old = ink_image(((300, 150), (300, 300)))
    prior = ink_image(((300, 150), (300, 600), (440, 600)))
    changed = ink_image(((300, 150), (300, 300), (500, 330)))
    mask, plan = update_plan(old, changed, prior)
    assert plan["kind"] == "correction"
    assert np.asarray(mask)[600, 430] == 255, "Old arm endpoint must become editable"
    mask, plan = update_plan(old, Image.new("L", old.size, 255), prior, eraser=True)
    assert plan["deleted_pixels"] > 0 and np.asarray(mask)[600, 430] == 255


def test_style_does_not_learn_from_erased_strokes_or_generated_candidate():
    strokes = [{"tool": "pen", "points": [{"x": x, "y": y} for x,y in [(100,100),(200,150),(100,200)]]}] * 5
    profile = style_profile(ink_image(TOP), strokes)
    assert profile["visible_strokes"] == 0
    assert profile["confidence"] == "insufficient"
    assert "angular" not in profile["description"]
