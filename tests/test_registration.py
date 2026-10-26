"""Small corrections must pass the original strict geometry gates."""
from __future__ import annotations

import copy
import math

import numpy as np
from PIL import Image, ImageDraw

from drawinglab.geometry import assess_geometry, build_geometry, select_continuations
from drawinglab.registration import align_geometry


def canvas(*paths):
    image = Image.new("L", (768, 768), 255)
    for path in paths:
        ImageDraw.Draw(image).line(path, fill=0, width=3)
    return image


def stroke(points, identifier="arm"):
    return {"id": identifier, "tool": "pen", "size": 3,
            "points": [{"x": x, "y": y} for x, y in points]}


def test_small_parallel_translation_becomes_a_strict_valid_continuation():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 204), (300, 204)]))
    original = copy.deepcopy(geometry)
    ink = canvas(actual)
    history = [stroke(actual)]
    assert not assess_geometry(ink, geometry, history)["accepted"]
    aligned, report = align_geometry(ink, geometry, history)
    assert report["applied"]
    assert report["max_displacement_px"] <= 6
    assert assess_geometry(ink, aligned, history)["accepted"]
    assert select_continuations(aligned, ink, history)
    assert geometry == original  # No in-place change or hidden original prefix edit.
    assert report["semantic_alignment_verified"] is False


def test_far_parallel_arm_cannot_be_dragged_back():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 215), (300, 215)]))
    aligned, report = align_geometry(canvas(actual), geometry, [stroke(actual)])
    assert not report["applied"]
    assert aligned is geometry
    assert not assess_geometry(canvas(actual), aligned, [stroke(actual)])["accepted"]


def test_misaligned_short_feature_still_vetoes_registration():
    head = [(100, 100), (600, 100)]
    eye = [(250, 250), (280, 250)]
    geometry = build_geometry(canvas([(100, 104), (600, 104)], [(250, 280), (280, 280)]))
    history = [stroke(head, "head"), stroke(eye, "eye")]
    aligned, report = align_geometry(canvas(head, eye), geometry, history)
    assert not report["applied"]
    assert aligned is geometry


def test_connected_suffix_cannot_move_far_while_prefix_looks_close():
    # A tiny rotation around the drawn beginning would move the remote suffix
    # too far. The displacement bound is checked on that suffix as well.
    actual = [(100, 200), (170, 200)]
    candidate = [(100, 204), (170, 202), (730, 186)]
    geometry = build_geometry(canvas(candidate))
    aligned, report = align_geometry(canvas(actual), geometry, [stroke(actual)])
    if report["applied"]:
        original = {p["id"]: np.asarray(p["points"]) for p in geometry["paths"]}
        for path in aligned["paths"]:
            assert np.linalg.norm(np.asarray(path["points"]) - original[path["id"]], axis=1).max() <= 6.000001
        for transform in report["transforms"]:
            assert abs(transform["rotation_degrees"]) <= 3
            assert .96 <= transform["scale"] <= 1.04


def test_already_good_candidate_is_unchanged_and_not_repeatedly_registered():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 200), (300, 200)]))
    aligned, report = align_geometry(canvas(actual), geometry, [stroke(actual)])
    assert aligned is geometry
    assert not report["applied"]
    assert report["attempts"] == 0


def test_incomplete_transport_does_not_use_registration_to_hide_missing_eye():
    actual = [(100, 200), (180, 200)]
    eye = [(250, 250), (280, 250)]
    geometry = build_geometry(canvas([(100, 204), (300, 204)]))
    aligned, report = align_geometry(canvas(actual, eye), geometry, [stroke(actual)])
    assert aligned is geometry
    assert not report["applied"] and report["attempts"] == 0


def test_registration_rejects_matched_full_path_that_would_leave_canvas():
    actual = [(130, 200), (180, 200)]
    geometry = {"version": 1, "width": 768, "height": 768, "density": .001,
                "paths": [{"id": "border-connected", "closed": False,
                           "points": [[100, 2], [100, 204], [300, 204]], "length": 402}]}
    # Fixing the local 4px offset by translation would put the same full path's
    # top endpoint at y=-2. It must not be clipped or sent to the UI as valid.
    aligned, report = align_geometry(canvas(actual), geometry, [stroke(actual)])
    assert not report["applied"]
    assert aligned is geometry
