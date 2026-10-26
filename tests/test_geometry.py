"""Discriminating geometry cases, all synthetic and independent of AI quality."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from drawinglab.geometry import assess_geometry, build_geometry, select_continuations
from drawinglab.imaging import alignment


def canvas(*paths, width=3):
    image = Image.new("L", (768, 768), 255)
    for path in paths:
        ImageDraw.Draw(image).line(path, fill=0, width=width)
    return image


def stroke(points, identifier="s1", tool="pen"):
    return {"id": identifier, "tool": tool, "size": 3,
            "points": [{"x": x, "y": y} for x, y in points]}


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def test_single_centreline_and_ordered_short_suffix():
    actual = [(100, 200), (160, 200)]
    proposed = [(100, 200), (350, 200)]
    geometry = build_geometry(canvas(proposed, width=7))
    assert len(geometry["paths"]) == 1  # Not two outline edges.
    ink = canvas(actual)
    assessment = assess_geometry(ink, geometry, [stroke(actual)])
    assert assessment["accepted"]
    assert assessment["matched_path_ids"] == [geometry["paths"][0]["id"]]
    hints = select_continuations(geometry, ink, [stroke(actual)], {"x": 160, "y": 200}, 45)
    assert len(hints) == 1
    hint = hints[0]
    assert hint["source_stroke_id"] == "s1"
    assert math.dist(hint["points"][0], actual[-1]) < .01
    assert 40 <= path_length(hint["points"]) <= 46
    assert hint["points"][-1][0] > 200
    assert max(abs(point[1] - 200) for point in hint["points"]) <= 1


def test_reversed_curve_storage_still_continues_actual_drawing_direction():
    actual = [(280, 220), (210, 220)]
    geometry = build_geometry(canvas([(100, 220), (280, 220)]))
    ink = canvas(actual)
    hints = select_continuations(geometry, ink, [stroke(actual)])
    assert hints and hints[0]["points"][-1][0] < 180


def test_shifted_eyes_and_mouth_cannot_hide_behind_large_correct_outline():
    head = [(384 + 225 * math.cos(t), 380 + 305 * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 650)]
    features = [[(280, 270), (310, 270)], [(430, 270), (460, 270)], [(350, 360), (395, 360)]]
    original = canvas(head, *features, width=5)
    candidate = canvas(head, *[[(x, y + 30) for x, y in path] for path in features], width=5)
    strokes = [stroke(head, "head")] + [stroke(points, name)
                                         for points, name in zip(features, ("eye-left", "eye-right", "mouth"))]
    assert alignment(original, candidate)["accepted"]  # Regression really distinguishes the old gate.
    assessment = assess_geometry(original, build_geometry(candidate), strokes)
    assert not assessment["accepted"]
    failed = {s["id"] for s in assessment["per_stroke"] if not s["accepted"]}
    assert {"eye-left", "eye-right", "mouth"} <= failed


def test_parallel_offset_arm_is_rejected_without_connector():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 206), (300, 206)]))
    ink = canvas(actual)
    assert not assess_geometry(ink, geometry, [stroke(actual)])["accepted"]
    assert select_continuations(geometry, ink, [stroke(actual)]) == []


def test_perpendicular_crossing_at_tip_is_not_a_matching_arm():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(180, 100), (180, 300)]))
    ink = canvas(actual)
    assert not assess_geometry(ink, geometry, [stroke(actual)])["accepted"]
    assert select_continuations(geometry, ink, [stroke(actual)]) == []


def test_only_backward_geometry_has_no_forward_hint():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas(actual))
    assert assess_geometry(canvas(actual), geometry, [stroke(actual)])["accepted"]
    assert select_continuations(geometry, canvas(actual), [stroke(actual)]) == []


def test_hint_stops_at_junction_without_choosing_a_branch():
    actual = [(100, 220), (150, 220)]
    geometry = build_geometry(canvas([(100, 220), (330, 220)], [(190, 170), (190, 270)]))
    hints = select_continuations(geometry, canvas(actual), [stroke(actual)], length=100)
    assert hints
    assert max(point[0] for point in hints[0]["points"]) <= 191
    assert all(abs(point[1] - 220) <= 1 for point in hints[0]["points"])


def test_dot_and_closed_outline_have_no_invented_next_stroke():
    dot = [(100, 200)]
    ink = Image.new("L", (768, 768), 255)
    ImageDraw.Draw(ink).ellipse((99, 199, 101, 201), fill=0)
    geometry = build_geometry(canvas([(100, 200), (300, 200)]))
    assert select_continuations(geometry, ink, [stroke(dot)]) == []
    circle = [(380 + 100 * math.cos(t), 380 + 100 * math.sin(t))
              for t in np.linspace(0, 2 * math.pi, 300)]
    assert select_continuations(build_geometry(canvas(circle)), canvas(circle), [stroke(circle)]) == []


def test_erased_vectors_do_not_remain_alignment_constraints():
    arm = [(100, 200), (220, 200)]
    mouth = [(300, 250), (340, 250)]
    ink = canvas(arm)  # Mouth has been erased from authoritative raster.
    history = [stroke(arm, "arm"), stroke(mouth, "mouth"), stroke(mouth, "erase", "eraser")]
    geometry = build_geometry(canvas([(100, 200), (300, 200)]))
    assessment = assess_geometry(ink, geometry, history)
    assert assessment["accepted"]
    assert [s["id"] for s in assessment["per_stroke"]] == ["arm"]
    assert select_continuations(geometry, ink, history) == []


def test_erased_last_tip_cannot_resurrect_old_continuation():
    actual = [(100, 200), (180, 200)]
    ink = canvas(actual)
    ImageDraw.Draw(ink).rectangle((160, 190, 190, 210), fill=255)
    geometry = build_geometry(canvas([(100, 200), (300, 200)]))
    assert select_continuations(geometry, ink, [stroke(actual)]) == []


def test_small_anchoring_error_has_bounded_correction_not_a_second_line():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 202), (300, 202)]))
    hints = select_continuations(geometry, canvas(actual), [stroke(actual)])
    assert hints
    assert hints[0]["points"][0] == [180, 200]
    assert 0 < hints[0]["gap"] <= 2.5
    assert all(200 <= point[1] <= 202 for point in hints[0]["points"])
    assert hints[0]["points"][-1][1] == 202


def test_raster_compatibility_is_explicit_and_also_rejects_shifted_short_features():
    actual = [[(100, 100), (600, 100)], [(250, 250), (280, 250)]]
    geometry = build_geometry(canvas(actual[0], [(250, 280), (280, 280)]))
    assessment = assess_geometry(canvas(*actual), geometry)
    assert assessment["source"] == "raster"
    assert not assessment["accepted"]
    assert assessment["failed_strokes"] >= 1


def test_parallel_distractor_cannot_steal_current_continuation():
    actual = [(100, 200), (180, 200)]
    geometry = build_geometry(canvas([(100, 200), (290, 200)], [(100, 208), (330, 208)]))
    hints = select_continuations(geometry, canvas(actual), [stroke(actual)])
    assert hints
    assert all(abs(point[1] - 200) < 1 for point in hints[0]["points"])


def test_no_hint_draws_through_existing_separate_ink():
    actual = [(100, 200), (150, 200)]
    blocker = [(190, 180), (190, 220)]
    ink = canvas(actual, blocker)
    geometry = build_geometry(canvas([(100, 200), (350, 200)]))
    hints = select_continuations(geometry, ink, [stroke(blocker, "blocker"), stroke(actual)], length=100)
    assert hints
    assert max(point[0] for point in hints[0]["points"]) < 190


def test_dense_or_empty_candidate_produces_no_geometry():
    actual = [(100, 200), (180, 200)]
    for colour in (0, 255):
        geometry = build_geometry(Image.new("L", (768, 768), colour))
        assert geometry["paths"] == []
        assert not assess_geometry(canvas(actual), geometry, [stroke(actual)])["accepted"]


def test_missing_eye_in_vector_payload_cannot_silently_drop_raster_constraint():
    head = [(100, 100), (600, 100)]
    eye = [(250, 250), (275, 250)]
    geometry = build_geometry(canvas(head))
    assessment = assess_geometry(canvas(head, eye), geometry, [stroke(head, "head")])
    assert not assessment["accepted"]
    assert assessment["unrepresented_components"] == 1
    assert assessment["unrepresented_pixels"] >= 20


def test_raster_only_isolated_dot_remains_a_position_constraint():
    head = [(100, 100), (600, 100)]
    ink = canvas(head)
    ImageDraw.Draw(ink).ellipse((250, 250, 252, 252), fill=0)
    assessment = assess_geometry(ink, build_geometry(canvas(head)))
    assert assessment["source"] == "raster"
    assert not assessment["accepted"]
    assert any(s["id"].startswith("raster:dot") and not s["accepted"] for s in assessment["per_stroke"])


def test_matched_path_owners_preserve_real_stroke_identity():
    actual = [(100, 200), (160, 200)]
    geometry = build_geometry(canvas([(100, 200), (300, 200)]))
    assessment = assess_geometry(canvas(actual), geometry, [stroke(actual, "arm-uuid")])
    assert assessment["accepted"]
    assert assessment["path_owners"] == {assessment["matched_path_ids"][0]: ["arm-uuid"]}


def test_equally_plausible_parallel_paths_produce_no_arbitrary_choice():
    actual = [(100, 200), (160, 200)]
    # One-pixel candidates keep the two paths separate during thinning.
    geometry = build_geometry(canvas([(100, 198), (300, 198)], [(100, 202), (300, 202)], width=1))
    assert len(geometry["paths"]) == 2
    assert select_continuations(geometry, canvas(actual), [stroke(actual)]) == []


def test_tiny_junction_connectors_do_not_create_an_artificial_gap_in_real_curve():
    # Several close junctions split the actual horizontal line into 2px edges.
    # Discarding short edges used to fabricate a >6px topological gap here.
    proposed = canvas([(100, 200), (260, 200)], width=1)
    draw = ImageDraw.Draw(proposed)
    for x in range(150, 161, 2):
        draw.line([(x, 198), (x, 202)], fill=0, width=1)
    actual = [(100, 200), (220, 200)]
    geometry = build_geometry(proposed)
    assert any(p["connector"] for p in geometry["paths"])
    assessment = assess_geometry(canvas(actual), geometry, [stroke(actual)])
    assert assessment["accepted"]
    assert all(fragment["ordered"] for fragment in assessment["per_stroke"][0]["fragments"])
    assert select_continuations(geometry, canvas(actual), [stroke(actual)])
