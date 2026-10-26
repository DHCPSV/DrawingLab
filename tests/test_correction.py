"""Geometry regressions with known targets; these do not rate AI art quality."""
import copy

import numpy as np
import pytest
from PIL import Image, ImageDraw

from drawinglab.correction import _crossings, _paint, _paint_patch, _safe_displacement, propose_correction


def stroke(identifier, points, tool="pen", size=3):
    return {"id": identifier, "tool": tool, "size": size,
            "points": [{"x": float(x), "y": float(y)} for x, y in points]}


def render(strokes, size=(180, 180)):
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    for item in strokes:
        xy = [(p["x"], p["y"]) for p in item["points"]]
        if len(xy) > 1:
            draw.line(xy, fill=0 if item["tool"] == "pen" else 255, width=int(item["size"]))
        elif xy:
            draw.point(xy[0], fill=0)
    return image


def points(update, key="target_points"):
    return np.array([[p["x"], p["y"]] for p in update[key]])


def test_coherent_shift_matches_ordered_target_and_preserves_sample_identity():
    original = stroke("cheek", [(x, 55 + .004*(x-65)**2) for x in np.linspace(25, 105, 41)])
    shifted = copy.deepcopy(original)
    for p in shifted["points"]:
        p["y"] += 4
    saved = copy.deepcopy(original)
    result = propose_correction(render([original]), render([shifted]), [original], source_revision=7, intent_revision=3)
    assert result["source_revision"] == 7 and result["intent_revision"] == 3
    assert original == saved
    assert result["metrics"]["ai_matched_strokes"] == 1
    update = result["stroke_updates"][0]
    assert update["original_points"] == original["points"]
    assert len(update["target_points"]) == len(original["points"])
    displacement = points(update) - points(update, "original_points")
    assert 2.5 < np.median(displacement[:, 1]) < 5.0
    assert np.std(displacement[:, 1]) < 1.0
    assert np.max(np.linalg.norm(displacement, axis=1)) <= update["displacement_limit"] + .001


def test_curved_target_improves_actual_curve_fit():
    xy = [(x, 65 + 3*np.sin((x-20)*np.pi/100)) for x in np.linspace(20, 120, 51)]
    target_xy = [(x, 65 + 7*np.sin((x-20)*np.pi/100)) for x in np.linspace(20, 120, 101)]
    original, target = stroke("contour", xy), stroke("contour", target_xy)
    result = propose_correction(render([original]), render([target]), [original])
    assert result["metrics"]["ai_matched_strokes"] == 1
    adjusted = points(result["stroke_updates"][0])
    expected = 65 + 7*np.sin((adjusted[:, 0]-20)*np.pi/100)
    before = np.mean(np.abs(np.array(xy)[:, 1] - (65 + 7*np.sin((np.array(xy)[:, 0]-20)*np.pi/100))))
    assert np.mean(np.abs(adjusted[:, 1] - expected)) < before * .6


def test_adjacent_feature_cannot_snap_onto_target_owned_by_another_stroke():
    eye = stroke("eye", [(x, 60) for x in range(30, 111, 2)])
    brow = stroke("brow", [(x, 50) for x in range(30, 111, 2)])
    target = stroke("only-brow", [(x, 53) for x in range(30, 111, 2)])
    result = propose_correction(render([eye, brow]), render([target]), [eye, brow])
    assert "eye" not in [u["id"] for u in result["stroke_updates"]]
    assert any(u["id"] == "brow" and u["method"] == "ai_curve_fit" for u in result["stroke_updates"])


def test_ambiguous_parallel_targets_keep_original_instead_of_choosing_arbitrarily():
    original = stroke("line", [(x, 70) for x in range(25, 126, 2)])
    targets = [stroke("a", [(25, 66), (125, 66)]), stroke("b", [(25, 74), (125, 74)])]
    result = propose_correction(render([original]), render(targets), [original])
    assert result["metrics"]["ai_matched_strokes"] == 0
    assert not result["stroke_updates"]


def test_locked_stroke_exactly_unchanged_and_region_does_not_leak():
    first = stroke("locked", [(x, 30) for x in range(20, 141, 2)])
    second = stroke("selected", [(x, 80) for x in range(20, 141, 2)])
    target = [stroke("a", [(20, 34), (140, 34)]), stroke("b", [(20, 84), (140, 84)])]
    region = {"x": 50, "y": 50, "width": 60, "height": 70}
    result = propose_correction(render([first, second]), render(target), [first, second], region, ["locked"])
    assert result["metrics"]["locked_strokes"] == 1
    assert [u["id"] for u in result["stroke_updates"]] == ["selected"]
    update = result["stroke_updates"][0]
    old, new = points(update, "original_points"), points(update)
    outside = (old[:, 0] <= 50) | (old[:, 0] >= 110)
    assert np.array_equal(old[outside], new[outside])
    assert (new[(old[:, 0] > 65) & (old[:, 0] < 95), 1] > 82).all()


def test_later_eraser_protects_whole_touched_stroke_and_keeps_operation_history():
    line = stroke("erased", [(x, 70 + np.sin(x)*.5) for x in range(20, 121)])
    eraser = stroke("eraser", [(65, 60), (65, 80)], "eraser", 10)
    later = stroke("later-pen", [(x, 120 + .7*np.sin(x)) for x in range(20, 121)])
    source = [line, eraser, later]
    snapshot = copy.deepcopy(source)
    result = propose_correction(render(source), None, source, mode="cleanup")
    assert source == snapshot
    assert result["metrics"]["eraser_protected_strokes"] == 1
    assert "erased" not in [u["id"] for u in result["stroke_updates"]]
    assert "eraser" not in [u["id"] for u in result["stroke_updates"]]
    assert "later-pen" in [u["id"] for u in result["stroke_updates"]]


def test_cleanup_reduces_high_frequency_jitter_without_claiming_ai():
    xy = [(x, 80 + .8*(-1)**x) for x in range(20, 121)]
    original = stroke("jitter", xy)
    result = propose_correction(render([original]), None, [original], mode="cleanup")
    assert result["metrics"]["ai_matched_strokes"] == 0
    assert result["metrics"]["cleanup_only_strokes"] == 1
    update = result["stroke_updates"][0]
    after = points(update)
    assert after[[0, -1]].tolist() == np.array(xy)[[0, -1]].tolist()
    assert np.std(after[5:-5, 1]) < np.std(np.array(xy)[5:-5, 1]) * .6
    assert update["max_displacement"] <= 1.401


def test_cleanup_retains_intentional_sharp_corner():
    xy = [(x, 80) for x in range(20, 81)] + [(80, y) for y in range(81, 141)]
    original = stroke("angular", xy)
    result = propose_correction(render([original]), None, [original], mode="cleanup")
    if result["stroke_updates"]:
        after = points(result["stroke_updates"][0])
        assert np.linalg.norm(after[60] - np.array([80, 80])) < .05


def test_joint_between_sparse_vertices_remains_attached():
    horizontal = stroke("sparse", [(20, 80), (140, 80)])
    vertical = stroke("joint", [(80, 80), (80, 130)])
    target = [stroke("a", [(20, 85), (140, 85)]), stroke("b", [(80, 85), (80, 130)])]
    result = propose_correction(render([horizontal, vertical]), render(target), [horizontal, vertical])
    assert "sparse" not in [u["id"] for u in result["stroke_updates"]]
    for update in result["stroke_updates"]:
        if update["id"] == "joint":
            assert update["target_points"][0] == vertical["points"][0]


def test_intersection_guard_backs_off_a_new_loop_including_slider_intermediates():
    original = np.array([[10., 10.], [30, 10], [30, 30], [10, 30]])
    folded = np.array([[10., 10.], [30, 30], [30, 10], [10, 30]])
    safe, scale = _safe_displacement(original, folded-original, [])
    assert scale < 1
    for amount in (.25, .5, .75, 1.):
        assert not _crossings(original + amount*safe)


def test_blank_ai_target_is_honest_cleanup_fallback_and_zero_input_is_no_change():
    original = stroke("jitter", [(x, 80 + .8*(-1)**x) for x in range(20, 121)])
    blank = Image.new("L", (180, 180), 255)
    result = propose_correction(render([original]), blank, [original])
    assert not result["metrics"]["candidate_usable"]
    assert result["metrics"]["ai_matched_strokes"] == 0
    assert result["metrics"]["cleanup_only_strokes"] == 1
    empty = propose_correction(blank, blank, [])
    assert empty["status"] == "no_change" and empty["stroke_updates"] == []


def test_duplicate_stroke_ids_rejected_before_unsafe_updates():
    source = stroke("same", [(10, 20), (80, 20)])
    with pytest.raises(ValueError, match="唯一"):
        propose_correction(render([source]), None, [source, source], mode="cleanup")


def test_copy_of_existing_raster_does_not_claim_new_ai_shape_work():
    source = stroke("jitter", [(x, 75 + .8*(-1)**x) for x in range(20, 121)])
    image = render([source])
    result = propose_correction(image, image.copy(), [source])
    assert result["metrics"]["ai_matched_strokes"] == 0
    assert result["metrics"]["cleanup_only_strokes"] == 1
    assert all(u["method"] == "geometric_cleanup" for u in result["stroke_updates"])


@pytest.mark.parametrize("color", [0, 255])
def test_constant_target_cannot_supply_an_ai_correspondence(color):
    source = stroke("line", [(20, 70), (120, 70)])
    result = propose_correction(render([source]), Image.new("L", (180, 180), color), [source])
    assert not result["metrics"]["candidate_usable"]
    assert result["metrics"]["ai_matched_strokes"] == 0


def test_legal_100000_point_document_has_bounded_geometry_work():
    # A legitimate heavy overdraw history used to cause quadratic crossing
    # checks against hundreds of coincident strokes. Preserve it and report
    # bounded skips instead of decimating vertices or stalling the worker.
    source = [stroke(f"s{i}", [(x, 75 + .8*(-1)**x) for x in range(20, 120)]) for i in range(1000)]
    image = render([source[0]])
    result = propose_correction(image, None, source, mode="cleanup")
    assert sum(len(s["points"]) for s in source) == 100_000
    assert result["metrics"]["budget_skipped_strokes"] == 1000
    assert result["metrics"]["crossing_pair_budget_used"] <= result["metrics"]["crossing_pair_budget"]
    assert result["stroke_updates"] == []


def test_cross_stroke_guard_checks_both_moving_strokes_at_slider_amounts():
    first = np.array([[10., 10.], [90., 10.]])
    neighbour = np.array([[30., 20.], [70., 20.]])
    changed_neighbour = neighbour + np.array([0., 1.])
    delta = np.array([[0., 20.], [0., 0.]])
    safe, scale = _safe_displacement(first, delta, [(neighbour, changed_neighbour)])
    assert scale < 1
    for amount in (.25, .5, .75, 1.):
        assert not _crossings(first + amount*safe, neighbour + amount*(changed_neighbour-neighbour))


def test_raster_patch_and_full_mask_agree_at_half_pixel_coordinates():
    xy = np.array([[29., 40.5], [45., 61.5], [77., 100.5]])
    full = _paint((180, 180), xy, 1)
    patch, (x, y, right, bottom) = _paint_patch((180, 180), xy, 1)
    reconstructed = np.zeros_like(full)
    reconstructed[y:bottom, x:right] = patch
    assert np.array_equal(full, reconstructed)


def test_junction_stitch_keeps_a_long_original_contour_matchable():
    original = stroke("contour", [(x, 60) for x in range(20, 131, 2)])
    target = [stroke("continued", [(20, 64), (130, 64)]), stroke("branch", [(70, 64), (85, 95)])]
    result = propose_correction(render([original]), render(target), [original])
    assert result["metrics"]["ai_matched_strokes"] == 1
    update = result["stroke_updates"][0]
    assert update["target_curve_id"].startswith("joined:")
    assert np.median(points(update)[:, 1]) > 62.5
    assert len(update["target_points"]) == len(original["points"])
