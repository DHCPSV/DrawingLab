"""Non-destructive, bounded vector correction proposals.

The AI supplies a raster target, not editable coordinates or semantic labels.
We match complete ordered centreline intervals, then smoothly transfer their
displacement to the *original* stroke samples. Ambiguous curves receive only
small geometric cleanup. No independent point-to-nearest-black-pixel snapping.
"""
from __future__ import annotations

import hashlib
import json
import math
import uuid

import cv2
import numpy as np
from PIL import Image

from .geometry import build_geometry
from .imaging import extract_lines


MAX_DISPLACEMENT = 12.0
MAX_AI_CURVES = 512
MAX_PROCESS_STROKES = 256
MAX_STROKE_POINTS = 2500
# Crossing tests retain the actual original vertices. Bound their work instead
# of dropping vertices (which could hide an erased gap or a tiny intersection).
MAX_CROSSING_PAIRS_PER_STROKE = 350_000
MAX_CROSSING_PAIRS_TOTAL = 2_000_000
MAX_CURVE_CANDIDATES = 32
CHECK_AMOUNTS = (.25, .5, .75, 1.0)


def _points(stroke):
    return np.asarray([[p["x"], p["y"]] for p in stroke.get("points", [])], dtype=float).reshape(-1, 2)


def _arc(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _at(points, arc, positions):
    keep = np.r_[np.diff(arc) > 1e-8, True]
    return np.column_stack([np.interp(positions, arc[keep], points[keep, axis]) for axis in (0, 1)])


def _resample(points, count=None):
    arc = _arc(points)
    count = count or max(2, min(320, int(math.ceil(arc[-1] / 1.5)) + 1))
    return _at(points, arc, np.linspace(0, arc[-1], count))


def _smooth(values, sigma=1.25):
    radius = max(1, int(math.ceil(3 * sigma)))
    kernel = np.exp(-.5 * (np.arange(-radius, radius+1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in (0, 1)])


def _corner_weight(points):
    """Identify persistent corners over a physical 6 px neighbourhood."""
    arc = _arc(points)
    before = points - _at(points, arc, np.maximum(0, arc - 6))
    after = _at(points, arc, np.minimum(arc[-1], arc + 6)) - points
    left, right = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    cosine = (before * after).sum(1) / np.maximum(left * right, 1e-8)
    corners = (cosine < np.cos(np.radians(55))) & (left > 4) & (right > 4)
    weight = np.ones(len(points))
    if corners.any():
        distance = np.abs(arc[:, None] - arc[corners][None, :]).min(1)
        weight = np.minimum(1, distance / 4)
    return weight


def _cleanup(points):
    """Small physical-scale steadiness, not an anatomy or style judgement."""
    arc = _arc(points)
    sampled = _resample(points)
    # A roughly 4 px support sigma removes multi-pixel hand wobble. Sampling
    # density must not determine how much smoothing a drawing receives.
    spacing = max(1e-6, arc[-1] / max(1, len(sampled)-1))
    smooth = _smooth(sampled, sigma=max(.6, 4.0/spacing))
    displacement = smooth - sampled
    tangent = np.gradient(smooth, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-8)
    # Steadiness should not shorten a perfectly straight line merely because
    # Gaussian padding changes its parameterization near the endpoints.
    displacement -= tangent * (displacement*tangent).sum(1, keepdims=True)
    sampled_arc = np.linspace(0, arc[-1], len(sampled))
    endpoint = np.minimum(1, np.minimum(sampled_arc, arc[-1] - sampled_arc) / 5)
    displacement *= (_corner_weight(sampled) * endpoint)[:, None]
    largest = float(np.linalg.norm(displacement, axis=1).max())
    displacement *= min(1, 3.0 / max(largest, 1e-8))
    return _at(displacement, sampled_arc, arc)


def _paint(shape, points, width):
    result = np.zeros(shape, np.uint8)
    if not len(points):
        return result
    xy = np.rint(points).astype(np.int32)
    thickness = max(1, int(math.ceil(width)))
    cv2.polylines(result, [xy], False, 1, thickness, cv2.LINE_8)
    radius = max(1, int(math.ceil(width / 2)))
    cv2.circle(result, tuple(xy[0]), radius, 1, -1)
    cv2.circle(result, tuple(xy[-1]), radius, 1, -1)
    return result


def _paint_patch(shape, points, width):
    """Rasterize a small history item without allocating a full canvas each time."""
    if not len(points):
        return np.zeros((0, 0), np.uint8), (0, 0, 0, 0)
    pad = int(math.ceil(width/2)) + 2
    low = np.maximum(0, np.floor(points.min(0)).astype(int)-pad)
    high = np.minimum([shape[1], shape[0]], np.ceil(points.max(0)).astype(int)+pad+1)
    x, y = low
    right, bottom = high
    if right <= x or bottom <= y:
        return np.zeros((0, 0), np.uint8), (0, 0, 0, 0)
    # Round in global coordinates first: bankers rounding of x.5 changes after
    # an odd integer translation and would make this stroke appear as its own
    # neighbour in the coverage map, pinning every point as a false junction.
    return _paint((bottom-y, right-x), np.rint(points)-low, width), (x, y, right, bottom)


def _lookup(array, points):
    xy = np.rint(points).astype(int)
    x, y = np.clip(xy[:, 0], 0, array.shape[1]-1), np.clip(xy[:, 1], 0, array.shape[0]-1)
    return array[y, x]


def _region_weight(points, region):
    if region is None:
        return np.ones(len(points))
    x, y, width, height = (float(region[k]) for k in ("x", "y", "width", "height"))
    distance = np.minimum.reduce((points[:, 0] - x, x + width - points[:, 0],
                                  points[:, 1] - y, y + height - points[:, 1]))
    fade = max(1, min(14, min(width, height) * .12))
    t = np.clip(distance / fade, 0, 1)
    return t*t*(3 - 2*t)


def _joint_weight(points, others):
    if not others.any():
        return np.ones(len(points))
    distance = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
    arc = _arc(points)
    sampled = _resample(points, max(2, min(2048, int(arc[-1]) + 1)))
    near = _lookup(distance, sampled) <= 2.0
    if not near.any():
        return np.ones(len(points))
    joint_arc = np.linspace(0, arc[-1], len(sampled))[near]
    proximity = np.abs(arc[:, None] - joint_arc[None, :]).min(1)
    weight = np.clip((proximity - 2.0) / 5, 0, 1)
    # A T-junction can fall between two sparse original vertices. Pin both
    # vertices of that segment; the slider cannot express an interior anchor.
    positions = np.searchsorted(arc, joint_arc, side="right")
    weight[np.clip(positions, 0, len(points)-1)] = 0
    weight[np.clip(positions-1, 0, len(points)-1)] = 0
    return weight


def _curve_intervals(source, target):
    """Whole curves first, or a contiguous interval of a longer target curve."""
    if len(target) < 2:
        return []
    source_closed = np.linalg.norm(source[0] - source[-1]) < 3
    target_closed = np.linalg.norm(target[0] - target[-1]) < 3
    if source_closed and target_closed:
        loop = target[:-1]
        if len(loop) < 3:
            return []
        start = int(np.argmin(np.linalg.norm(loop - source[0], axis=1)))
        forward = np.roll(loop, -start, axis=0)
        reverse = np.roll(loop[::-1], -(len(loop)-1-start), axis=0)
        return [(np.vstack((p, p[0])), .0) for p in (forward, reverse)]
    if source_closed != target_closed:
        return []
    result = [(target, .0), (target[::-1], .0)]
    first = int(np.argmin(np.linalg.norm(target - source[0], axis=1)))
    last = int(np.argmin(np.linalg.norm(target - source[-1], axis=1)))
    if abs(first-last) >= 3 and {first, last} != {0, len(target)-1}:
        interval = target[min(first, last):max(first, last)+1]
        result.append((interval if first < last else interval[::-1], .12))
    return result


def _target_curves(graph):
    """Offer tangent-consistent junction continuations as well as graph edges.

    Thinning splits a cheek into several edges where hair or another stroke
    touches it. Joining the unambiguous straight-through continuation keeps
    long original strokes matchable; actual branch ambiguity remains a stop.
    """
    raw = [{"id": p["id"], "points": np.asarray(p["points"], dtype=float),
            "length": p["length"]} for p in graph["paths"] if len(p["points"]) >= 2]
    if not raw:
        return []
    endpoints = {}
    for index, path in enumerate(raw):
        for reverse in (False, True):
            point = path["points"][-1 if reverse else 0]
            key = tuple(np.rint(point).astype(int))
            endpoints.setdefault(key, []).append((index, reverse))

    def extend(points, used):
        for _ in range(18):
            end = points[-1]
            arc = _arc(points)
            tangent = end - _at(points, arc, [max(0, arc[-1]-6)])[0]
            tangent /= max(np.linalg.norm(tangent), 1e-8)
            options = []
            ex, ey = np.rint(end).astype(int)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for index, reverse in endpoints.get((ex+dx, ey+dy), []):
                        if index in used:
                            continue
                        outgoing = raw[index]["points"][::-1] if reverse else raw[index]["points"]
                        out_arc = _arc(outgoing)
                        direction = _at(outgoing, out_arc, [min(out_arc[-1], 6)])[0] - outgoing[0]
                        direction /= max(np.linalg.norm(direction), 1e-8)
                        score = float(tangent @ direction) - .08*float(np.linalg.norm(outgoing[0]-end))
                        if score > .4:
                            options.append((score, index, outgoing))
            options.sort(key=lambda item: item[0], reverse=True)
            if not options:
                break
            if len(options) > 1 and options[0][0]-options[1][0] < .12:
                break
            _, index, outgoing = options[0]
            used.add(index)
            points = np.vstack((points, outgoing[1:] if np.linalg.norm(points[-1]-outgoing[0]) < .1 else outgoing))
            if _arc(points)[-1] > 1800:
                break
        return points, used

    result = list(raw)
    used_combinations = set()
    for index, path in enumerate(raw):
        if path["length"] < 8:
            continue
        joined, used = extend(path["points"], {index})
        joined, used = extend(joined[::-1], used)
        key = tuple(sorted(used))
        if len(used) > 1 and key not in used_combinations:
            used_combinations.add(key)
            result.append({"id": "joined:" + "+".join(raw[i]["id"] for i in key),
                           "points": joined, "length": float(_arc(joined)[-1])})
    return result


def _match(source, paths, bound, others_distance):
    """Ordered arclength registration with curve-level rejection/ambiguity."""
    sampled = _resample(source)
    source_length = _arc(sampled)[-1]
    candidates = []
    source_low, source_high = sampled.min(0) - bound*1.7, sampled.max(0) + bound*1.7
    source_tangent = np.diff(_smooth(sampled), axis=0)
    nearby = []
    for path in paths:
        target = path["points"]
        if np.any(target.max(0) < source_low) or np.any(target.min(0) > source_high):
            continue
        endpoint_cost = (np.linalg.norm(target-sampled[0], axis=1).min()
                         + np.linalg.norm(target-sampled[-1], axis=1).min())
        nearby.append((float(endpoint_cost), path))
    nearby.sort(key=lambda item: item[0])
    for _, path in nearby[:MAX_CURVE_CANDIDATES]:
        target = path["points"]
        best = None
        for interval, trim_cost in _curve_intervals(sampled, target):
            length = _arc(interval)[-1]
            ratio = length / max(source_length, 1e-8)
            if not .66 < ratio < 1.5:
                continue
            # Refine correspondence *inside this one ordered curve*. Pure
            # normalized arclength can slide samples many pixels along an
            # unchanged shape when the hand-drawn path contains small wiggles.
            # Monotone projected parameters remove that reparameterization
            # artifact without allowing samples to switch to a nearby feature.
            fine = _resample(interval, max(len(sampled), min(640, int(length)+1)))
            fine_arc = _arc(fine)
            closest = ((sampled[:, None, :] - fine[None, :, :])**2).sum(2).argmin(1)
            parameter = np.maximum.accumulate(fine_arc[closest])
            parameter = .85*parameter + .15*np.linspace(0, fine_arc[-1], len(sampled))
            parameter[0], parameter[-1] = 0., fine_arc[-1]
            paired = _at(fine, fine_arc, parameter)
            delta = paired - sampled
            norms = np.linalg.norm(delta, axis=1)
            if np.quantile(norms, .90) > bound*1.6 or norms.mean() > bound*1.15:
                continue
            target_tangent = np.diff(_smooth(paired), axis=0)
            cosine = (source_tangent * target_tangent).sum(1) / np.maximum(
                np.linalg.norm(source_tangent, axis=1) * np.linalg.norm(target_tangent, axis=1), 1e-8)
            if np.median(cosine) < .78 or (cosine > .35).mean() < .78:
                continue
            # A target lying nearer another source feature is not evidence that
            # this stroke should jump onto that feature (eye/brow, parallel hair).
            if others_distance is not None:
                distance_to_own = np.sqrt(((paired[:, None, :] - sampled[None, :, :])**2).sum(2).min(1))
                owns = _lookup(others_distance, paired) + .9 >= distance_to_own
                if owns.mean() < .80:
                    continue
            shape_cost = float(np.linalg.norm(np.diff(delta, axis=0), axis=1).mean())
            score = float(norms.mean()/bound + .65*(1 - np.median(cosine))
                          + .30*abs(math.log(ratio)) + .05*shape_cost + trim_cost)
            if best is None or score < best[0]:
                best = (score, paired, path["id"])
        if best:
            candidates.append(best)
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    winner = candidates[0]
    if len(candidates) > 1:
        second = candidates[1]
        separation = np.linalg.norm(winner[1] - second[1], axis=1).mean()
        if second[0] - winner[0] < .14 and separation > 2.0:
            return None
    # A raster copy is not a new AI shape proposal. Pixel quantization and
    # changed sample parameterization alone must not manufacture AI edits.
    nearest_source = np.sqrt(((winner[1][:, None, :] - sampled[None, :, :])**2).sum(2).min(1))
    endpoint_change = np.linalg.norm(winner[1][[0, -1]] - sampled[[0, -1]], axis=1).max()
    if np.quantile(nearest_source, .9) <= 1.35 and endpoint_change <= 1.8:
        return None
    displacement = _smooth(winner[1] - sampled, 1.35)
    mapped = _at(displacement, np.linspace(0, _arc(source)[-1], len(sampled)), _arc(source))
    return mapped, float(np.clip(1 - winner[0] / 2, .15, .95)), winner[2]


def _crossings(a, b=None):
    """Segment identities of proper crossings; existing crossings are allowed."""
    same = b is None
    b = a if same else b
    if len(a) < 2 or len(b) < 2:
        return set()
    start, end = b[:-1], b[1:]
    answer = set()
    for i, (p, q) in enumerate(zip(a[:-1], a[1:])):
        low, high = np.minimum(p, q), np.maximum(p, q)
        possible = ((np.maximum(start, end) >= low-1e-7).all(1)
                    & (np.minimum(start, end) <= high+1e-7).all(1))
        if same:
            possible[:i+2] = False
            if i == 0 and np.linalg.norm(a[0] - a[-1]) < 1e-7:
                possible[-1] = False
        js = np.where(possible)[0]
        if not len(js):
            continue
        r, s = start[js], end[js]
        cross = lambda u, v: u[..., 0]*v[..., 1] - u[..., 1]*v[..., 0]
        o1, o2 = cross(q-p, r-p), cross(q-p, s-p)
        o3, o4 = cross(s-r, p-r), cross(s-r, q-r)
        valid = (o1*o2 < -1e-8) & (o3*o4 < -1e-8)
        answer.update((i, int(j)) for j in js[valid])
    return answer


def _safe_displacement(original, delta, neighbours):
    """Back off changes introducing crossings at the sampled slider amounts."""
    baseline = _crossings(original)
    pairs = [(other, changed, _crossings(original, other)) for other, changed in neighbours]
    scale = 1.0
    for _ in range(9):
        safe = True
        for amount in CHECK_AMOUNTS:
            current = original + delta * scale * amount
            if _crossings(current) - baseline:
                safe = False
                break
            for other, changed, allowed in pairs:
                if _crossings(current, other + (changed-other)*amount) - allowed:
                    safe = False
                    break
            if not safe:
                break
        if safe:
            return delta * scale, scale
        scale *= .5
    return np.zeros_like(delta), 0.0


def propose_correction(ink: Image.Image, candidate: Image.Image | None, strokes: list[dict],
                       region: dict | None = None, locked_ids: list[str] | None = None,
                       source_revision: int = 0, intent_revision: int = 0,
                       mode: str = "ai") -> dict:
    """Return a reversible proposal; never mutate input strokes or artist ink.

    Coordinates are in the input canvas. Every update has exactly the original
    sample count, so a client can interpolate at any assistance amount. Locked
    and erased strokes are omitted, not removed. Structural deletions are not
    implemented by this first continuous-deformation version.
    """
    if mode not in ("ai", "cleanup"):
        raise ValueError("修稿模式必须是 ai 或 cleanup")
    shape = (ink.height, ink.width)
    locked = set(locked_ids or [])
    records = []
    identifiers = set()
    for stroke in strokes:
        identifier = str(stroke.get("id", ""))
        points = _points(stroke)
        if not identifier or identifier in identifiers or not np.isfinite(points).all():
            raise ValueError("修稿需要唯一笔画 ID 和有限坐标")
        identifiers.add(identifier)
        records.append((stroke, points))
    if region is not None:
        if not all(k in region and np.isfinite(region[k]) for k in ("x", "y", "width", "height")):
            raise ValueError("修稿范围无效")
        if region["width"] <= 0 or region["height"] <= 0:
            raise ValueError("修稿范围必须有面积")
    metrics = {"modified_strokes": 0, "max_displacement": 0., "ai_matched_strokes": 0,
               "cleanup_only_strokes": 0, "locked_strokes": 0, "eraser_protected_strokes": 0,
               "unchanged_strokes": 0, "budget_skipped_strokes": 0, "intersection_backoffs": 0,
               "semantic_alignment_verified": False, "personal_style_learned": False,
               "original_ink_modified": False, "candidate_provided": candidate is not None,
               "candidate_usable": False, "target_paths": 0, "candidate_reason": "not_requested",
               "crossing_pair_budget": MAX_CROSSING_PAIRS_TOTAL,
               "method": "ordered_centerline_arclength_with_bounded_smooth_displacement",
               "intersection_check_amounts": list(CHECK_AMOUNTS)}
    paths = []
    if mode == "ai" and candidate is not None:
        if candidate.size != ink.size:
            candidate = candidate.resize(ink.size, Image.Resampling.LANCZOS)
        graph = build_geometry(extract_lines(candidate))
        exact_copy = np.array_equal(np.asarray(candidate.convert("L")), np.asarray(ink.convert("L")))
        usable = [p for p in _target_curves(graph) if p["length"] >= 5] if not exact_copy else []
        if sum(p["length"] for p in usable) < 12:
            usable = []
        usable.sort(key=lambda p: p["length"], reverse=True)
        paths = [{"id": p["id"], "points": _resample(np.asarray(p["points"]))}
                 for p in usable[:MAX_AI_CURVES]]
        metrics.update(candidate_usable=bool(paths), target_paths=len(paths),
                       candidate_density=graph["density"],
                       candidate_reason=("unchanged_raster_target" if exact_copy else
                                         "line_target" if paths else "blank_dense_or_insufficient_contours"),
                       target_paths_truncated=bool(graph["truncated"] or len(usable) > MAX_AI_CURVES))
    # Preserve operation chronology. An entire pen stroke near a later eraser
    # is protected so displacement cannot uncover any previously erased ink.
    later_erasers = np.zeros(shape, np.uint8)
    has_later_eraser = False
    eraser_protected = set()
    coverage = np.zeros(shape, np.uint16)
    for stroke, points in reversed(records):
        if stroke.get("tool") == "eraser":
            mask, (x, y, right, bottom) = _paint_patch(shape, points, float(stroke.get("size", 20)))
            later_erasers[y:bottom, x:right] |= mask
            has_later_eraser = True
        elif stroke.get("tool") == "pen":
            identifier = stroke["id"]
            if has_later_eraser:
                mask, (x, y, right, bottom) = _paint_patch(shape, points, 3 + MAX_DISPLACEMENT*2)
                if (mask & later_erasers[y:bottom, x:right]).any():
                    eraser_protected.add(identifier)
            mask, (x, y, right, bottom) = _paint_patch(shape, points, 1)
            coverage[y:bottom, x:right] += mask
    updates, accepted = [], {}
    processed = 0
    crossing_budget = MAX_CROSSING_PAIRS_TOTAL
    pens = [(s, p) for s, p in records if s.get("tool") == "pen" and len(p) >= 2]
    pen_low = np.asarray([p.min(0) for _, p in pens]).reshape(-1, 2)
    pen_high = np.asarray([p.max(0) for _, p in pens]).reshape(-1, 2)
    pen_segments = np.asarray([len(p)-1 for _, p in pens])
    for stroke, original in records:
        if stroke.get("tool") != "pen":
            continue
        identifier = stroke["id"]
        if identifier in locked:
            metrics["locked_strokes"] += 1
            continue
        if identifier in eraser_protected:
            metrics["eraser_protected_strokes"] += 1
            continue
        if len(original) < 2 or _arc(original)[-1] < 4 or not _region_weight(original, region).any():
            metrics["unchanged_strokes"] += 1
            continue
        if processed >= MAX_PROCESS_STROKES or len(original) > MAX_STROKE_POINTS:
            metrics["budget_skipped_strokes"] += 1
            continue
        processed += 1
        low, high = original.min(0)-MAX_DISPLACEMENT*2, original.max(0)+MAX_DISPLACEMENT*2
        neighbours_index = np.where((pen_high >= low).all(1) & (pen_low <= high).all(1))[0]
        neighbours_index = [int(i) for i in neighbours_index if pens[i][0]["id"] != identifier]
        segments = len(original)-1
        crossing_cost = segments*(segments-1)//2 + segments*int(pen_segments[neighbours_index].sum())
        if crossing_cost > min(MAX_CROSSING_PAIRS_PER_STROKE, crossing_budget):
            metrics["budget_skipped_strokes"] += 1
            continue
        # Keep only one temporary canvas mask; retaining one per historical
        # stroke would consume gigabytes on otherwise valid long documents.
        other_ink = coverage > _paint(shape, original, 1)
        others_distance = (cv2.distanceTransform((~other_ink).astype(np.uint8), cv2.DIST_L2, 5)
                           if other_ink.any() else None)
        cleanup = _cleanup(original)
        bound = float(np.clip(2 + .045*np.linalg.norm(np.ptp(original, axis=0)), 3, MAX_DISPLACEMENT))
        match = _match(original + cleanup, paths, bound, others_distance) if paths else None
        delta = cleanup.copy()
        if match is not None:
            delta += match[0]
        weight = _region_weight(original, region) * _joint_weight(original, other_ink)
        delta *= weight[:, None]
        largest = float(np.linalg.norm(delta, axis=1).max())
        delta *= min(1, bound / max(largest, 1e-8))
        proposed = np.clip(original + delta, [0, 0], [ink.width, ink.height])
        delta = proposed - original
        if np.linalg.norm(delta, axis=1).max() < .08:
            metrics["unchanged_strokes"] += 1
            continue
        # Limit neighbours to overlapping displacement envelopes. Existing
        # unrelated crossings elsewhere in the page need no quadratic work.
        neighbours = []
        for index in neighbours_index:
            other, other_points = pens[index]
            neighbours.append((other_points, accepted.get(other["id"], other_points)))
        crossing_budget -= crossing_cost
        delta, scale = _safe_displacement(original, delta, neighbours)
        if scale < 1:
            metrics["intersection_backoffs"] += 1
        distances = np.linalg.norm(delta, axis=1)
        if distances.max() < .08 or distances.mean() < .025:
            metrics["unchanged_strokes"] += 1
            continue
        target = original + delta
        accepted[identifier] = target
        ai_changed = match is not None and float(np.linalg.norm(match[0] * weight[:, None] * scale, axis=1).max()) >= .15
        method = "ai_curve_fit" if ai_changed else "geometric_cleanup"
        metrics["ai_matched_strokes" if ai_changed else "cleanup_only_strokes"] += 1
        metrics["max_displacement"] = max(metrics["max_displacement"], float(distances.max()))
        updates.append({"id": identifier,
                        "original_points": [{"x": float(p[0]), "y": float(p[1])} for p in original],
                        "target_points": [{"x": float(p[0]), "y": float(p[1])} for p in target],
                        "method": method, "confidence": round(match[1], 3) if ai_changed else None,
                        "target_curve_id": match[2] if ai_changed else None,
                        "max_displacement": round(float(distances.max()), 3),
                        "displacement_limit": round(bound, 3)})
    metrics["modified_strokes"] = len(updates)
    metrics["crossing_pair_budget_used"] = MAX_CROSSING_PAIRS_TOTAL-crossing_budget
    metrics["max_displacement"] = round(metrics["max_displacement"], 3)
    limitations = ["曲线对应基于几何，未识别眼睛、嘴或人体解剖。",
                   "未匹配笔画只提供小幅稳线；本版不自动删笔、合并或补入黑色笔画。",
                   "与后续橡皮轨迹接近的整笔保持原样，避免复活已擦除的线。",
                   "笔画连接处暂作固定锚点；本版不会把多笔构成的五官整组移位。",
                   "超过几何检查预算的复杂笔画保持原样，可缩小选区后再试。",
                   "交叉保护检查 25%、50%、75%、100% 幅度，尚非连续区间拓扑证明。"]
    payload = json.dumps(strokes, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {"id": uuid.uuid4().hex, "source_revision": source_revision, "intent_revision": intent_revision,
            "input_hash": hashlib.sha256(ink.convert("L").tobytes()).hexdigest(),
            "strokes_hash": hashlib.sha256(payload.encode()).hexdigest(), "mode": mode,
            "region": region, "locked_ids": sorted(locked), "stroke_updates": updates,
            "structural_edits": [], "metrics": metrics, "limitations": limitations,
            "status": "ready" if updates else "no_change"}
