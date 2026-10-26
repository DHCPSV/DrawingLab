"""Bounded similarity registration of model curves, with unchanged final gates.

This may repair a small uniform offset, rotation or scale error. It cannot turn
an eye into a jaw, repair a different pose, or nonlinearly bend a limb to fit.
All connected paths move together; original artist pixels never move.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
from PIL import Image

from .geometry import (_Index, _length, _resample, _stroke_tangents, _visible_fragments,
                       assess_geometry, select_continuations)


MAX_DISPLACEMENT = 6.0
MAX_ROTATION_DEGREES = 3.0
MIN_SCALE, MAX_SCALE = .96, 1.04
MAX_ITERATIONS = 8


def _apply(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:2, :2].T + matrix[:2, 2]


def _transform(geometry: dict, matrices: dict[str, np.ndarray]) -> dict:
    result = copy.deepcopy(geometry)
    for path in result["paths"]:
        matrix = matrices.get(path["id"])
        if matrix is None:
            continue
        points = _apply(np.asarray(path["points"], dtype=float), matrix)
        path["points"] = np.round(points, 6).tolist()
        path["length"] = round(_length(points), 6)
    return result


def _components(geometry: dict) -> list[list[str]]:
    """Join shared junctions, including tiny discarded centreline connectors."""
    paths = geometry.get("paths", [])
    parent = list(range(len(paths)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    cells: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = {}
    for i, path in enumerate(paths):
        for point in (path["points"][0], path["points"][-1]):
            point = np.asarray(point, dtype=float)
            cell = np.floor(point / 4).astype(int)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j, other in cells.get((cell[0] + dx, cell[1] + dy), []):
                        if np.linalg.norm(point - other) <= 4:
                            a, b = root(i), root(j)
                            if a != b:
                                parent[a] = b
            cells.setdefault(tuple(cell), []).append((i, point))
    groups: dict[int, list[str]] = {}
    for i, path in enumerate(paths):
        groups.setdefault(root(i), []).append(path["id"])
    return list(groups.values())


def _samples(ink: Image.Image, strokes: list[dict]):
    raster = np.asarray(ink.convert("L"))
    points, tangents, weights = [], [], []
    for stroke in strokes:
        if stroke.get("tool", "pen") != "pen":
            continue
        fragments = _visible_fragments(stroke, raster)
        sampled = [_resample(fragment, 3.0) for fragment in fragments if _length(fragment) >= 6]
        count = sum(len(fragment) for fragment in sampled)
        for fragment in sampled:
            points.extend(fragment)
            tangents.extend(_stroke_tangents(fragment))
            # Give small features a voice in the fit, not just the long outline.
            weights.extend([1.0 / max(1, count)] * len(fragment))
    return np.asarray(points), np.asarray(tangents), np.asarray(weights)


def _fit_similarity(source: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = weights / weights.sum()
    source_centre = np.sum(source * weights[:, None], axis=0)
    target_centre = np.sum(target * weights[:, None], axis=0)
    x, y = source - source_centre, target - target_centre
    denominator = float(np.sum(weights * np.sum(x * x, axis=1)))
    if denominator < 1e-6:
        return np.eye(3)
    real = float(np.sum(weights * np.sum(x * y, axis=1))) / denominator
    imaginary = float(np.sum(weights * (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]))) / denominator
    angle = math.atan2(imaginary, real)
    angle = float(np.clip(angle, -math.radians(MAX_ROTATION_DEGREES), math.radians(MAX_ROTATION_DEGREES)))
    scale = float(np.clip(math.hypot(real, imaginary), MIN_SCALE, MAX_SCALE))
    matrix = np.eye(3)
    matrix[:2, :2] = scale * np.asarray([[math.cos(angle), -math.sin(angle)],
                                         [math.sin(angle), math.cos(angle)]])
    matrix[:2, 2] = target_centre - matrix[:2, :2] @ source_centre
    return matrix


def _bounded(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """All points of a component obey the cap, not only its fitted prefix."""
    angle = math.atan2(matrix[1, 0], matrix[0, 0])
    scale = math.hypot(matrix[0, 0], matrix[1, 0])
    displacement = float(np.linalg.norm(_apply(points, matrix) - points, axis=1).max())
    fraction = min(1.0, MAX_DISPLACEMENT / max(displacement, 1e-8))
    if abs(angle) > math.radians(MAX_ROTATION_DEGREES):
        fraction = min(fraction, math.radians(MAX_ROTATION_DEGREES) / abs(angle))
    if scale < MIN_SCALE or scale > MAX_SCALE:
        fraction = min(fraction, .04 / abs(scale - 1))
    # A convex blend of these 2D similarity matrices remains a similarity.
    result = np.eye(3) + fraction * (matrix - np.eye(3))
    # The final numbers, rather than optimizer intent, must obey all limits.
    for _ in range(8):
        scale = math.hypot(result[0, 0], result[1, 0])
        rotation = abs(math.degrees(math.atan2(result[1, 0], result[0, 0])))
        movement = float(np.linalg.norm(_apply(points, result) - points, axis=1).max())
        if MIN_SCALE <= scale <= MAX_SCALE and rotation <= MAX_ROTATION_DEGREES + 1e-9 and movement <= MAX_DISPLACEMENT + 1e-9:
            break
        result = np.eye(3) + .98 * (result - np.eye(3))
    return result


def _icp(geometry: dict, real_points: np.ndarray, tangents: np.ndarray,
         weights: np.ndarray, groups: list[list[str]]):
    original = {p["id"]: np.asarray(p["points"], dtype=float) for p in geometry["paths"]}
    group_for = {path_id: i for i, group in enumerate(groups) for path_id in group}
    group_points = [np.vstack([original[path_id] for path_id in group]) for group in groups]
    matrices = [np.eye(3) for _ in groups]
    history = []
    for _ in range(MAX_ITERATIONS):
        mapping = {path_id: matrices[i] for i, group in enumerate(groups) for path_id in group}
        current = _transform(geometry, mapping)
        index = _Index(current)
        pairs: dict[int, list] = {}
        for point, tangent, weight in zip(real_points, tangents, weights):
            match = index.nearest(point)
            if match is None or match["distance"] > 9:
                continue
            candidate_tangent = index.curves[match["path_id"]].tangent(match["s"])
            if abs(float(tangent @ candidate_tangent)) < math.cos(math.radians(35)):
                continue
            pairs.setdefault(group_for[match["path_id"]], []).append((match["point"], point, weight, match["distance"]))
        moved = False
        for i, values in pairs.items():
            if len(values) < 4:
                continue
            source = np.asarray([v[0] for v in values])
            target = np.asarray([v[1] for v in values])
            residual = np.asarray([v[3] for v in values])
            pair_weights = np.asarray([v[2] for v in values]) * np.minimum(1.0, 3.0 / np.maximum(residual, .001))
            increment = _fit_similarity(source, target, pair_weights)
            proposed = _bounded(increment @ matrices[i], group_points[i])
            change = float(np.linalg.norm(_apply(group_points[i], proposed) - _apply(group_points[i], matrices[i]), axis=1).max())
            moved |= change > .03
            matrices[i] = proposed
        mapping = {path_id: matrices[i] for i, group in enumerate(groups) for path_id in group}
        current = _transform(geometry, mapping)
        history.append((current, [matrix.copy() for matrix in matrices]))
        if not moved:
            break
    return history


def align_geometry(ink: Image.Image, geometry: dict, strokes: list[dict],
                   focus: dict | None = None, length: float = 60) -> tuple[dict, dict]:
    """Return an adjustment only after the unchanged per-stroke gate passes.

    Registration uses actual visible stroke samples, never a synthetic prefix
    traced from the model's own output. Failed optimization returns the original
    graph unchanged and exposes why it was not applied.
    """
    started = time.perf_counter()
    before = assess_geometry(ink, geometry, strokes)
    before_hints = select_continuations(geometry, ink, strokes, focus, length) if before["accepted"] else []
    metadata = {"applied": False, "method": "bounded_similarity_icp", "max_displacement_px": 0.0,
                "before_accepted": before["accepted"], "before_distance_p95": before["distance_p95"],
                "after_accepted": before["accepted"], "after_distance_p95": before["distance_p95"],
                "before_suggestions": len(before_hints), "after_suggestions": len(before_hints),
                "attempts": 0, "semantic_alignment_verified": False}
    if before_hints or not strokes or not geometry.get("paths") or before.get("unrepresented_components"):
        metadata["reason"] = "已有合格续画，无需调整" if before_hints else "没有完整可配准笔画"
        metadata["seconds"] = round(time.perf_counter() - started, 3)
        return geometry, metadata
    real_points, tangents, weights = _samples(ink, strokes)
    if len(real_points) < 4:
        metadata["reason"] = "有效笔画过短，不能确定小幅配准"
        metadata["seconds"] = round(time.perf_counter() - started, 3)
        return geometry, metadata
    components = _components(geometry)
    modes = [("global", [[p["id"] for p in geometry["paths"]]])]
    if len(components) > 1:
        modes.append(("connected_components", components))
    best_failed = before
    for mode, groups in modes:
        for adjusted, matrices in _icp(geometry, real_points, tangents, weights, groups):
            metadata["attempts"] += 1
            assessment = assess_geometry(ink, adjusted, strokes)
            if assessment["distance_p95"] < best_failed["distance_p95"]:
                best_failed = assessment
            if not assessment["accepted"]:
                continue
            # The UI validates every point of every published matched curve.
            # Reject an out-of-canvas match; clipping would be a deformation.
            published = set(assessment["matched_path_ids"])
            width, height = float(adjusted.get("width", ink.width)), float(adjusted.get("height", ink.height))
            bounded_paths = True
            for path in adjusted["paths"]:
                if path["id"] not in published:
                    continue
                points = np.asarray(path["points"], dtype=float)
                if (not np.isfinite(points).all() or (points < 0).any()
                        or (points[:, 0] > width).any() or (points[:, 1] > height).any()):
                    bounded_paths = False
                    break
            if not bounded_paths:
                continue
            hints = select_continuations(adjusted, ink, strokes, focus, length)
            # Do not apply a cosmetic improvement that still offers no useful
            # suffix. Acceptance is not sufficient without the tip/direction gate.
            if not hints:
                continue
            transforms = []
            original_paths = {p["id"]: p for p in geometry["paths"]}
            for group, matrix in zip(groups, matrices):
                points = np.vstack([original_paths[path_id]["points"] for path_id in group])
                movement = float(np.linalg.norm(_apply(points, matrix) - points, axis=1).max())
                if movement < .01:
                    continue
                transforms.append({"path_ids": group, "matrix": np.round(matrix, 8).tolist(),
                                   "max_displacement_px": round(movement, 5),
                                   "rotation_degrees": round(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), 5),
                                   "scale": round(math.hypot(matrix[0, 0], matrix[1, 0]), 7)})
            metadata.update({"applied": True, "mode": mode, "transforms": transforms,
                             "max_displacement_px": max((t["max_displacement_px"] for t in transforms), default=0),
                             "after_accepted": True, "after_distance_p95": assessment["distance_p95"],
                             "after_suggestions": len(hints), "reason": "小幅配准通过逐笔和笔尖续画检查"})
            metadata["seconds"] = round(time.perf_counter() - started, 3)
            return adjusted, metadata
    metadata.update({"reason": "小幅配准仍不满足逐笔或笔尖约束，保留原候选并拒绝不合格建议",
                     "best_attempt_distance_p95": best_failed["distance_p95"],
                     "seconds": round(time.perf_counter() - started, 3)})
    return geometry, metadata
