"""Conservative, vector-aware continuation of model-proposed line art.

The model owns no artist pixels. A candidate must match each visible stroke,
and only the short suffix of an ordered, direction-compatible curve may be
suggested. These are geometric checks, not claims of anatomical understanding.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
from PIL import Image


MATCH_DISTANCE = 3.0
TIP_DISTANCE = 2.5
TIP_ANGLE = 25.0
MAX_PATHS = 2048
MAX_PATH_POINTS = 100_000
SAMPLE_STEP = 1.5


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning, keeping stroke centres instead of both ink edges."""
    work = np.pad(mask.astype(np.uint8), 1)
    for _ in range(160):
        changed = False
        for step in (0, 1):
            p = [work[:-2, 1:-1], work[:-2, 2:], work[1:-1, 2:],
                 work[2:, 2:], work[2:, 1:-1], work[2:, :-2],
                 work[1:-1, :-2], work[:-2, :-2]]
            count = sum(p)
            transitions = sum(((p[i] == 0) & (p[(i + 1) % 8] == 1)).astype(np.uint8)
                              for i in range(8))
            if step == 0:
                clear = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                clear = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            remove = ((work[1:-1, 1:-1] == 1) & (count >= 2) & (count <= 6)
                      & (transitions == 1) & clear)
            if remove.any():
                work[1:-1, 1:-1][remove] = 0
                changed = True
        if not changed:
            break
    return work[1:-1, 1:-1].astype(bool)


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _points(stroke: dict) -> np.ndarray:
    points = stroke.get("points", [])
    if not points:
        return np.empty((0, 2), dtype=float)
    if isinstance(points[0], dict):
        return np.asarray([[p["x"], p["y"]] for p in points], dtype=float)
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _resample(points: np.ndarray, step: float = SAMPLE_STEP) -> np.ndarray:
    if len(points) < 2:
        return points.copy()
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.r_[True, distances > 1e-7]]
    if len(points) < 2:
        return points.copy()
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    query = np.linspace(0.0, arc[-1], max(2, int(math.ceil(arc[-1] / step)) + 1))
    return np.column_stack((np.interp(query, arc, points[:, 0]),
                            np.interp(query, arc, points[:, 1])))


def build_geometry(lines: Image.Image) -> dict:
    """Return a bounded JSON graph of centreline paths, split at junctions.

    Each path is a complete graph edge, not just the visible future suffix.
    Keeping its prefix lets the browser verify and advance the same match.
    """
    binary = np.asarray(lines.convert("L")) < 185
    height, width = binary.shape
    density = float(binary.mean())
    result = {"version": 1, "width": width, "height": height,
              "density": round(density, 6), "paths": [], "truncated": False}
    if not binary.any() or density >= .32:
        return result
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 4
    skeleton = _thin(keep[labels])
    ys, xs = np.where(skeleton)
    pixels = set((ys * width + xs).tolist())
    graph: dict[int, list[int]] = {}
    for pixel in sorted(pixels):
        y, x = divmod(pixel, width)
        neighbours = []
        for dy, dx in ((-1, 0), (0, 1), (1, 0), (0, -1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)):
            ny, nx = y + dy, x + dx
            other = ny * width + nx
            if not (0 <= ny < height and 0 <= nx < width) or other not in pixels:
                continue
            # Do not create triangular graph junctions at ordinary pixel elbows.
            if dy and dx and (y * width + nx in pixels or ny * width + x in pixels):
                continue
            neighbours.append(other)
        graph[pixel] = neighbours
    visited: set[tuple[int, int]] = set()
    raw_paths: list[tuple[list[int], bool]] = []

    def walk(start: int, following: int):
        path = [start]
        previous, current = start, following
        while True:
            visited.add((min(previous, current), max(previous, current)))
            path.append(current)
            if current == start:
                raw_paths.append((path, True))
                return
            if len(graph[current]) != 2:
                raw_paths.append((path, False))
                return
            next_pixel = graph[current][0] if graph[current][1] == previous else graph[current][1]
            if (min(current, next_pixel), max(current, next_pixel)) in visited:
                raw_paths.append((path, False))
                return
            previous, current = current, next_pixel

    for start, neighbours in graph.items():
        if len(neighbours) != 2:
            for following in neighbours:
                if (min(start, following), max(start, following)) not in visited:
                    walk(start, following)
    # Components with no endpoint or junction are closed curves.
    for start, neighbours in graph.items():
        for following in neighbours:
            if (min(start, following), max(start, following)) not in visited:
                walk(start, following)
    paths = []
    for raw, closed in raw_paths:
        # Tiny edges inside a cluster of junction pixels carry real topology.
        # Dropping them can split an otherwise continuous head outline into a
        # false gap. They are retained as connectors, never long future hints.
        if len(raw) < 2:
            continue
        points = np.asarray([[pixel % width, pixel // width] for pixel in raw], dtype=np.float32)
        simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2), .45, False).reshape(-1, 2)
        if closed and not np.array_equal(simplified[0], simplified[-1]):
            simplified = np.vstack((simplified, simplified[0]))
        if len(simplified) < 2:
            continue
        paths.append({"id": f"p{len(paths)}", "points": simplified.astype(float).tolist(),
                      "closed": bool(closed), "length": round(_length(simplified), 3),
                      "connector": _length(simplified) < 4,
                      "start_degree": len(graph[raw[0]]), "end_degree": len(graph[raw[-1]])})
    # Prioritise meaningful curves if an excessively fragmented model output hits
    # the transport cap. The cap is recorded; silently losing anchors is unsafe.
    if len(paths) > MAX_PATHS or sum(len(p["points"]) for p in paths) > MAX_PATH_POINTS:
        result["truncated"] = True
        paths.sort(key=lambda path: path["length"], reverse=True)
        bounded, total = [], 0
        for path in paths[:MAX_PATHS]:
            if total + len(path["points"]) > MAX_PATH_POINTS:
                break
            bounded.append(path)
            total += len(path["points"])
        paths = bounded
    result["paths"] = paths
    result["point_count"] = sum(len(path["points"]) for path in paths)
    return result


@dataclass
class _Curve:
    id: str
    points: np.ndarray
    closed: bool

    def __post_init__(self):
        lengths = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self.arc = np.r_[0.0, np.cumsum(lengths)]
        self.length = float(self.arc[-1])

    def point(self, arc):
        query = np.asarray(arc)
        return np.stack((np.interp(query, self.arc, self.points[:, 0]),
                         np.interp(query, self.arc, self.points[:, 1])), axis=-1)

    def tangent(self, arc: float, radius: float = 5.0):
        low, high = max(0.0, arc - radius), min(self.length, arc + radius)
        if self.closed:
            low, high = (arc - radius) % self.length, (arc + radius) % self.length
        delta = self.point(high) - self.point(low)
        return delta / max(float(np.linalg.norm(delta)), 1e-8)


class _Index:
    """Small spatial index for precise segment projections; no extra runtime."""
    CELL = 16
    SEARCH = 8

    def __init__(self, geometry: dict):
        self.curves = {p["id"]: _Curve(p["id"], np.asarray(p["points"], dtype=float),
                                       bool(p.get("closed")))
                       for p in geometry.get("paths", []) if len(p.get("points", [])) >= 2}
        self.segments = []
        self.cells: dict[tuple[int, int], list[int]] = {}
        for curve in self.curves.values():
            for i, (a, b) in enumerate(zip(curve.points[:-1], curve.points[1:])):
                delta = b - a
                length2 = float(delta @ delta)
                if length2 < 1e-8:
                    continue
                index = len(self.segments)
                self.segments.append((curve, a, delta, length2, float(curve.arc[i])))
                low = np.floor((np.minimum(a, b) - self.SEARCH) / self.CELL).astype(int)
                high = np.floor((np.maximum(a, b) + self.SEARCH) / self.CELL).astype(int)
                for y in range(low[1], high[1] + 1):
                    for x in range(low[0], high[0] + 1):
                        self.cells.setdefault((x, y), []).append(index)

    def nearest(self, point: np.ndarray, allowed: set[str] | None = None):
        cell = tuple(np.floor(point / self.CELL).astype(int))
        best = None
        for segment_index in self.cells.get(cell, []):
            curve, a, delta, length2, arc = self.segments[segment_index]
            if allowed is not None and curve.id not in allowed:
                continue
            fraction = float(np.clip((point - a) @ delta / length2, 0.0, 1.0))
            position = a + fraction * delta
            distance = float(np.linalg.norm(position - point))
            if best is None or distance < best["distance"]:
                best = {"path_id": curve.id, "s": arc + fraction * math.sqrt(length2),
                        "point": position, "distance": distance}
        return best

    def nearby_paths(self, point: np.ndarray) -> set[str]:
        cell = tuple(np.floor(point / self.CELL).astype(int))
        return {self.segments[i][0].id for i in self.cells.get(cell, [])}


def _visible_fragments(stroke: dict, ink: np.ndarray) -> list[np.ndarray]:
    """PNG is authoritative after erasing; hidden old vectors are not anchors."""
    points = _resample(_points(stroke))
    if not len(points):
        return []
    height, width = ink.shape
    coordinates = np.rint(points).astype(int)
    valid = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
             & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height))
    visible = np.zeros(len(points), dtype=bool)
    visible[valid] = ink[coordinates[valid, 1], coordinates[valid, 0]] < 220
    fragments, start = [], None
    for i, exists in enumerate(np.r_[visible, False]):
        if exists and start is None:
            start = i
        elif not exists and start is not None:
            fragments.append(points[start:i])
            start = None
    return fragments


def _stroke_tangents(points: np.ndarray) -> np.ndarray:
    result = np.zeros_like(points)
    for i in range(len(points)):
        delta = points[min(len(points) - 1, i + 3)] - points[max(0, i - 3)]
        norm = float(np.linalg.norm(delta))
        if norm > 1e-8:
            result[i] = delta / norm
    return result


def _fragment_match(points: np.ndarray, index: _Index) -> dict:
    fragment_length = _length(points)
    projections = [index.nearest(point) for point in points]
    errors = np.asarray([p["distance"] if p is not None else 99.0 for p in projections])
    tangents = _stroke_tangents(points)
    cosines = []
    runs: list[dict] = []
    last_id = None
    for i, projection in enumerate(projections):
        if projection is None or projection["distance"] > MATCH_DISTANCE:
            last_id = None
            continue
        path_id = projection["path_id"]
        if path_id != last_id:
            runs.append({"path_id": path_id, "start_index": i, "end_index": i,
                         "arcs": [projection["s"]]})
        else:
            runs[-1]["end_index"] = i
            runs[-1]["arcs"].append(projection["s"])
        last_id = path_id
        if fragment_length >= 6:
            curve_tangent = index.curves[path_id].tangent(projection["s"])
            cosines.append(abs(float(tangents[i] @ curve_tangent)))
    direction_fraction = float(np.mean(np.asarray(cosines) >= math.cos(math.radians(40)))) if cosines else 1.0
    ordered = True
    for run in runs:
        arcs = np.asarray(run.pop("arcs"))
        curve = index.curves[run["path_id"]]
        delta = np.diff(arcs)
        if curve.closed:
            delta = (delta + curve.length / 2) % curve.length - curve.length / 2
        meaningful = delta[np.abs(delta) > .3]
        direction = 1 if meaningful.sum() >= 0 else -1
        # A matching prefix must move along its candidate rather than hopping
        # backwards between adjacent folds or jumping ahead across a curve.
        if len(meaningful) and (np.mean(meaningful * direction >= 0) < .9
                                or np.max(np.abs(meaningful)) > SAMPLE_STEP * 4):
            ordered = False
        run["start_s"] = round(float(arcs[0]), 3)
        run["end_s"] = round(float(arcs[-1]), 3)
        run["direction"] = direction
    # Switching paths is only allowed through nearby graph endpoints/junctions.
    for before, after in zip(runs[:-1], runs[1:]):
        if before["path_id"] == after["path_id"]:
            continue
        a, b = index.curves[before["path_id"]], index.curves[after["path_id"]]
        a_endpoint = min(before["end_s"], a.length - before["end_s"]) <= 5
        b_endpoint = min(after["start_s"], b.length - after["start_s"]) <= 5
        adjacent = np.linalg.norm(a.point(before["end_s"]) - b.point(after["start_s"])) <= 6
        if not (a_endpoint and b_endpoint and adjacent):
            ordered = False
    p95 = float(np.quantile(errors, .95))
    matched = float(np.mean(errors <= MATCH_DISTANCE))
    accepted = bool(p95 <= MATCH_DISTANCE and float(errors.max()) <= 6
                    and matched >= .95 and direction_fraction >= .8 and ordered)
    return {"accepted": accepted, "length": round(fragment_length, 2),
            "distance_p95": round(p95, 3), "distance_max": round(float(errors.max()), 3),
            "matched": round(matched, 4), "direction_fraction": round(direction_fraction, 4),
            "ordered": ordered, "path_ids": list(dict.fromkeys(run["path_id"] for run in runs)),
            "runs": runs}


def _stroke_coverage(strokes: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Account for all transported pen vectors, including now-erased prefixes."""
    coverage = np.zeros(shape, dtype=np.uint8)
    for stroke in strokes:
        if stroke.get("tool", "pen") != "pen":
            continue
        points = _points(stroke)
        if not len(points):
            continue
        pixels = np.rint(points).astype(np.int32)
        if len(pixels) == 1:
            cv2.circle(coverage, tuple(pixels[0]), 2, 1, -1)
        else:
            cv2.polylines(coverage, [pixels.reshape(-1, 1, 2)], False, 1, 3)
    return cv2.dilate(coverage, np.ones((3, 3), np.uint8)).astype(bool)


def _raster_constraints(ink: Image.Image) -> list[dict]:
    traced = build_geometry(ink)
    strokes = [{"id": f"raster:{path['id']}", "tool": "pen", "points": path["points"]}
               for path in traced["paths"]]
    # Graph edges intentionally omit isolated dots; they remain position
    # constraints in old raster-only input and must not disappear from the gate.
    binary = (np.asarray(ink.convert("L")) < 200).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(binary, 8)
    coverage = _stroke_coverage(strokes, binary.shape)
    touched = set(np.unique(labels[coverage & (binary > 0)]).tolist())
    for label in range(1, count):
        if label not in touched and stats[label, cv2.CC_STAT_AREA] >= 2:
            strokes.append({"id": f"raster:dot{label}", "tool": "pen",
                            "points": [[float(centres[label, 0]), float(centres[label, 1])]]})
    return strokes


def assess_geometry(ink: Image.Image, geometry: dict, strokes: list[dict] | None = None) -> dict:
    """Validate every visible real stroke independently; small features can veto.

    Older raster-only requests have an explicit raster-centreline fallback. Its
    fragments are geometric constraints, never invented artist stroke IDs.
    """
    source = "strokes"
    real_ink = np.asarray(ink.convert("L"))
    unrepresented_pixels = 0
    unrepresented_components = 0
    if not strokes:
        source = "raster"
        strokes = _raster_constraints(ink)
    else:
        # A clipped or stale vector payload must not silently remove a small
        # eye/mouth constraint that is still present in the authoritative PNG.
        residual = ((real_ink < 200) & ~_stroke_coverage(strokes, real_ink.shape)).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(residual, 8)
        sizes = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([], dtype=int)
        unrepresented_pixels = int(sizes[sizes >= 3].sum())
        unrepresented_components = int(np.count_nonzero(sizes >= 3))
    index = _Index(geometry)
    per_stroke = []
    matched_path_ids = set()
    path_owners: dict[str, list[str]] = {}
    for number, stroke in enumerate(strokes):
        if stroke.get("tool", "pen") != "pen":
            continue
        fragments = _visible_fragments(stroke, real_ink)
        # Ignore subpixel replay remnants left by erasing, but retain actual
        # single-point pen marks as position constraints (never continuations).
        fragments = [p for p in fragments if len(p) >= 2 or len(_points(stroke)) == 1]
        if not fragments:
            continue
        results = [_fragment_match(points, index) for points in fragments]
        accepted = all(result["accepted"] for result in results)
        ids = list(dict.fromkeys(path_id for result in results for path_id in result["path_ids"]))
        stroke_id = str(stroke.get("id", f"stroke:{number}"))
        if accepted:
            matched_path_ids.update(ids)
            for path_id in ids:
                path_owners.setdefault(path_id, []).append(stroke_id)
        per_stroke.append({"id": stroke_id, "accepted": accepted,
                           "path_ids": ids, "fragments": results})
    accepted = bool(index.curves and per_stroke and all(s["accepted"] for s in per_stroke)
                    and not geometry.get("truncated", False) and not unrepresented_components)
    if accepted:
        reason = "逐笔几何检查通过；只显示与笔尖连接的短路径"
    elif not index.curves:
        reason = "候选没有可用的线条中心路径"
    elif not per_stroke:
        reason = "没有可验证的可见笔画"
    elif unrepresented_components:
        reason = "笔画记录未覆盖当前画布，已隐藏建议；请重新提交完整笔画"
    elif geometry.get("truncated", False):
        reason = "候选过于碎乱，无法完整检查已有笔画"
    else:
        reason = "候选与已有笔画的位置或方向不一致，已隐藏；可以换一个方案"
    fragment_results = [fragment for stroke in per_stroke for fragment in stroke["fragments"]]
    return {"accepted": accepted, "reason": reason, "source": source,
            "density": geometry.get("density", 0.0), "per_stroke": per_stroke,
            "matched_path_ids": sorted(matched_path_ids),
            "path_owners": path_owners,
            "unrepresented_pixels": unrepresented_pixels,
            "unrepresented_components": unrepresented_components,
            "stroke_count": len(per_stroke),
            "failed_strokes": sum(not stroke["accepted"] for stroke in per_stroke),
            "matched": round(float(np.mean([f["matched"] for f in fragment_results])), 4) if fragment_results else 0.0,
            "distance_p95": max((f["distance_p95"] for f in fragment_results), default=99.0)}


def select_continuations(geometry: dict, ink: Image.Image, strokes: list[dict],
                         focus: dict | None = None, length: float = 60) -> list[dict]:
    """Select a short suffix from the visible last pen tip, never a nearby limb.

    No previous direction can be inferred from an isolated dot or a closed
    outline. No new independent feature is guessed here. A tiny offset may be
    blended out only after ordered tail correspondence is established.
    """
    if not strokes:
        return []
    last = strokes[-1]
    if last.get("tool", "pen") != "pen":
        return []
    original = _points(last)
    if len(original) < 2 or np.linalg.norm(original[-1] - original[0]) < 3:
        return []
    real_ink = np.asarray(ink.convert("L"))
    fragments = _visible_fragments(last, real_ink)
    if not fragments or np.linalg.norm(fragments[-1][-1] - original[-1]) > .1:
        return []
    tail = fragments[-1]
    if _length(tail) < 8:
        return []
    # Only the last 18 px determine continuation direction. Whole-stroke
    # correspondence has already been checked separately by assess_geometry.
    tail = _resample(tail)
    remaining = np.cumsum(np.r_[0.0, np.linalg.norm(np.diff(tail[::-1], axis=0), axis=1)])
    tail = tail[max(0, len(tail) - int(np.searchsorted(remaining, 18)) - 1):]
    index = _Index(geometry)
    tip = original[-1]
    delta = tip - tail[max(0, len(tail) - 7)]
    norm = float(np.linalg.norm(delta))
    if norm < 5:
        return []
    direction_vector = delta / norm
    options = []
    for path_id in index.nearby_paths(tip):
        projection = index.nearest(tip, {path_id})
        if projection is None or projection["distance"] > TIP_DISTANCE:
            continue
        curve = index.curves[path_id]
        # A complete tail must match this particular curve, not unrelated nearby
        # fragments. Two equally plausible parallel paths are also ambiguous.
        matched = [index.nearest(point, {path_id}) for point in tail]
        if any(p is None or p["distance"] > TIP_DISTANCE for p in matched):
            continue
        dot = float(direction_vector @ curve.tangent(projection["s"]))
        if abs(dot) < math.cos(math.radians(TIP_ANGLE)):
            continue
        direction = 1 if dot >= 0 else -1
        arcs = np.asarray([p["s"] for p in matched])
        steps = np.diff(arcs)
        if curve.closed:
            steps = (steps + curve.length / 2) % curve.length - curve.length / 2
        if np.any(steps * direction < -.4) or np.any(np.abs(steps) > SAMPLE_STEP * 3):
            continue
        score = float(np.mean([p["distance"] for p in matched]))
        options.append((score, curve, projection, direction))
    options.sort(key=lambda option: option[0])
    if not options or (len(options) > 1 and options[1][0] - options[0][0] <= .3):
        return []
    _, curve, projection, direction = options[0]
    path_id = curve.id
    start_s = float(projection["s"])
    limit = min(max(float(length), 12.0), 120.0)
    available = curve.length - start_s if direction > 0 else start_s
    # Paths terminate at graph junctions. Do not choose an unseen branch.
    travel = min(limit, available)
    if travel < 5:
        return []
    offsets = np.linspace(0.0, travel, max(3, int(math.ceil(travel / 1.5)) + 1))
    future = curve.point(start_s + direction * offsets)
    # C0 anchoring is allowed only inside TIP_DISTANCE with a compatible tail.
    # It decays over 12 px, so no distant synthetic connector is ever invented.
    correction = tip - projection["point"]
    future += np.maximum(0.0, 1.0 - offsets / 12.0)[:, None] * correction
    coordinates = np.rint(future).astype(int)
    height, width = real_ink.shape
    free = ((coordinates[:, 0] >= 0) & (coordinates[:, 0] < width)
            & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < height))
    occupied = np.ones(len(future), dtype=bool)
    occupied[free] = real_ink[coordinates[free, 1], coordinates[free, 0]] < 220
    # The initial tip itself is ink. Once the suffix reaches unpainted paper,
    # stop before any other ink instead of drawing through an existing part.
    departure = next((i for i, value in enumerate(occupied) if not value), None)
    if departure is None or offsets[departure] > 6:
        return []
    stop = next((i for i in range(departure, len(future)) if occupied[i]), len(future))
    if stop - departure < 3:
        return []
    future = future[:stop]
    return [{"path_id": path_id, "points": np.round(future, 3).tolist(),
             "kind": "continuation", "source_stroke_id": str(last.get("id", "")),
             "direction": direction, "start_s": round(start_s, 3),
             "end_s": round(start_s + direction * float(offsets[stop - 1]), 3),
             "anchor": np.round(tip, 3).tolist(), "gap": round(projection["distance"], 3)}]
