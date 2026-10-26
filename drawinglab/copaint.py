"""Whole-sketch completion. User ink is authoritative; every request is inferred.

The statistics below describe visible marks, not a learned personal style model.
Preview quality checks are diagnostics, not claims of semantic correspondence.
"""
from __future__ import annotations

import json
import secrets
import time

import cv2
import numpy as np
from PIL import Image

from .imaging import encode_png, extract_lines
from .jobs import JobManager


def style_profile(ink: Image.Image, strokes: list) -> dict:
    pixels = np.asarray(ink.convert("L")) < 200
    y, x = np.where(pixels)
    if not len(x):
        return {"source": "visible_user_ink", "description": "", "confidence": "insufficient"}
    area = max(1, (int(x.max()) - int(x.min()) + 1) * (int(y.max()) - int(y.min()) + 1))
    density = float(pixels.sum() / area)
    lengths, corners, segments = [], 0, 0
    for stroke in strokes:
        if stroke.get("tool") != "pen" or stroke.get("provenance") == "assisted":
            continue
        points = np.array([[p["x"], p["y"]] for p in stroke.get("points", [])], dtype=np.float32)
        if len(points) < 3:
            continue
        # Split erased portions. Old erased lines must never become style evidence.
        xy = np.rint(points).astype(int)
        xy[:, 0] = np.clip(xy[:, 0], 0, pixels.shape[1]-1)
        xy[:, 1] = np.clip(xy[:, 1], 0, pixels.shape[0]-1)
        visible = pixels[xy[:, 1], xy[:, 0]]
        for run in np.split(points, np.where(np.diff(visible.astype(int)) != 0)[0] + 1):
            if len(run) < 3:
                continue
            px, py = np.clip(np.rint(run[0]).astype(int), 0, pixels.shape[0]-1)
            if not pixels[py, px]:
                continue
            length = float(np.linalg.norm(np.diff(run, axis=0), axis=1).sum())
            if length < 12:
                continue
            lengths.append(length)
            simplified = cv2.approxPolyDP(run.reshape(-1, 1, 2), 2.5, False).reshape(-1, 2)
            tangent = np.diff(simplified, axis=0)
            if len(tangent) > 1:
                cosine = (tangent[:-1] * tangent[1:]).sum(1) / np.maximum(
                    np.linalg.norm(tangent[:-1], axis=1) * np.linalg.norm(tangent[1:], axis=1), 1e-6)
                angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
                corners += int((angles > 42).sum())
                segments += len(angles)
    terms = ["black pen line drawing", "white background", "same drawing style"]
    # Sparse early input alone does not establish a preference for minimalism.
    if len(lengths) >= 4:
        if density < .065:
            terms.append("spare linework, few interior details")
        if segments >= 6:
            terms.append("angular contours" if corners / segments > .28 else "flowing contours")
    return {"source": "visible_user_ink", "method": "stroke_statistics_not_trained_style_encoder",
            "description": ", ".join(terms), "visible_strokes": len(lengths),
            "ink_density_in_bounds": round(density, 4),
            "corner_fraction": round(corners / max(1, segments), 3),
            "confidence": "limited" if len(lengths) >= 4 else "insufficient"}


def update_plan(old_ink: Image.Image | None, ink: Image.Image,
                previous: Image.Image | None, *, eraser=False) -> tuple[Image.Image | None, dict]:
    """Enlarge conflicting edits to connected prior structure, never lock it in."""
    if old_ink is None or previous is None:
        return None, {"kind": "initial", "edit_fraction": 1.0, "conflict_fraction": 0.0}
    old, new = np.asarray(old_ink.convert("L")), np.asarray(ink.convert("L"))
    changed = np.abs(old.astype(np.int16) - new.astype(np.int16)) > 30
    added = changed & (new < 200)
    deleted = changed & (old < 200) & (new >= 200)
    y, x = np.where(changed)
    if not len(x):
        return Image.new("L", ink.size, 255), {"kind": "refresh", "edit_fraction": 1.0, "conflict_fraction": 0.0}
    prior = np.asarray(extract_lines(previous)) < 185
    distances = cv2.distanceTransform((~prior).astype(np.uint8), cv2.DIST_L2, 5)
    conflict = float((distances[added] > 9).mean()) if added.any() else 0.0
    correction = eraser or bool(deleted.any()) or conflict > .25
    mask = np.zeros(new.shape, dtype=np.uint8)
    margin = 100 if correction else 64
    left, right = max(0, int(x.min())-margin), min(ink.width, int(x.max())+margin+1)
    top, bottom = max(0, int(y.min())-margin), min(ink.height, int(y.max())+margin+1)
    mask[top:bottom, left:right] = 255
    if correction:
        connected = cv2.dilate(prior.astype(np.uint8), np.ones((13, 13), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
        affected = np.unique(labels[mask > 0])
        for label in affected:
            if not label or stats[label, cv2.CC_STAT_AREA] < 20:
                continue
            lx, ly, width, height = stats[label, :4]
            mask[max(0, ly-48):min(ink.height, ly+height+48),
                 max(0, lx-48):min(ink.width, lx+width+48)] = 255
    if (mask > 0).mean() > .7:
        mask[:] = 255
    return Image.fromarray(mask, "L"), {
        "kind": "correction" if correction else "addition",
        "edit_fraction": round(float((mask > 0).mean()), 4),
        "conflict_fraction": round(conflict, 4), "deleted_pixels": int(deleted.sum()),
        "added_pixels": int(added.sum()), "scope": "connected_geometry_not_semantic_segmentation"}


def completion_preview(candidate: Image.Image, ink: Image.Image) -> tuple[Image.Image, Image.Image, dict]:
    """Publish actual generated marks only, with exact real-ink overlap removed."""
    lines = extract_lines(candidate)
    actual = np.asarray(ink.convert("L")) < 220
    strength = 255 - np.asarray(lines)
    proposed = strength > 70
    density = float(proposed.mean())
    distance = cv2.distanceTransform((~proposed).astype(np.uint8), cv2.DIST_L2, 5)
    errors = distance[actual]
    p95 = float(np.quantile(errors, .95)) if len(errors) else None
    # Diagnostics use the raw candidate, before hiding overlap. This must not be
    # called a semantic gate: a nearby different facial feature can still match.
    occupied = cv2.dilate(actual.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    alpha = strength.copy()
    alpha[occupied] = 0
    novel = int((alpha > 70).sum())
    accepted = .0001 < density < .32 and novel >= 20
    outcome = "completion" if accepted else ("noop" if .0001 < density < .32 else "invalid")
    rgba = np.zeros((*actual.shape, 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha if accepted else 0
    return lines, Image.fromarray(rgba, "RGBA"), {
        "accepted": bool(accepted), "preview_kind": "whole_candidate",
        "outcome": outcome,
        "novel_pixels": novel, "density": round(density, 5),
        "input_distance_p95": round(p95, 2) if p95 is not None else None,
        "nearby_input_fraction": round(float((errors <= 6).mean()), 4) if len(errors) else None,
        "semantic_alignment_verified": False, "personal_style_learned": False,
        "alignment_warning": bool(p95 is not None and p95 > 6),
        "original_ink_modified": False,
    }


class CopaintJobManager(JobManager):
    """V3 scheduler: latest pending wins; no generated image is user ink."""

    def invalidate(self, identifier, revision, intent_revision, reset):
        with self.condition:
            s = self.sessions[identifier]
            if revision < s.revision or intent_revision < s.intent_revision:
                return
            super().invalidate(identifier, revision, intent_revision, reset)
            s.correction = None
            s.raw_reference = None

    def submit(self, identifier: str, request: dict) -> str:
        with self.condition:
            s = self.sessions[identifier]
            # Stable seed reduces arbitrary changes; an explicit variation gets
            # a new seed. Never use generated strokes as fresh style examples.
            seed = request.get("seed")
            if seed is None:
                seed = (getattr(s, "copaint_seed", None)
                        if not request.get("force") else None)
            s.copaint_seed = seed if seed is not None else secrets.randbelow(2**31)
            job = super().submit(identifier, dict(request, seed=s.copaint_seed))
            s.ghost = None
            s.correction = None
            s.task = request.get("task", "completion")
            s.raw_reference = None
            return job

    def result(self, identifier: str) -> dict:
        with self.condition:
            s = self.sessions[identifier]
            s.touched = time.monotonic()
            return {"serial": s.serial, "status": s.status, "revision": s.revision,
                    "intent_revision": s.intent_revision, "candidate_id": s.candidate_id,
                    "source_revision": getattr(s, "source_revision", None),
                    "preview_revision": getattr(s, "source_revision", None),
                    "ghost": s.ghost if s.status == "ready" else None,
                    "task": getattr(s, "task", "completion"),
                    "correction": getattr(s, "correction", None) if s.status == "ready" else None,
                    "raw_reference": getattr(s, "raw_reference", None) if s.status in ("ready", "noop", "rejected") else None,
                    "paths_version": 1, "paths": [], "suggestions": [],
                    "message": s.message, "metrics": s.metrics, "mode": "copaint-v4"}

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.queue)
                if self.stopped:
                    return
                identifier = self.queue.popleft()
                s = self.sessions.get(identifier)
                if not s or s.pending is None:
                    continue
                request, s.pending = s.pending, None
                self.active = identifier
                task = request.get("task", "completion")
                s.status, s.message = "running", ("正在根据当前画稿补全；可以继续画" if task == "completion"
                                                  else "正在准备修稿建议；原稿保持不变，可以继续画")
                s.serial += 1
                previous, old_ink = s.candidate, s.candidate_ink
            started = time.monotonic()
            try:
                style = style_profile(request["ink"], request.get("strokes", []))
                mask, plan = update_plan(old_ink, request["ink"], previous, eraser=request.get("eraser", False))
                # Content comes first so long descriptions do not crowd it out
                # of the current SDXL encoder. This is an explicit baseline.
                proposal = None
                cancel = lambda: self.stopped or s.epoch != request["epoch"]
                if task == "cleanup":
                    candidate = request["ink"].copy()
                    metrics = {"generated_by_ai": False, "mode": "geometric_cleanup", "inference_seconds": 0}
                elif task == "correction":
                    candidate, metrics = self.engine.generate_correction(
                        ink=request["ink"], prompt=request["prompt"].strip(), seed=request["seed"],
                        cancel=cancel, region=request.get("region"))
                else:
                    prompt = request["prompt"].strip() + ", " + style["description"]
                    candidate, metrics = self.engine.generate(
                        ink=request["ink"], prompt=prompt, previous=previous,
                        mask=mask, seed=request["seed"], cancel=cancel,
                    )
                if candidate.size != request["ink"].size:
                    raise RuntimeError("模型返回了不匹配的画布尺寸")
                lines, preview, check = completion_preview(candidate, request["ink"])
                if task != "completion":
                    if cancel():
                        continue
                    from .correction import propose_correction
                    proposal = propose_correction(
                        ink=request["ink"], candidate=candidate if task == "correction" else None,
                        strokes=request.get("strokes", []), region=request.get("region"),
                        locked_ids=request.get("locked_ids", []), source_revision=request["revision"],
                        intent_revision=request["intent_revision"], mode="ai" if task == "correction" else "cleanup")
                    check.update(accepted=bool(proposal["stroke_updates"]),
                                 outcome="correction" if proposal["stroke_updates"] else "noop",
                                 preview_kind="vector_correction", correction=proposal["metrics"])
                metrics = dict(metrics, **check, cached=False, source_revision=request["revision"],
                               style=style, update=plan, elapsed_seconds=round(time.monotonic()-started, 2))
                with self.condition:
                    if self.stopped or s.epoch != request["epoch"] or s.latest is None:
                        continue
                    if s.latest["job_id"] != request["job_id"]:
                        # A completed old revision can be a soft warm start, but
                        # is never relabelled/published as if it saw newer ink.
                        if check["accepted"] and task == "completion":
                            s.candidate, s.candidate_ink = candidate, request["ink"].copy()
                        s.status, s.message = "queued", "已有新笔画，正在更新补全"
                        s.serial += 1
                        continue
                    s.metrics = metrics
                    s.source_revision = request["revision"]
                    # Keep the actual complete model output, independent of
                    # vector acceptance and the new-ink-only ghost mask.
                    s.raw_reference = encode_png(candidate) if task != "cleanup" else None
                    s.ghost = encode_png(preview) if check["accepted"] and task == "completion" else None
                    s.correction = proposal if check["accepted"] else None
                    if check["accepted"] and task != "completion":
                        s.status = "ready"
                        pm = proposal["metrics"]
                        ai_count = pm.get("ai_matched_strokes", 0)
                        label = "AI 修形建议已准备好" if ai_count else "线条整理已准备好"
                        if task == "correction" and not ai_count:
                            label = "AI 目标暂未匹配到可用修形，已提供轻度线条整理"
                        s.message = label + "；拖动幅度预览，应用后才会修改作品"
                    elif check["accepted"]:
                        s.candidate, s.lines = candidate, lines
                        s.candidate_ink = request["ink"].copy()
                        s.candidate_id = secrets.token_hex(8)
                        s.status = "ready"
                        s.message = ("整幅参考已更新；部分线条仍有偏差，可继续改画或换一个方案"
                                     if check["alignment_warning"] else "整幅参考已更新；继续落笔会重新补全")
                    else:
                        if check["outcome"] == "noop":
                            s.status = "noop"
                            s.message = ("本次没有新增参考线；可以尝试辅助修稿，或继续画" if task == "completion"
                                         else "本次没有需要采用的微调；可调整选区、解锁笔画或换一个方案")
                        else:
                            s.status, s.message = "rejected", "模型返回空白或黑块过多，未显示本次参考；原稿已保留，可以重试"
                    s.serial += 1
                    folder = self.root / "data" / "sessions" / identifier
                    folder.mkdir(parents=True, exist_ok=True)
                    candidate.save(folder / "candidate.png")
                    lines.save(folder / "lines.png")
                    preview.save(folder / "ghost.png")
                    request["ink"].save(folder / "input.png")
                    (folder / "strokes.json").write_text(json.dumps(request.get("strokes", []), ensure_ascii=False), encoding="utf-8")
                    (folder / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
                    (folder / "correction.json").write_text(json.dumps(proposal, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                with self.condition:
                    if s.epoch == request["epoch"] and s.latest and s.latest["job_id"] == request["job_id"]:
                        s.status, s.message = "error", str(exc)[:800]
                        s.ghost = None
                        s.serial += 1
                        s.metrics = {"elapsed_seconds": round(time.monotonic()-started, 2)}
            finally:
                with self.condition:
                    self.active = None
