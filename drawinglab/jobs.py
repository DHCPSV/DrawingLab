from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import json
import secrets
import threading
import time
from typing import Any
from PIL import Image
from .imaging import extract_lines, update_mask
from .geometry import build_geometry, assess_geometry, select_continuations
from .registration import align_geometry

@dataclass
class Session:
    id: str
    revision: int = -1
    intent_revision: int = 0
    epoch: int = 0
    serial: int = 0
    status: str = "idle"
    message: str = "画几笔后生成参考线"
    pending: dict | None = None
    latest: dict | None = None
    candidate: Image.Image | None = None
    lines: Image.Image | None = None
    geometry: dict | None = None
    candidate_ink: Image.Image | None = None
    candidate_id: str | None = None
    ghost: str | None = None
    paths: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    touched: float = field(default_factory=time.monotonic)

class JobManager:
    """One GPU worker; each session retains at most one latest pending request."""
    def __init__(self, engine: Any, root: Path):
        self.engine, self.root = engine, root
        self.sessions: dict[str, Session] = {}
        self.condition = threading.Condition(threading.RLock())
        self.queue: deque[str] = deque()
        self.stopped = False
        self.active: str | None = None
        self.worker = threading.Thread(target=self._work, name="drawinglab-gpu", daemon=True)
        self.worker.start()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.worker.join(timeout=2)

    def create(self) -> str:
        with self.condition:
            if len(self.sessions) >= 16:
                idle = [s for s in self.sessions.values() if not s.pending and s.id != self.active]
                if not idle:
                    raise ValueError("同时打开的绘画会话过多")
                del self.sessions[min(idle, key=lambda s: s.touched).id]
            identifier = secrets.token_hex(12)
            self.sessions[identifier] = Session(identifier)
            return identifier

    def invalidate(self, identifier: str, revision: int, intent_revision: int, reset: bool):
        with self.condition:
            session = self.sessions[identifier]
            if revision < session.revision or intent_revision < session.intent_revision:
                return
            session.epoch += 1
            session.revision, session.intent_revision = revision, intent_revision
            session.pending = None
            session.latest = None
            session.ghost = None
            session.paths, session.suggestions = [], []
            session.serial += 1
            session.status, session.message = "idle", "已更新画稿，等待下一次建议"
            if reset:
                session.candidate = session.lines = session.candidate_ink = None
                session.geometry = None
                session.candidate_id = None

    def submit(self, identifier: str, request: dict) -> str:
        with self.condition:
            session = self.sessions[identifier]
            if request["revision"] < session.revision or request["intent_revision"] < session.intent_revision:
                raise ValueError("已忽略过期画稿")
            if request["intent_revision"] != session.intent_revision or request.get("force"):
                session.epoch += 1
                session.candidate = session.lines = session.candidate_ink = None
                session.geometry = None
                session.candidate_id = None
                session.ghost = None
                session.paths, session.suggestions = [], []
            elif request.get("eraser"):
                # Deletion is not additive. Cancel older work but keep its last
                # accepted image as context for the automatically masked repair.
                session.epoch += 1
                session.ghost = None
                session.paths, session.suggestions = [], []
            session.revision = request["revision"]
            session.intent_revision = request["intent_revision"]
            session.touched = time.monotonic()
            request = dict(request, job_id=secrets.token_hex(8), epoch=session.epoch)
            session.latest = session.pending = request
            # A new request has not yet passed per-stroke validation. Do not
            # publish old paths with the new revision during the queued state.
            session.paths, session.suggestions = [], []
            session.status, session.message = "queued", "等待生成参考线"
            session.serial += 1
            if identifier not in self.queue:
                self.queue.append(identifier)
            self.condition.notify()
            return request["job_id"]

    def result(self, identifier: str) -> dict:
        with self.condition:
            s = self.sessions[identifier]
            s.touched = time.monotonic()
            return {"serial": s.serial, "status": s.status, "revision": s.revision,
                    "intent_revision": s.intent_revision, "candidate_id": s.candidate_id,
                    "ghost": None, "paths_version": 1, "paths": s.paths,
                    "suggestions": s.suggestions, "message": s.message, "metrics": s.metrics}

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
                s.status, s.message = "running", "正在生成；画布可以继续使用"
                s.serial += 1
                previous, previous_ink, old_lines, old_geometry = s.candidate, s.candidate_ink, s.lines, s.geometry
                previous_registration = s.metrics.get("registration", {"applied": False})
            started = time.monotonic()
            try:
                # A real compatible candidate can serve successive traced strokes instantly.
                cached = old_geometry is not None and not request.get("eraser") and not request.get("force")
                check = assess_geometry(request["ink"], old_geometry, request.get("strokes", [])) if cached else {"accepted": False}
                if cached and check["accepted"]:
                    candidate, lines, geometry = previous, old_lines, old_geometry
                    metrics = {"cached": True, "registration": previous_registration}
                else:
                    mask = update_mask(previous_ink, request["ink"], request["focus"]) if previous is not None and previous_ink is not None else None
                    candidate, metrics = self.engine.generate(
                        ink=request["ink"], prompt=request["prompt"], previous=previous,
                        mask=mask, seed=request["seed"] if request.get("seed") is not None else secrets.randbelow(2**31),
                        cancel=lambda: self.stopped or s.epoch != request["epoch"],
                    )
                    if candidate.size != request["ink"].size:
                        raise RuntimeError("模型返回了不匹配的画布尺寸")
                    lines = extract_lines(candidate)
                    geometry = build_geometry(lines)
                    with self.condition:
                        if self.stopped or s.epoch != request["epoch"] or not s.latest:
                            continue
                        target = s.latest
                    geometry, registration = align_geometry(target["ink"], geometry, target.get("strokes", []),
                                                              target["focus"], target.get("lookahead", 60))
                    metrics = dict(metrics, cached=False, registration=registration)
                with self.condition:
                    if self.stopped or s.epoch != request["epoch"] or not s.latest:
                        continue
                    # A newer ADDITIVE stroke is allowed only after checking the newest ink.
                    latest = s.latest
                    check = assess_geometry(latest["ink"], geometry, latest.get("strokes", []))
                    s.metrics = dict(metrics, **check, elapsed_seconds=round(time.monotonic()-started, 2))
                    if not check["accepted"]:
                        s.ghost = None
                        s.paths, s.suggestions = [], []
                        s.status, s.message = "rejected", check["reason"]
                    else:
                        s.candidate, s.lines, s.geometry = candidate, lines, geometry
                        s.candidate_ink = latest["ink"].copy()
                        s.candidate_id = s.candidate_id if metrics.get("cached") else secrets.token_hex(8)
                        matched = set(check.get("matched_path_ids", []))
                        owners = check.get("path_owners", {})
                        s.paths = [dict(path, source_stroke_ids=owners.get(path["id"], []))
                                   for path in geometry["paths"] if path["id"] in matched]
                        s.suggestions = select_continuations(geometry, latest["ink"], latest.get("strokes", []),
                                                             latest["focus"], latest.get("lookahead", 60))
                        s.metrics["path_count"] = len(s.paths)
                        s.metrics["suggestion_count"] = len(s.suggestions)
                        s.ghost = None
                        s.status = "ready"
                        s.message = (("已沿当前候选更新短线建议" if metrics.get("cached") else "短线建议已更新")
                                     if s.suggestions else "已有轮廓通过检查；当前笔尖没有可靠的续接线，请继续画或移到已有线的端点")
                    s.serial += 1
                    folder = self.root / "data" / "sessions" / identifier
                    folder.mkdir(parents=True, exist_ok=True)
                    # Bounded on-disk evidence: overwrite latest artifacts, never the artwork.
                    candidate.save(folder / "candidate.png")
                    lines.save(folder / "lines.png")
                    latest["ink"].save(folder / "input.png")
                    (folder / "strokes.json").write_text(json.dumps(latest.get("strokes", []), ensure_ascii=False), encoding="utf-8")
                    (folder / "paths.json").write_text(json.dumps({"paths_version": 1, "paths": s.paths,
                                                                  "suggestions": s.suggestions}, ensure_ascii=False), encoding="utf-8")
                    (folder / "metrics.json").write_text(json.dumps(s.metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                with self.condition:
                    if s.epoch == request["epoch"]:
                        s.status, s.message = "error", str(exc)[:800]
                        s.ghost = None
                        s.paths, s.suggestions = [], []
                        s.serial += 1
                        s.metrics = {"elapsed_seconds": round(time.monotonic()-started, 2)}
            finally:
                with self.condition:
                    self.active = None
