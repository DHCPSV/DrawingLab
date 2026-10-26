from __future__ import annotations
from .paths import configure_paths
ROOT = configure_paths()

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .imaging import decode_ink, has_ink
from .copaint import CopaintJobManager

class Focus(BaseModel):
    x: float = Field(default=384, ge=0, le=768)
    y: float = Field(default=384, ge=0, le=768)

class StrokePoint(BaseModel):
    x: float = Field(ge=0, le=768)
    y: float = Field(ge=0, le=768)

class Stroke(BaseModel):
    id: str = Field(min_length=1, max_length=100)
    tool: str = Field(pattern="^(pen|eraser)$")
    size: float = Field(ge=1, le=100)
    points: list[StrokePoint] = Field(min_length=1, max_length=100_000)
    provenance: str = Field(default="human", pattern="^(human|assisted)$")

    @model_validator(mode="after")
    def fixed_pen(self):
        if self.tool == "pen" and self.size != 3:
            raise ValueError("画笔固定为 3 px")
        return self

class EditRegion(BaseModel):
    x: float = Field(ge=0, le=768)
    y: float = Field(ge=0, le=768)
    width: float = Field(gt=0, le=768)
    height: float = Field(gt=0, le=768)

    @model_validator(mode="after")
    def inside_canvas(self):
        if self.x + self.width > 768 or self.y + self.height > 768:
            raise ValueError("修稿范围必须在画布内")
        return self

class Suggestion(BaseModel):
    revision: int = Field(ge=0)
    intent_revision: int = Field(default=0, ge=0)
    prompt: str = Field(min_length=1, max_length=2000)
    ink: str = Field(max_length=5_000_000)
    focus: Focus = Field(default_factory=Focus)
    eraser: bool = False
    force: bool = False
    seed: int | None = Field(default=None, ge=0, le=2**32-1)
    strokes: list[Stroke] = Field(default_factory=list, max_length=20_000)
    lookahead: float = Field(default=60, ge=20, le=100)
    task: str = Field(default="completion", pattern="^(completion|correction|cleanup)$")
    region: EditRegion | None = None
    locked_ids: list[str] = Field(default_factory=list, max_length=20_000)

    @model_validator(mode="after")
    def stroke_budget(self):
        if sum(len(stroke.points) for stroke in self.strokes) > 100_000:
            raise ValueError("建议请求的笔画点数超过 100,000；请减少画稿复杂度")
        identifiers = [stroke.id for stroke in self.strokes]
        identifier_set = set(identifiers)
        if len(identifier_set) != len(identifiers):
            raise ValueError("笔画 ID 必须唯一")
        if any(identifier not in identifier_set for identifier in self.locked_ids):
            raise ValueError("锁定的笔画不在当前画稿中")
        if self.task != "completion" and not self.strokes:
            raise ValueError("修稿需要原始矢量笔迹，请打开保存的项目或画几笔")
        return self

class Invalidation(BaseModel):
    revision: int = Field(ge=0)
    intent_revision: int = Field(default=0, ge=0)
    reset: bool = False

def validate_document(document: dict):
    if document.get("version") != 1 or document.get("width") != 768 or document.get("height") != 768:
        raise ValueError("不支持的项目格式或画布尺寸")
    strokes = document.get("strokes")
    if not isinstance(strokes, list) or len(strokes) > 20_000:
        raise ValueError("笔画数据无效或过多")
    if not isinstance(document.get("prompt", ""), str) or len(document.get("prompt", "")) > 2000:
        raise ValueError("描述过长")
    total = 0
    for stroke in strokes:
        if stroke.get("tool") not in ("pen", "eraser") or not isinstance(stroke.get("points"), list):
            raise ValueError("笔画格式无效")
        if stroke["tool"] == "pen":
            stroke["size"] = 3
        if not isinstance(stroke.get("size"), (float, int)) or not 1 <= stroke["size"] <= 100:
            raise ValueError("橡皮擦大小无效")
        total += len(stroke["points"])
        for point in stroke["points"]:
            if not isinstance(point, dict) or any(not isinstance(point.get(axis), (int, float)) or not 0 <= point[axis] <= 768 for axis in ("x", "y")):
                raise ValueError("笔画坐标无效")
    if total > 500_000:
        raise ValueError("项目点数过多")

def create_app(engine=None, root: Path = ROOT) -> FastAPI:
    if engine is None:
        from .engines import create_engine
        engine = create_engine(root)
    manager = CopaintJobManager(engine, root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        manager.close()

    app = FastAPI(title="DrawingLab", lifespan=lifespan)
    app.state.manager = manager
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "testserver"):
            return JSONResponse({"detail": "仅接受本机画布请求"}, status_code=403)
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return JSONResponse({"detail": "无效的请求长度"}, status_code=400)
        if content_length > 12_000_000:
            return JSONResponse({"detail": "请求过大"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    def session_exists(identifier):
        if identifier not in manager.sessions:
            raise HTTPException(404, "绘画会话已失效，请刷新页面后重新打开已保存项目")

    @app.get("/api/status")
    def status():
        return {"version": 4, "release": "0.1.0", "model": engine.status(), "features": {
            "correction": bool(getattr(engine, "supports_correction", False)), "cleanup": True},
            "canvas": {"width": 768, "height": 768, "pen_size": 3}, "project_root": str(root)}

    @app.post("/api/sessions")
    def create_session():
        try:
            return {"id": manager.create()}
        except ValueError as exc:
            raise HTTPException(429, str(exc))

    @app.post("/api/sessions/{identifier}/suggest", status_code=202)
    def suggest(identifier: str, body: Suggestion):
        session_exists(identifier)
        try:
            ink = decode_ink(body.ink)
            if not has_ink(ink):
                raise ValueError("请先在画布上画几笔")
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        model = engine.status()
        if body.task != "cleanup" and not model.get("ready"):
            raise HTTPException(503, model.get("message", "模型尚未就绪"))
        if body.task == "correction" and not getattr(engine, "supports_correction", False):
            raise HTTPException(422, "当前后端不支持 AI 修稿；可以使用整理线条，或启用 Klein 后端")
        try:
            return {"job_id": manager.submit(identifier, dict(body.model_dump(), ink=ink))}
        except ValueError as exc:
            raise HTTPException(409, str(exc))

    @app.get("/api/sessions/{identifier}/result")
    def result(identifier: str, after: int = -1):
        session_exists(identifier)
        return manager.result(identifier)

    @app.post("/api/sessions/{identifier}/invalidate")
    def invalidate(identifier: str, body: Invalidation):
        session_exists(identifier)
        manager.invalidate(identifier, **body.model_dump())
        return {"ok": True}

    @app.post("/api/projects")
    async def save_project(request: Request):
        try:
            payload = await request.json()
            document = payload["document"]
            validate_document(document)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise HTTPException(422, str(exc))
        directory = root / "data" / "projects"
        directory.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("drawing-%Y%m%d-%H%M%S-%f.json")
        path = directory / name
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"document": document}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return {"name": name, "path": str(path)}

    @app.get("/api/projects")
    def projects():
        directory = root / "data" / "projects"
        return {"projects": [{"name": p.name, "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()} for p in sorted(directory.glob("*.json"), reverse=True)[:100]]}

    @app.get("/api/projects/{name}")
    def load_project(name: str):
        if not re.fullmatch(r"drawing-[0-9-]+\.json", name):
            raise HTTPException(404, "项目不存在")
        path = root / "data" / "projects" / name
        if not path.is_file():
            raise HTTPException(404, "项目不存在")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/exports")
    async def export(request: Request):
        try:
            ink = decode_ink((await request.json())["ink"])
        except (ValueError, KeyError, TypeError) as exc:
            raise HTTPException(422, str(exc))
        directory = root / "data" / "exports"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / datetime.now().strftime("drawing-%Y%m%d-%H%M%S-%f.png")
        ink.save(path)
        return {"name": path.name, "path": str(path)}

    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
    @app.get("/")
    def index():
        return FileResponse(ROOT / "static" / "index.html")
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="127.0.0.1", port=int(os.environ.get("DRAWINGLAB_PORT", "8765")), access_log=False)
