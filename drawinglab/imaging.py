"""Image boundary and deterministic line-art checks; never invents suggestions."""
from __future__ import annotations
import base64
from io import BytesIO
import math
import cv2
import numpy as np
from PIL import Image

SIZE = 768

def decode_ink(data: str) -> Image.Image:
    if not data.startswith("data:image/png;base64,") or len(data) > 5_000_000:
        raise ValueError("需要小于 4MB 的 PNG 画布")
    try:
        raw = base64.b64decode(data.split(",", 1)[1], validate=True)
        image = Image.open(BytesIO(raw))
        if image.format != "PNG" or image.size != (SIZE, SIZE):
            raise ValueError("画布必须是 768 × 768 PNG")
        image.load()
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("L")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError(f"无法读取画布：{exc}") from exc

def encode_png(image: Image.Image) -> str:
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

def has_ink(ink: Image.Image) -> bool:
    return int(np.count_nonzero(np.asarray(ink) < 200)) >= 8

def extract_lines(candidate: Image.Image) -> Image.Image:
    """The generator is prompted for monochrome line art; retain its actual lines.

    No edge detector on the two sides of a thick black stroke, no template body,
    no SVG reconstruction. Dense/blank outputs are rejected by the next stage.
    """
    gray = np.asarray(candidate.convert("L"), dtype=np.uint8)
    # Estimate nearby paper tone, so pale pencil contours survive while broad
    # background gradients disappear. Retain genuinely dark filled eye details.
    paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((17, 17), np.uint8))
    contrast = paper.astype(np.float32) - gray.astype(np.float32)
    local_ink = np.clip((contrast - 2) * 5.5, 0, 255)
    dark_ink = np.clip((135 - gray.astype(np.float32)) * 2.5, 0, 255)
    strength = np.maximum(local_ink, dark_ink).astype(np.uint8)
    binary = (strength > 12).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 3).astype(np.uint8)
    strength *= keep[labels]
    return Image.fromarray(255 - strength, "L")

def alignment(ink: Image.Image, lines: Image.Image) -> dict:
    """Legacy raster diagnostic retained for before/after research only.

    A high score does not establish correspondence for individual strokes.
    Production path publication and cache reuse must use assess_geometry.
    """
    ink_pixels = np.asarray(ink) < 200
    line_pixels = np.asarray(lines) < 185
    density = float(line_pixels.mean())
    if not ink_pixels.any() or not line_pixels.any():
        return {"accepted": False, "reason": "候选没有可用轮廓", "density": density, "matched": 0.0}
    distance = cv2.distanceTransform((~line_pixels).astype(np.uint8), cv2.DIST_L2, 5)
    errors = distance[ink_pixels]
    matched = float((errors <= 10).mean())
    p90 = float(np.quantile(errors, .90))
    accepted = .0002 < density < .32 and matched >= .70 and p90 <= 22
    reason = "通过笔迹贴合检查" if accepted else "新建议未能贴合已有笔迹，已隐藏；可以调整描述或换一个方案"
    if density >= .32:
        reason = "生成图的黑块过多，无法作为清晰参考线；请换一个方案"
    return {"accepted": bool(accepted), "reason": reason, "density": round(density, 4),
            "matched": round(matched, 4), "distance_p90": round(p90, 2)}

def ghost_image(lines: Image.Image, ink: Image.Image) -> Image.Image:
    """Offline raster preview for research scripts; never a V2 UI suggestion."""
    strength = 255 - np.asarray(lines)
    occupied = cv2.dilate((np.asarray(ink) < 220).astype(np.uint8), np.ones((5, 5), np.uint8))
    alpha = strength.copy()
    alpha[occupied > 0] = 0
    rgba = np.empty((SIZE, SIZE, 4), dtype=np.uint8)
    rgba[:, :, :3] = 145
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba, "RGBA")

def update_mask(old_ink: Image.Image, new_ink: Image.Image, focus: dict) -> Image.Image:
    """Include added AND erased pixels, with a generous connection margin."""
    difference = np.abs(np.asarray(old_ink, dtype=np.int16) - np.asarray(new_ink, dtype=np.int16)) > 25
    y, x = np.where(difference)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    if not len(x):
        cx, cy = int(focus.get("x", SIZE/2)), int(focus.get("y", SIZE/2))
        cv2.circle(mask, (cx, cy), 155, 255, -1)
    else:
        margin = 112
        left, right = max(0, int(x.min())-margin), min(SIZE, int(x.max())+margin+1)
        top, bottom = max(0, int(y.min())-margin), min(SIZE, int(y.max())+margin+1)
        mask[top:bottom, left:right] = 255
    return Image.fromarray(mask, "L")
