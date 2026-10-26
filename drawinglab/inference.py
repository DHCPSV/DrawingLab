"""Offline SDXL + anime-painter inference for DrawingLab.

One ControlNet inpainting pipeline handles both a fully unknown canvas (all-white
mask, strength=1) and local revisions of the previous candidate. Its components
are loaded once, shared by all requests, and guarded by a serial generation lock.
The 4-channel SDXL UNet path is supported by Diffusers' ControlNet Inpaint API.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Callable

from PIL import Image, ImageOps


BASE_MODEL = "gsdf/CounterfeitXL"
BASE_REVISION = "4708675873bd09833aabc3fd4cb2de5fcd1726ac"
CONTROL_MODEL = "xinsir/anime-painter"
CONTROL_REVISION = "18185a73b6e7fe49f2f2de1bb9d7db0b74a41773"

# Exact sizes from the pinned public Hugging Face manifests. A .part file never
# counts as a ready model. Cryptographic verification belongs to the downloader.
REQUIRED_FILES = {
    "counterfeit-xl/model_index.json": 577,
    "counterfeit-xl/scheduler/scheduler_config.json": 474,
    "counterfeit-xl/text_encoder/config.json": 565,
    "counterfeit-xl/text_encoder/model.fp16.safetensors": 246144152,
    "counterfeit-xl/text_encoder_2/config.json": 575,
    "counterfeit-xl/text_encoder_2/model.fp16.safetensors": 1389382176,
    "counterfeit-xl/tokenizer/merges.txt": 524619,
    "counterfeit-xl/tokenizer/special_tokens_map.json": 472,
    "counterfeit-xl/tokenizer/tokenizer_config.json": 737,
    "counterfeit-xl/tokenizer/vocab.json": 1059962,
    "counterfeit-xl/tokenizer_2/merges.txt": 524619,
    "counterfeit-xl/tokenizer_2/special_tokens_map.json": 460,
    "counterfeit-xl/tokenizer_2/tokenizer_config.json": 725,
    "counterfeit-xl/tokenizer_2/vocab.json": 1059962,
    "counterfeit-xl/unet/config.json": 1675,
    "counterfeit-xl/unet/diffusion_pytorch_model.fp16.safetensors": 5135149760,
    "counterfeit-xl/vae/config.json": 602,
    "counterfeit-xl/vae/diffusion_pytorch_model.fp16.safetensors": 167335342,
    "anime-painter/config.json": 1235,
    "anime-painter/diffusion_pytorch_model.safetensors": 2502139104,
}


class GenerationCancelled(RuntimeError):
    """The request became obsolete or the user cancelled it."""


class ModelUnavailableError(RuntimeError):
    """Required local models, dependencies, or GPU are unavailable."""


def _number(name: str, default: float, lower: float, upper: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


class ModelEngine:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.model_dir = Path(os.environ.get("DRAWINGLAB_MODEL_DIR", str(self.root / "models"))).resolve()
        # The released SDXL control model targets 1024px. On the local 12GB
        # GPU this stays within memory with offload and preserves sketch
        # placement substantially better than the previous 768px recipe.
        # The artist canvas remains 768px; only inference uses this resolution.
        self.size = int(_number("DRAWINGLAB_SIZE", 1024, 512, 1536))
        if self.size % 64:
            raise ValueError("DRAWINGLAB_SIZE must be divisible by 64")
        self.steps = int(_number("DRAWINGLAB_STEPS", 26, 8, 80))
        self.control_scale = _number("DRAWINGLAB_CONTROL_SCALE", 1.0, 0.1, 2.0)
        self.guidance_scale = _number("DRAWINGLAB_GUIDANCE", 6.5, 1.0, 15.0)
        self.inpaint_strength = _number("DRAWINGLAB_INPAINT_STRENGTH", 0.9, 0.25, 1.0)
        self.cpu_offload = os.environ.get("DRAWINGLAB_CPU_OFFLOAD", "1").lower() not in ("0", "false", "no")
        self.device = os.environ.get("DRAWINGLAB_DEVICE", "cuda")
        if self.device not in ("cuda", "cpu"):
            raise ValueError("DRAWINGLAB_DEVICE must be cuda or cpu")
        self._serial = threading.Lock()
        self._state_lock = threading.Lock()
        self._pipe = None
        self._torch = None
        self._versions: dict = {}
        self._state = {"busy": False, "loaded": False, "phase": "idle", "step": 0,
                       "total_steps": self.steps, "last_error": None, "message": "模型将在首次生成时加载。"}
        self._configure_environment()

    def _configure_environment(self) -> None:
        paths = {
            "HF_HOME": self.root / ".cache/huggingface",
            "HF_HUB_CACHE": self.root / ".cache/huggingface/hub",
            "HUGGINGFACE_HUB_CACHE": self.root / ".cache/huggingface/hub",
            "HF_XET_CACHE": self.root / ".cache/huggingface/xet",
            "TRANSFORMERS_CACHE": self.root / ".cache/huggingface/transformers",
            "DIFFUSERS_CACHE": self.root / ".cache/huggingface/diffusers",
            "TORCH_HOME": self.root / ".cache/torch",
            "TORCHINDUCTOR_CACHE_DIR": self.root / ".cache/torchinductor",
            "TRITON_CACHE_DIR": self.root / ".cache/triton",
            "CUDA_CACHE_PATH": self.root / ".cache/cuda",
            "XDG_CACHE_HOME": self.root / ".cache",
            "TEMP": self.root / ".cache/tmp",
            "TMP": self.root / ".cache/tmp",
            "TMPDIR": self.root / ".cache/tmp",
        }
        for name, path in paths.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ[name] = str(path)
        tempfile.tempdir = str(self.root / ".cache/tmp")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"

    def _update(self, **values) -> None:
        with self._state_lock:
            self._state.update(values)

    def _missing_files(self) -> list[str]:
        missing = []
        for filename, expected_bytes in REQUIRED_FILES.items():
            path = self.model_dir / filename
            try:
                if not path.is_file() or path.stat().st_size != expected_bytes:
                    missing.append(filename)
            except OSError:
                missing.append(filename)
        return missing

    def status(self) -> dict:
        missing = self._missing_files()
        dependencies = [name for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors")
                        if importlib.util.find_spec(name) is None]
        with self._state_lock:
            result = dict(self._state)
        result.update({"ready": not missing and not dependencies,
                       "models_present": not missing,
                       "missing_files": missing,
                       "missing_dependencies": dependencies,
                       "model": BASE_MODEL, "controlnet": CONTROL_MODEL,
                       "size": self.size, "steps": self.steps, "device": self.device,
                       "cpu_offload": self.cpu_offload, "offline": True,
                       "expected_model_bytes": sum(REQUIRED_FILES.values())})
        if missing:
            result["message"] = f"模型未下载完整（还缺 {len(missing)} 个文件）。运行 scripts/download_models.py。"
        elif dependencies:
            result["message"] = "缺少推理依赖：" + ", ".join(dependencies)
        return result

    @staticmethod
    def _check_cancel(cancel: Callable[[], bool]) -> None:
        if cancel():
            raise GenerationCancelled("已取消这次生成；没有产生新的候选图。")

    def _ensure_loaded(self, cancel: Callable[[], bool]) -> float:
        if self._pipe is not None:
            return 0.0
        started = time.monotonic()
        self._check_cancel(cancel)
        current = self.status()
        if not current["ready"]:
            raise ModelUnavailableError(current["message"])
        self._update(phase="loading", message="正在从本地磁盘加载动漫模型，首次加载需要一些时间。")
        try:
            import torch
            import diffusers
            import transformers
            from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetInpaintPipeline
        except (ImportError, OSError) as exc:
            raise ModelUnavailableError(f"无法导入本地推理环境：{exc}") from exc
        self._torch = torch
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError("没有可用的 CUDA GPU。请用项目的 Python 环境并安装对应的 CUDA PyTorch；程序不会偷偷改用 CPU。")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
        self._check_cancel(cancel)
        controlnet = ControlNetModel.from_pretrained(
            str(self.model_dir / "anime-painter"), torch_dtype=dtype,
            use_safetensors=True, local_files_only=True,
        )
        self._check_cancel(cancel)
        # No custom_pipeline/trust_remote_code, remote URL, or pickle checkpoint.
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            str(self.model_dir / "counterfeit-xl"), controlnet=controlnet,
            torch_dtype=dtype, variant="fp16", use_safetensors=True,
            local_files_only=True, add_watermarker=False,
        )
        self._check_cancel(cancel)
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda" and self.cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            pipe.to(self.device)
        self._pipe = pipe
        self._versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                          "transformers": transformers.__version__,
                          "dtype": str(dtype),
                          "gpu": torch.cuda.get_device_name(0) if self.device == "cuda" else None}
        self._update(loaded=True, phase="ready", message="本地模型已加载。")
        return time.monotonic() - started

    def generate(self, ink: Image.Image, prompt: str, previous: Image.Image | None,
                 mask: Image.Image | None, seed: int, cancel: Callable[[], bool]) -> tuple[Image.Image, dict]:
        """Return a real color candidate. Black mask pixels preserve the previous image.

        Cancellation is checked before loading and between denoising steps. A
        cancelled request raises GenerationCancelled and never returns a stale
        partial image. Model loading and an individual GPU operation cannot be
        preempted mid-call. All model inference is local and serialized.
        """
        if not isinstance(ink, Image.Image):
            raise TypeError("ink 必须是 PIL.Image")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("请先填写要画的内容。")
        if mask is not None and previous is None:
            raise ValueError("局部遮罩需要上一张候选图。")
        if previous is not None and not isinstance(previous, Image.Image):
            raise TypeError("previous 必须是 PIL.Image 或 None")
        if mask is not None and not isinstance(mask, Image.Image):
            raise TypeError("mask 必须是 PIL.Image 或 None")
        self._check_cancel(cancel)
        queued_at = time.monotonic()
        while not self._serial.acquire(timeout=0.1):
            self._check_cancel(cancel)
        started = time.monotonic()
        self._update(busy=True, phase="preparing", step=0, total_steps=self.steps, last_error=None)
        try:
            self._check_cancel(cancel)
            grayscale = ink.convert("L").resize((self.size, self.size), Image.Resampling.LANCZOS)
            # The released anime-painter expects WHITE strokes on BLACK. Keep
            # foreground antialias coverage when reducing a larger canvas.
            control = grayscale.point(lambda value: 255 if value < 224 else 0).convert("RGB")
            if control.getbbox() is None:
                raise ValueError("画布没有可见笔迹，请先画几笔。")
            if previous is None:
                mode = "global"
                context = Image.new("RGB", (self.size, self.size), "white")
                edit_mask = Image.new("L", context.size, 255)
                strength = 1.0
            else:
                context = previous.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS)
                edit_mask = (mask.convert("L").resize(context.size, Image.Resampling.BILINEAR)
                             if mask is not None else Image.new("L", context.size, 255))
                if edit_mask.getbbox() is None:
                    raise ValueError("本次局部修订的遮罩为空。")
                mode = "inpaint" if mask is not None else "context_refresh"
                strength = self.inpaint_strength
            # Diffusers binarizes masks; give every nonzero feather pixel to
            # diffusion, then apply the original alpha for exact outside keep.
            diffusion_mask = edit_mask.point(lambda value: 255 if value > 0 else 0)
            load_seconds = self._ensure_loaded(cancel)
            self._check_cancel(cancel)
            pipe = self._pipe
            torch = self._torch
            effective_prompt = "monochrome lineart, black lines on pure white background, " + prompt.strip()
            negative_prompt = "color, colored background, gray background, painting, photograph, paper texture, desk, shadows, border, frame, blurry, extra limbs, watermark, text"
            token_counts = [len(tokenizer(effective_prompt, truncation=False)["input_ids"])
                            for tokenizer in (pipe.tokenizer, pipe.tokenizer_2)]
            truncated = any(count > tokenizer.model_max_length
                            for count, tokenizer in zip(token_counts, (pipe.tokenizer, pipe.tokenizer_2)))
            prompt_warning = "提示词超过 CLIP 长度上限，尾部会被截断。" if truncated else None
            if re.search(r"[\u3400-\u9fff]", prompt):
                prompt_warning = ((prompt_warning + " ") if prompt_warning else "") + "当前动漫模型主要使用英文／Danbooru 标签；中文理解能力有限。"
            completed_steps = 0
            planned_steps = max(1, int(self.steps * strength))

            def on_step_end(pipeline, step_index, timestep, callback_kwargs):
                nonlocal completed_steps
                self._check_cancel(cancel)
                completed_steps = step_index + 1
                self._update(phase="denoising", step=completed_steps, total_steps=planned_steps,
                             message=f"本地 AI 正在推理：{completed_steps}/{planned_steps} 步。")
                return callback_kwargs

            self._update(phase="denoising", total_steps=planned_steps, message="本地 AI 正在生成相容候选图。")
            actual_seed = int(seed) % (2**63 - 1)
            generator = torch.Generator(device="cpu").manual_seed(actual_seed)
            inference_started = time.monotonic()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                result = pipe(
                    prompt=effective_prompt, negative_prompt=negative_prompt,
                    image=context, mask_image=diffusion_mask, control_image=control,
                    height=self.size, width=self.size, strength=strength,
                    num_inference_steps=self.steps, guidance_scale=self.guidance_scale,
                    controlnet_conditioning_scale=self.control_scale,
                    control_guidance_start=0.0, control_guidance_end=1.0,
                    generator=generator, num_images_per_prompt=1, output_type="pil",
                    callback_on_step_end=on_step_end,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            self._check_cancel(cancel)
            candidate = result.images[0].convert("RGB")
            if candidate.size != ink.size:
                candidate = candidate.resize(ink.size, Image.Resampling.LANCZOS)
            if previous is not None:
                original_context = previous.convert("RGB").resize(ink.size, Image.Resampling.LANCZOS)
                original_mask = mask.convert("L").resize(ink.size, Image.Resampling.BILINEAR) if mask is not None else Image.new("L", ink.size, 255)
                candidate = Image.composite(candidate, original_context, original_mask)
            metadata = {
                "model": BASE_MODEL, "model_revision": BASE_REVISION,
                "controlnet": CONTROL_MODEL, "controlnet_revision": CONTROL_REVISION,
                "pipeline": type(pipe).__name__, "mode": mode,
                "seed": actual_seed, "size": [self.size, self.size],
                "output_size": list(ink.size),
                "requested_steps": self.steps, "completed_steps": completed_steps,
                "strength": strength, "control_scale": self.control_scale,
                "guidance_scale": self.guidance_scale,
                "scheduler": type(pipe.scheduler).__name__,
                "prompt": effective_prompt, "negative_prompt": negative_prompt,
                "prompt_warning": prompt_warning, "prompt_token_counts": token_counts,
                "prompt_truncated": truncated,
                "queue_seconds": round(started - queued_at, 3),
                "load_seconds": round(load_seconds, 3),
                "inference_seconds": round(time.monotonic() - inference_started, 3),
                "duration_seconds": round(time.monotonic() - started, 3),
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
                "device": self.device, "cpu_offload": self.cpu_offload and self.device == "cuda",
                "offline": True, "generated_by_ai": True,
                "outside_mask_preserved": previous is not None,
                **self._versions,
            }
            self._update(phase="ready", message="候选图已生成。", last_error=None)
            return candidate, metadata
        except GenerationCancelled:
            self._update(phase="cancelled", message="这次生成已取消。")
            raise
        except Exception as exc:
            self._update(phase="error", last_error=str(exc), message=f"生成失败：{exc}")
            if self._torch is not None and isinstance(exc, self._torch.cuda.OutOfMemoryError):
                raise ModelUnavailableError("GPU 显存不足。请关闭占用显存的应用，保持 DRAWINGLAB_CPU_OFFLOAD=1，或降低 DRAWINGLAB_SIZE。") from exc
            raise
        finally:
            # A callback exception skips Diffusers' normal final offload. Restore
            # idle hooks before the next request; never let cleanup hide the error.
            if self._pipe is not None and self.cpu_offload and self.device == "cuda":
                try:
                    self._pipe.maybe_free_model_hooks()
                except Exception:
                    pass
            self._update(busy=False)
            self._serial.release()
