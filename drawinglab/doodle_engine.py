"""Optional DoodleAssist backend; never downloads or substitutes a base model.

The released ControlNeXt is loaded from reviewed, hash-pinned local source. Its
injection into the first UNet down block and latent distribution alignment follow
the official implementation at SOURCE_REVISION. We use installed Diffusers rather
than vendoring its older UNet copy, one FP16 candidate, and deterministic VAE means.
This backend has not been quality-validated until its required base is available.
"""
from __future__ import annotations

import hashlib
import importlib.util
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .inference import ModelEngine, ModelUnavailableError, GenerationCancelled, _number

SOURCE_REVISION = "b2c8c124e066e0160a040740b28a1162ab01af8b"
CONTROL_SOURCE_SHA256 = "b360e3973069dfa66d3e6a952b115e112f0a2da61080bcf0665b15f7ca8b6c5b"
CONTROL_SHA256 = "5c95d586f980b992f164244648598c61b8e50f907c742db5247a09af4ff46d00"
BASE_URL = "https://civitai.com/models/123631?modelVersionId=142306"


def load_control_class(source: Path):
    """Load only the explicitly reviewed official module, checking before import."""
    content = source.read_bytes()
    if hashlib.sha256(content).hexdigest() != CONTROL_SOURCE_SHA256:
        raise ModelUnavailableError("DoodleAssist ControlNeXt 源码校验失败，请重新运行下载脚本。")
    spec = importlib.util.spec_from_file_location("drawinglab_doodle_controlnext", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ControlNeXtModel


def align_distribution(source, target):
    """Official global mean/std alignment, evaluated in FP32 for FP16 stability."""
    source32, target32 = source.float(), target.float()
    axes = (1, 2, 3)
    source_mean = source32.mean(dim=axes, keepdim=True)
    target_mean = target32.mean(dim=axes, keepdim=True)
    result = ((source32 - source_mean)
              * target32.std(dim=axes, keepdim=True)
              / source32.std(dim=axes, keepdim=True).clamp_min(1e-6)
              + target_mean)
    return result.to(target)


def inject_control(sample, control, scale: float):
    """Equivalent to official UNet lines 1236–1244; residual skips stay unchanged."""
    import torch.nn.functional as F
    control = F.adaptive_avg_pool2d(control, sample.shape[-2:]).to(sample)
    return sample + align_distribution(control, sample) * scale


class DoodleEngine(ModelEngine):
    def __init__(self, root: Path):
        super().__init__(root)
        self.model_dir = self.model_dir / "doodle-assist"
        self.size = 512
        self.steps = int(_number("DRAWINGLAB_DOODLE_STEPS", 20, 8, 50))
        self.control_scale = _number("DRAWINGLAB_DOODLE_CONTROL_SCALE", 0.9, 0.1, 2.0)
        self.guidance_scale = _number("DRAWINGLAB_DOODLE_GUIDANCE", 7.5, 1.0, 15.0)
        # SD1.5 at 512px and batch one fits substantially below the official
        # Gradio's four-candidate configuration; avoid unverified custom offload.
        self.cpu_offload = False
        self._control = None
        self._active_control = None
        self._hook = None
        self._update(total_steps=self.steps)

    def _missing_files(self) -> list[str]:
        required = ["source/controlnext.py", "controlnext-48000.safetensors",
                    "base/model_index.json", "base/unet/config.json",
                    "base/unet/diffusion_pytorch_model.safetensors",
                    "base/vae/config.json", "base/vae/diffusion_pytorch_model.safetensors",
                    "base/text_encoder/config.json", "base/text_encoder/model.safetensors",
                    "base/tokenizer/vocab.json", "base/tokenizer/merges.txt",
                    "base/tokenizer/tokenizer_config.json", "base/scheduler/scheduler_config.json"]
        return [name for name in required if not (self.model_dir / name).is_file()]

    def status(self) -> dict:
        missing = self._missing_files()
        dependencies = [name for name in ("torch", "diffusers", "transformers", "safetensors")
                        if importlib.util.find_spec(name) is None]
        with self._state_lock:
            result = dict(self._state)
        result.update(ready=not missing and not dependencies, models_present=not missing,
                      missing_files=missing, missing_dependencies=dependencies,
                      model="DoodleAssist / foolkatGODOF_v3", controlnet="DoodleAssist ControlNeXt",
                      source_revision=SOURCE_REVISION, size=self.size, steps=self.steps,
                      device=self.device, cpu_offload=False, offline=True,
                      quality_validated=False, base_download_url=BASE_URL)
        if missing:
            result["message"] = ("DoodleAssist 尚缺本地基础模型。运行 scripts/download_doodle_models.py；"
                                 "若官方 Civitai 下载返回 451，需要自行取得 foolkatGODOF_v3.safetensors 后放入本项目。")
        elif dependencies:
            result["message"] = "缺少推理依赖：" + ", ".join(dependencies)
        return result

    def _ensure_loaded(self, cancel: Callable[[], bool]) -> float:
        if self._pipe is not None:
            return 0.0
        self._check_cancel(cancel)
        status = self.status()
        if not status["ready"]:
            raise ModelUnavailableError(status["message"])
        started = time.monotonic()
        self._update(phase="loading", message="正在加载本地 DoodleAssist 线稿补全模型。")
        import torch
        import diffusers
        from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
        from safetensors.torch import load_file
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError("没有可用的 CUDA GPU。")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        ControlNeXt = load_control_class(self.model_dir / "source/controlnext.py")
        control = ControlNeXt(controlnext_scale=self.control_scale)
        control.load_state_dict(load_file(str(self.model_dir / "controlnext-48000.safetensors")), strict=True)
        control.eval().to(device=self.device, dtype=dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            str(self.model_dir / "base"), torch_dtype=dtype, local_files_only=True,
            use_safetensors=True, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe.enable_vae_slicing()
        pipe.to(self.device)
        self._check_cancel(cancel)

        def apply_control(module, inputs, output):
            if self._active_control is None:
                return output
            sample, residuals = output
            return inject_control(sample, self._active_control, self.control_scale), residuals

        self._hook = pipe.unet.down_blocks[0].register_forward_hook(apply_control)
        self._pipe, self._control, self._torch = pipe, control, torch
        self._versions = {"torch": torch.__version__, "diffusers": diffusers.__version__,
                          "dtype": str(dtype), "gpu": torch.cuda.get_device_name(0)
                          if self.device == "cuda" else None}
        self._update(loaded=True, phase="ready", message="DoodleAssist 已加载。")
        return time.monotonic() - started

    def generate(self, ink: Image.Image, prompt: str, previous: Image.Image | None,
                 mask: Image.Image | None, seed: int, cancel: Callable[[], bool]):
        if not isinstance(ink, Image.Image) or not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("需要有效的笔迹图和提示词。")
        if previous is None and mask is not None:
            raise ValueError("更新遮罩需要上一张候选图。")
        self._check_cancel(cancel)
        queued = time.monotonic()
        while not self._serial.acquire(timeout=0.1):
            self._check_cancel(cancel)
        started = time.monotonic()
        self._update(busy=True, phase="preparing", step=0, total_steps=self.steps, last_error=None)
        try:
            sketch = ink.convert("RGB").resize((self.size, self.size), Image.Resampling.LANCZOS)
            if np.asarray(sketch).min() > 224:
                raise ValueError("请先在画布上画几笔。")
            load_seconds = self._ensure_loaded(cancel)
            torch, pipe = self._torch, self._pipe
            dtype = self._control.dtype
            generator = torch.Generator(device=self.device).manual_seed(int(seed) % (2**63 - 1))
            # The DoodleAssist adapter expects BLACK lines on WHITE, [0,1].
            sketch_tensor = torch.from_numpy(np.asarray(sketch).copy()).permute(2, 0, 1)[None]
            sketch_tensor = sketch_tensor.to(self.device, dtype=dtype) / 255
            context = (previous or Image.new("RGB", ink.size, "white")).convert("RGB")
            context = context.resize(sketch.size, Image.Resampling.LANCZOS)
            context_tensor = torch.from_numpy(np.asarray(context).copy()).permute(2, 0, 1)[None]
            context_tensor = context_tensor.to(self.device, dtype=dtype) / 127.5 - 1
            mask_image = mask.convert("L") if mask is not None else Image.new("L", ink.size, 255)
            mask_array = np.asarray(mask_image.resize((64, 64), Image.Resampling.BILINEAR)).copy()
            edit = torch.from_numpy(mask_array)[None, None].to(self.device, dtype=dtype) / 255
            effective_prompt = "lineart, monochrome, white background, " + prompt.strip()
            negative = "color, colored background, photograph, watermark, text, blurry"
            inference_started = time.monotonic()
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                positive, negative_embeds = pipe.encode_prompt(
                    effective_prompt, self.device, 1, True, negative_prompt=negative,
                )
                embeddings = torch.cat([negative_embeds, positive])
                background_latents = pipe.vae.encode(context_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor
                pipe.scheduler.set_timesteps(self.steps, device=self.device)
                noise = torch.randn((1, 4, 64, 64), generator=generator, device=self.device, dtype=dtype)
                latents = noise * pipe.scheduler.init_noise_sigma
                controls = torch.cat([sketch_tensor, sketch_tensor])
                for index, timestep in enumerate(pipe.scheduler.timesteps):
                    self._check_cancel(cancel)
                    if previous is not None:
                        noisy_background = pipe.scheduler.add_noise(background_latents, noise, timestep.reshape(1))
                        latents = latents * edit + align_distribution(noisy_background, latents) * (1 - edit)
                    model_input = pipe.scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
                    self._active_control = self._control(controls, timestep)["output"]
                    prediction = pipe.unet(model_input, timestep, encoder_hidden_states=embeddings, return_dict=False)[0]
                    unconditional, conditional = prediction.chunk(2)
                    prediction = unconditional + self.guidance_scale * (conditional - unconditional)
                    latents = pipe.scheduler.step(prediction, timestep, latents, return_dict=False)[0]
                    self._update(phase="denoising", step=index + 1,
                                 message=f"DoodleAssist 正在补全：{index + 1}/{self.steps} 步。")
                self._check_cancel(cancel)
                if previous is not None:
                    latents = latents * edit + align_distribution(background_latents, latents) * (1 - edit)
                decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                candidate = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            self._check_cancel(cancel)
            candidate = candidate.resize(ink.size, Image.Resampling.LANCZOS).convert("RGB")
            metadata = {"model": "DoodleAssist / foolkatGODOF_v3", "source_revision": SOURCE_REVISION,
                        "controlnet": "DoodleAssist ControlNeXt", "control_sha256": CONTROL_SHA256,
                        "pipeline": "DrawingLabDoodleAssist", "mode": "global" if previous is None else "soft_revision",
                        "size": [self.size, self.size], "output_size": list(ink.size), "seed": int(seed),
                        "requested_steps": self.steps, "completed_steps": self.steps,
                        "control_scale": self.control_scale, "guidance_scale": self.guidance_scale,
                        "prompt": effective_prompt, "negative_prompt": negative,
                        "prompt_warning": "此模型主要理解英文提示词。" if re.search(r"[\u3400-\u9fff]", prompt) else None,
                        "queue_seconds": round(started - queued, 3), "load_seconds": round(load_seconds, 3),
                        "inference_seconds": round(time.monotonic() - inference_started, 3),
                        "duration_seconds": round(time.monotonic() - started, 3),
                        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else None,
                        "device": self.device, "cpu_offload": False, "offline": True, "generated_by_ai": True,
                        "outside_mask_preserved": False, "style_personalization_trained": False,
                        "adaptations": ["single_candidate_fp16", "native_unet_equivalent_hook", "deterministic_vae_mean"],
                        **self._versions}
            self._update(phase="ready", message="线稿候选已生成。")
            return candidate, metadata
        except GenerationCancelled:
            self._update(phase="cancelled", message="这次生成已取消。")
            raise
        except Exception as exc:
            self._update(phase="error", last_error=str(exc), message=f"生成失败：{exc}")
            raise
        finally:
            self._active_control = None
            self._update(busy=False)
            self._serial.release()
