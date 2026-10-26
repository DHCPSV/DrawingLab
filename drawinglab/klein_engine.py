"""Offline image-conditioned FLUX.2 klein 4B completion.

Uses the official Diffusers Flux2KleinPipeline, not a remote/custom pipeline.
Current user ink is always reference image 1. A previous AI hypothesis may be
reference image 2 only for a small, geometrically compatible update. There is
no outside-mask compositing or image-result cache in this engine.
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from pathlib import Path
import threading
import time
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .inference import GenerationCancelled, ModelEngine, ModelUnavailableError, _number
from .klein_manifest import KLEIN_DIRECTORY, KLEIN_FILES, KLEIN_MODEL, KLEIN_REVISION


def _on_white(image: Image.Image) -> Image.Image:
    rgba = image.convert('RGBA')
    paper = Image.new('RGBA', rgba.size, 'white')
    paper.alpha_composite(rgba)
    return paper.convert('RGB')


def correction_prompt(prompt: str, region: dict | None = None) -> str:
    """Correction deliberately permits geometry edits; completion does not."""
    instruction = (
        'Refine the rough character sketch in image 1. Make a visibly improved version of this same drawing: '
        'gently correct uneven facial contours, the placement of facial features and awkward neck/shoulder connections. '
        'Replace hesitant or wobbly contours with confident, flowing single strokes. Small changes to positions, '
        'curve shapes and lengths are allowed and necessary. Preserve the same character, head tilt, pose, '
        'expression, hairstyle silhouette, framing and simple hand-drawn style. Keep intentional sharp corners '
        'and stylized proportions. Do not add shading, rendering, textures, fine decorative detail, new objects '
        'or a different hairstyle. Keep the drawing at the same overall position and scale. '
        'Return only the revised drawing: thin uniform black lines on a pure white background. '
        'Subject and user intent: ' + prompt.strip()
    )
    if region:
        instruction += (f' Focus the subtle refinements on the region from '
                        f'({round(region["x"])}, {round(region["y"])}) to '
                        f'({round(region["x"]+region["width"])}, {round(region["y"]+region["height"])}) '
                        'in the 768 by 768 canvas; retain surrounding context.')
    return instruction


def prepare_references(ink: Image.Image, previous: Image.Image | None,
                       mask: Image.Image | None, size: int) -> tuple[list[Image.Image], dict]:
    """Previous output is optional context, never evidence of the user's style."""
    actual = _on_white(ink).resize((size, size), Image.Resampling.LANCZOS)
    metadata = {'reference_count': 1, 'style_source': 'current_user_ink_only',
                'previous_reference_used': False, 'previous_reference_reason': 'no small compatible update'}
    references = [actual]
    if previous is None or mask is None:
        return references, metadata
    edit = mask.convert('L').resize(ink.size, Image.Resampling.NEAREST)
    edit_fraction = float(np.mean(np.asarray(edit) > 0))
    metadata['edit_fraction'] = round(edit_fraction, 4)
    if not 0 < edit_fraction <= 0.12:
        metadata['previous_reference_reason'] = 'structural or global update; old hypothesis discarded'
        return references, metadata
    actual_pixels = np.asarray(_on_white(ink).convert('L')) < 200
    prior = _on_white(previous).resize(ink.size, Image.Resampling.LANCZOS)
    prior_pixels = np.asarray(prior.convert('L')) < 200
    if not actual_pixels.any() or not prior_pixels.any():
        return references, metadata
    distances = cv2.distanceTransform((~prior_pixels).astype(np.uint8), cv2.DIST_L2, 5)
    incompatible = float(np.mean(distances[actual_pixels] > 6))
    metadata['previous_ink_mismatch_fraction'] = round(incompatible, 4)
    if incompatible > 0.05:
        metadata['previous_reference_reason'] = 'current ink disagrees with old hypothesis'
        return references, metadata
    # Remove changed territory from the optional reference. Its remaining
    # content can still be changed by the model; no pixels are locked to it.
    prior = Image.composite(Image.new('RGB', prior.size, 'white'), prior, edit)
    references.append(prior.resize((size, size), Image.Resampling.LANCZOS))
    metadata.update(reference_count=2, previous_reference_used=True,
                    previous_reference_reason='small compatible update; prior is editable context only')
    return references, metadata


class KleinModelEngine(ModelEngine):
    supports_correction = True
    def __init__(self, root: Path):
        # Reuse project isolation, locks/state helpers and cancellation, without
        # inheriting SDXL parameter limits or checking any SDXL assets.
        self.root = Path(root).resolve()
        self.model_dir = Path(os.environ.get('DRAWINGLAB_MODEL_DIR', str(self.root / 'models'))).resolve()
        self.size = int(_number('DRAWINGLAB_KLEIN_SIZE', 768, 512, 1024))
        if self.size % 64:
            raise ValueError('DRAWINGLAB_KLEIN_SIZE must be divisible by 64')
        self.steps = int(_number('DRAWINGLAB_KLEIN_STEPS', 4, 1, 8))
        self.guidance_scale = 1.0
        self.cpu_offload = os.environ.get('DRAWINGLAB_CPU_OFFLOAD', '1').lower() not in ('0', 'false', 'no')
        self.device = os.environ.get('DRAWINGLAB_DEVICE', 'cuda')
        if self.device not in ('cuda', 'cpu'):
            raise ValueError('DRAWINGLAB_DEVICE must be cuda or cpu')
        self._serial = threading.Lock()
        self._state_lock = threading.Lock()
        self._pipe = None
        self._torch = None
        self._versions = {}
        self._encoded_prompt_key = None
        self._encoded_prompt = None
        self._state = {'busy': False, 'loaded': False, 'phase': 'idle', 'step': 0,
                       'total_steps': self.steps, 'last_error': None,
                       'message': 'Klein 模型将在首次补全时从项目加载。'}
        self._configure_environment()

    def _missing_files(self) -> list[str]:
        missing = []
        for filename, expected, _ in KLEIN_FILES:
            path = self.model_dir / KLEIN_DIRECTORY / filename
            if not path.is_file() or path.stat().st_size != expected:
                missing.append(f'{KLEIN_DIRECTORY}/{filename}')
        return missing

    def status(self) -> dict:
        missing = self._missing_files()
        dependencies = [name for name in ('torch', 'diffusers', 'transformers', 'accelerate', 'safetensors')
                        if importlib.util.find_spec(name) is None]
        try:
            version = importlib.metadata.version('diffusers')
            if tuple(int(part) for part in version.split('.')[:2]) < (0, 37):
                dependencies.append('diffusers with Flux2KleinPipeline (>=0.37)')
        except (ValueError, importlib.metadata.PackageNotFoundError):
            pass
        with self._state_lock:
            result = dict(self._state)
        result.update(ready=not missing and not dependencies, models_present=not missing,
                      missing_files=missing, missing_dependencies=dependencies,
                      model=KLEIN_MODEL, model_revision=KLEIN_REVISION, engine='klein',
                      size=self.size, steps=self.steps, device=self.device,
                      cpu_offload=self.cpu_offload, offline=True,
                      expected_model_bytes=sum(spec[1] for spec in KLEIN_FILES))
        if missing:
            result['message'] = f'Klein 模型尚未下载完整（缺 {len(missing)} 个文件）。运行 scripts/download_klein.py。'
        elif dependencies:
            result['message'] = 'Klein 推理依赖尚未就绪：' + ', '.join(dependencies)
        return result

    def _ensure_loaded(self, cancel: Callable[[], bool]) -> float:
        if self._pipe is not None:
            return 0.0
        started = time.monotonic()
        self._check_cancel(cancel)
        current = self.status()
        if not current['ready']:
            raise ModelUnavailableError(current['message'])
        self._update(phase='loading', message='正在加载本地 Klein 图像编辑模型，首次加载需要一些时间。')
        try:
            import torch
            import diffusers
            import transformers
            from diffusers import Flux2KleinPipeline
        except (ImportError, OSError) as exc:
            raise ModelUnavailableError(f'无法导入项目中的 Klein 推理环境：{exc}') from exc
        self._torch = torch
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ModelUnavailableError('没有可用的 CUDA GPU；Klein 不会自动切换到 CPU。')
        if self.device == 'cuda' and not torch.cuda.is_bf16_supported():
            raise ModelUnavailableError('当前 GPU 不支持 Klein 所需的 BF16 推理。')
        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        pipe = Flux2KleinPipeline.from_pretrained(
            str(self.model_dir / KLEIN_DIRECTORY), torch_dtype=dtype,
            use_safetensors=True, local_files_only=True,
        )
        self._check_cancel(cancel)
        pipe.vae.enable_tiling()
        if hasattr(pipe.vae, 'enable_slicing'):
            pipe.vae.enable_slicing()
        pipe.set_progress_bar_config(disable=True)
        if self.device == 'cuda' and self.cpu_offload:
            pipe.enable_model_cpu_offload(gpu_id=0)
        else:
            pipe.to(self.device)
        self._pipe = pipe
        self._versions = {'torch': torch.__version__, 'diffusers': diffusers.__version__,
                          'transformers': transformers.__version__, 'dtype': str(dtype),
                          'gpu': torch.cuda.get_device_name(0) if self.device == 'cuda' else None}
        self._update(loaded=True, phase='ready', message='Klein 本地模型已加载。')
        return time.monotonic() - started

    def generate_correction(self, ink, prompt, seed, cancel, region=None):
        return self.generate(ink, prompt, None, None, seed, cancel, task='correction', region=region)

    def generate(self, ink: Image.Image, prompt: str, previous: Image.Image | None,
                 mask: Image.Image | None, seed: int, cancel: Callable[[], bool],
                 task: str = 'completion', region: dict | None = None) -> tuple[Image.Image, dict]:
        if not isinstance(ink, Image.Image):
            raise TypeError('ink 必须是 PIL.Image')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('请先填写要画的内容。')
        if previous is not None and not isinstance(previous, Image.Image):
            raise TypeError('previous 必须是 PIL.Image 或 None')
        if mask is not None and not isinstance(mask, Image.Image):
            raise TypeError('mask 必须是 PIL.Image 或 None')
        if not np.any(np.asarray(_on_white(ink).convert('L')) < 200):
            raise ValueError('画布没有可见笔迹，请先画几笔。')
        self._check_cancel(cancel)
        queued_at = time.monotonic()
        while not self._serial.acquire(timeout=0.1):
            self._check_cancel(cancel)
        started = time.monotonic()
        self._update(busy=True, phase='preparing', step=0, total_steps=self.steps, last_error=None)
        try:
            self._check_cancel(cancel)
            references, reference_info = prepare_references(ink, previous, mask, self.size)
            effective_prompt = (
                'Complete the unfinished drawing in image 1. Keep its existing contours and features at their exact '
                'positions and scale, including the face, eyes, mouth, body and limbs that are already drawn. '
                'Draw only missing parts around those marks. Match image 1 in proportions, angularity, curvature, '
                'simplicity and amount of detail. Return a flat digital black line drawing on a pure white background, '
                'without photographs, paper texture, shadows, borders or drawing tools. '
                'The white areas are unfinished space. Make one coherent drawing without duplicate contours or features. '
                'The intended subject is: ' + prompt.strip()
            )
            if task == 'correction':
                effective_prompt = correction_prompt(prompt, region)
            if len(references) > 1:
                effective_prompt += (' Image 2 is an earlier optional composition idea with the changed region removed. '
                                     'Use it only where it agrees with image 1. Image 1 determines all geometry and style; '
                                     'change any conflicting part of image 2.')
            load_seconds = self._ensure_loaded(cancel)
            self._check_cancel(cancel)
            pipe, torch = self._pipe, self._torch
            formatted = pipe.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': effective_prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            token_count = len(pipe.tokenizer(formatted, truncation=False)['input_ids'])
            prompt_warning = '描述过长，模型只读取前 512 个 token；可以缩短角色描述。' if token_count > 512 else None
            completed_steps = 0

            def on_step_end(pipeline, step_index, timestep, callback_kwargs):
                nonlocal completed_steps
                self._check_cancel(cancel)
                completed_steps = step_index + 1
                self._update(phase='denoising', step=completed_steps,
                             message=f'正在根据当前笔迹补全：{completed_steps}/{self.steps} 步。')
                return callback_kwargs

            self._update(phase='denoising', message='正在理解当前线稿并补全。')
            actual_seed = int(seed) % (2**63 - 1)
            generator = torch.Generator(device='cpu').manual_seed(actual_seed)
            inference_started = time.monotonic()
            if self.device == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                # Only text encoding is cached. Image references and all four
                # denoising steps are recomputed for every new drawing revision.
                prompt_cached = self._encoded_prompt_key == effective_prompt and self._encoded_prompt is not None
                if prompt_cached:
                    prompt_embeds = self._encoded_prompt.to(self.device)
                else:
                    prompt_embeds, _ = pipe.encode_prompt(effective_prompt, device=self.device, max_sequence_length=512)
                    self._encoded_prompt = prompt_embeds.detach().to('cpu').clone()
                    self._encoded_prompt_key = effective_prompt
                self._check_cancel(cancel)
                result = pipe(
                    image=references, prompt_embeds=prompt_embeds, height=self.size, width=self.size,
                    num_inference_steps=self.steps, guidance_scale=1.0, generator=generator,
                    num_images_per_prompt=1, output_type='pil', max_sequence_length=512,
                    callback_on_step_end=on_step_end, callback_on_step_end_tensor_inputs=['latents'],
                )
            self._check_cancel(cancel)
            candidate = result.images[0].convert('RGB').resize(ink.size, Image.Resampling.LANCZOS)
            metadata = {
                'model': KLEIN_MODEL, 'model_revision': KLEIN_REVISION, 'engine': 'klein',
                'pipeline': type(pipe).__name__, 'mode': 'image_conditioned_' + task,
                'seed': actual_seed, 'size': [self.size, self.size], 'output_size': list(ink.size),
                'requested_steps': self.steps, 'completed_steps': completed_steps, 'guidance_scale': 1.0,
                'scheduler': type(pipe.scheduler).__name__, 'prompt': effective_prompt,
                'prompt_warning': prompt_warning, 'prompt_token_count': token_count,
                'prompt_truncated': token_count > 512, 'queue_seconds': round(started - queued_at, 3),
                'prompt_encoding_cached': prompt_cached,
                'load_seconds': round(load_seconds, 3),
                'inference_seconds': round(time.monotonic() - inference_started, 3),
                'duration_seconds': round(time.monotonic() - started, 3),
                'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated() if self.device == 'cuda' else None,
                'device': self.device, 'cpu_offload': self.cpu_offload and self.device == 'cuda',
                'offline': True, 'generated_by_ai': True, 'outside_mask_preserved': False,
                'raw_alignment_validated': False, **reference_info, **self._versions,
            }
            self._update(phase='ready', message='新的整幅补全已生成，正在检查与原稿的贴合。', last_error=None)
            return candidate, metadata
        except GenerationCancelled:
            self._update(phase='cancelled', message='已取消旧线稿的补全。')
            raise
        except Exception as exc:
            self._update(phase='error', last_error=str(exc), message=f'Klein 生成失败：{exc}')
            if self._torch is not None and isinstance(exc, self._torch.cuda.OutOfMemoryError):
                raise ModelUnavailableError('Klein 显存不足；保持 CPU_OFFLOAD=1，并降低 DRAWINGLAB_KLEIN_SIZE 后重试。') from exc
            raise
        finally:
            if self._pipe is not None and self.cpu_offload and self.device == 'cuda':
                try:
                    self._pipe.maybe_free_model_hooks()
                except Exception:
                    pass
            self._update(busy=False)
            self._serial.release()
