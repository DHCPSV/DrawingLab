# Third-party components and publication scope

DrawingLab's original checkout contained no root software license. This release does not invent one. No additional permission to reuse, redistribute or commercialize the application is granted here. Request permission from the author where needed.

Dependencies and model downloads retain their own licenses. They are installed/downloaded separately, not vendored or relicensed by this repository.

| Component | Source and scope |
| --- | --- |
| FLUX.2 klein 4B | [Official model](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), pinned revision in `drawinglab/klein_manifest.py`. The downloaded LICENSE.md identifies Apache License 2.0; consult the upstream license and model card. Weights are excluded from Git. |
| Diffusers, Transformers, Accelerate | [Hugging Face](https://github.com/huggingface); installed as pinned Python dependencies. |
| PyTorch | [Official project](https://pytorch.org/); separate CUDA wheel installation. |
| CounterfeitXL, anime-painter | [gsdf/CounterfeitXL](https://huggingface.co/gsdf/CounterfeitXL), [xinsir/anime-painter](https://huggingface.co/xinsir/anime-painter); optional legacy integration. Review the respective model conditions before use. No weights are redistributed. |
| DoodleAssist | [MarkMoHR/DoodleAssist](https://github.com/MarkMoHR/DoodleAssist); experimental integration references a pinned upstream implementation. Upstream source, adapters and base-model weights are excluded. The prior downloader records no explicit root upstream license; do not infer permission to redistribute those assets. Not release-validated. |
| FastAPI, Uvicorn, Pydantic, Pillow, NumPy, OpenCV, HTTPX, pytest | Installed from PyPI; their own package notices and licenses apply. |
| Playwright | Optional browser test dependency from the npm registry; no browsers or node_modules bundled. |

The included cat sketch was newly drawn for this release through the application UI, without using existing personal artwork, photos, identifiable people or a third-party character. The output is the actual local Klein result for that sketch; it is illustrative, not a guarantee of originality or model quality for arbitrary inputs. The model output has not been manually enhanced. Publication does not assert exclusive rights in AI-generated imagery.
