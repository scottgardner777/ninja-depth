"""Monocular depth estimation — Depth Anything V2 + DPT-Hybrid (MiDaS).

Both models use HuggingFace transformers depth-estimation pipeline for
reliable loading and inference.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("ninja-depth")

MODELS = {
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "midas": "Intel/dpt-hybrid-midas",
}

VALID_MODELS = set(MODELS)


class DepthEstimator:
    """Lazy-loaded singleton for depth estimation models."""

    def __init__(self) -> None:
        self._pipelines: dict = {}
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self, model: str):
        if model in self._pipelines:
            return
        hf_id = MODELS[model]
        log.info("Loading %s (%s) …", model, hf_id)
        from transformers import pipeline

        self._pipelines[model] = pipeline(
            "depth-estimation",
            model=hf_id,
            device=self._device,
        )
        log.info("%s ready (%s)", model, self._device)

    def estimate(
        self, image: Image.Image, model: str = "depth-anything-v2"
    ) -> np.ndarray:
        """Return HxW float32 depth array normalized 0-1 (1 = far)."""
        if model not in VALID_MODELS:
            raise ValueError(f"Unknown model: {model}")

        with self._lock:
            self._load(model)
            pipe = self._pipelines[model]

        result = pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)

        # Normalize to 0-1
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # DPT-Hybrid/MiDaS returns inverse depth (close = high value).
        # Invert so convention is 0 = near, 1 = far.
        if model == "midas":
            depth = 1.0 - depth

        return depth

    def loaded_models(self) -> list[str]:
        """Return list of currently loaded model names."""
        return list(self._pipelines.keys())


# Module-level singleton
estimator = DepthEstimator()
