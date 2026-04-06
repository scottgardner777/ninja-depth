"""Monocular depth estimation — Depth Anything V2 + MiDaS v3.1."""

from __future__ import annotations

import logging
import threading

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("ninja-depth")

MODELS = {
    "depth-anything-v2": "depth-anything/Depth-Anything-V2-Small-hf",
    "midas": "intel-isl/MiDaS",
}

VALID_MODELS = set(MODELS)


class DepthEstimator:
    """Lazy-loaded singleton for depth estimation models."""

    def __init__(self) -> None:
        self._da2_pipe = None
        self._midas_model = None
        self._midas_transform = None
        self._lock = threading.Lock()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    # -- lazy loaders ---------------------------------------------------

    def _load_da2(self):
        if self._da2_pipe is not None:
            return
        log.info("Loading Depth Anything V2 Small …")
        from transformers import pipeline

        self._da2_pipe = pipeline(
            "depth-estimation",
            model=MODELS["depth-anything-v2"],
            device=self._device,
        )
        log.info("Depth Anything V2 ready (%s)", self._device)

    def _load_midas(self):
        if self._midas_model is not None:
            return
        log.info("Loading MiDaS v3.1 Small …")
        self._midas_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
        )
        self._midas_model.to(self._device).eval()
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self._midas_transform = midas_transforms.small_transform
        log.info("MiDaS v3.1 ready (%s)", self._device)

    # -- public API -----------------------------------------------------

    def estimate(
        self, image: Image.Image, model: str = "depth-anything-v2"
    ) -> np.ndarray:
        """Return HxW float32 depth array normalized 0-1 (1 = far).

        Uses the specified model backend.
        """
        with self._lock:
            if model == "depth-anything-v2":
                return self._estimate_da2(image)
            elif model == "midas":
                return self._estimate_midas(image)
            else:
                raise ValueError(f"Unknown model: {model}")

    def _estimate_da2(self, image: Image.Image) -> np.ndarray:
        self._load_da2()
        result = self._da2_pipe(image)
        depth = np.array(result["depth"], dtype=np.float32)
        # Normalize to 0-1
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth

    def _estimate_midas(self, image: Image.Image) -> np.ndarray:
        self._load_midas()
        img_np = np.array(image.convert("RGB"))
        input_batch = self._midas_transform(img_np).to(self._device)
        with torch.no_grad():
            prediction = self._midas_model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_np.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        # MiDaS returns inverse depth (close = high), invert so 1 = far
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            depth = 1.0 - (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
        return depth

    def loaded_models(self) -> list[str]:
        """Return list of currently loaded model names."""
        loaded = []
        if self._da2_pipe is not None:
            loaded.append("depth-anything-v2")
        if self._midas_model is not None:
            loaded.append("midas")
        return loaded


# Module-level singleton
estimator = DepthEstimator()
