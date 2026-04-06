"""File-based depth result cache.

Hashes image bytes + model name → cached numpy depth array on disk.
Avoids re-running inference for duplicate/repeated images.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("ninja-depth")

CACHE_DIR = Path(os.getenv("CACHE_DIR", "cache"))
MAX_CACHE_MB = int(os.getenv("MAX_CACHE_MB", "500"))


def _ensure_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(image: Image.Image, model: str) -> str:
    """SHA256 of image bytes + model name."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    h = hashlib.sha256(buf.getvalue())
    h.update(model.encode())
    return h.hexdigest()


def get(image: Image.Image, model: str) -> np.ndarray | None:
    """Return cached depth array or None."""
    key = _cache_key(image, model)
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        try:
            arr = np.load(path)
            log.info("Cache hit: %s (%s)", key[:12], model)
            return arr
        except Exception:
            path.unlink(missing_ok=True)
    return None


def put(image: Image.Image, model: str, depth: np.ndarray) -> None:
    """Store depth array in cache."""
    _ensure_dir()
    _evict_if_needed()
    key = _cache_key(image, model)
    path = CACHE_DIR / f"{key}.npy"
    try:
        np.save(path, depth)
        log.info("Cached: %s (%s)", key[:12], model)
    except Exception as e:
        log.warning("Cache write failed: %s", e)


def _evict_if_needed():
    """Remove oldest cache files if total size exceeds limit."""
    if not CACHE_DIR.exists():
        return
    files = sorted(CACHE_DIR.glob("*.npy"), key=lambda p: p.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    limit = MAX_CACHE_MB * 1024 * 1024
    while total > limit and files:
        old = files.pop(0)
        total -= old.stat().st_size
        old.unlink(missing_ok=True)
        log.info("Evicted cache: %s", old.name)


def clear() -> int:
    """Clear all cached files. Returns count removed."""
    if not CACHE_DIR.exists():
        return 0
    files = list(CACHE_DIR.glob("*.npy"))
    for f in files:
        f.unlink(missing_ok=True)
    return len(files)


def stats() -> dict:
    """Return cache statistics."""
    if not CACHE_DIR.exists():
        return {"entries": 0, "size_mb": 0.0}
    files = list(CACHE_DIR.glob("*.npy"))
    total = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "size_mb": round(total / 1024 / 1024, 2),
        "max_mb": MAX_CACHE_MB,
    }
