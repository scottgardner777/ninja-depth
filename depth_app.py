"""ninja-depth — Monocular depth estimation API.

POST /depth → grayscale depth map PNG
POST /mesh  → textured GLB mesh for Three.js / VR
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from jose import JWTError, jwt
from PIL import Image

from core import cache
from core.depth import VALID_MODELS, estimator
from core.mesh import depth_to_glb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_SECRET = os.getenv("JWT_SECRET", "")
API_KEY = os.getenv("API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000").split(",")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "depth-anything-v2")
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "4096"))
DEFAULT_MESH_RES = int(os.getenv("DEFAULT_MESH_RESOLUTION", "512"))
MAX_MESH_RES = int(os.getenv("MAX_MESH_RESOLUTION", "1024"))
JWT_ALGORITHM = "HS256"
COOKIE_NAME = "ninja-depth-token"

STATIC_DIR = Path(__file__).parent / "static"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("ninja-depth")

# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ninja-depth starting — models loaded on first request")
    log.info("Cache: %s", cache.stats())
    yield
    log.info("ninja-depth shutting down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ninja-depth",
    description="Monocular depth estimation — depth maps & 3D meshes",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


def _verify_jwt(token: str) -> dict | None:
    if not JWT_SECRET:
        return None
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None


def require_auth(request: Request):
    """Authenticate via JWT cookie, Bearer token, or X-API-Key header."""
    # 1. API key
    api_key = request.headers.get("X-API-Key")
    if api_key and API_KEY and api_key == API_KEY:
        return {"sub": "api-key", "role": "api"}

    # 2. Bearer token
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        claims = _verify_jwt(auth_header[7:])
        if claims:
            return claims

    # 3. Cookie
    token = request.cookies.get(COOKIE_NAME)
    if token:
        claims = _verify_jwt(token)
        if claims:
            return claims

    # 4. Dev mode — no secrets configured, allow all
    if not JWT_SECRET and not API_KEY:
        return {"sub": "dev", "role": "dev"}

    raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _read_image(file: UploadFile) -> Image.Image:
    """Read and validate an uploaded image."""
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    data = await file.read()
    if len(data) > 50 * 1024 * 1024:  # 50 MB max
        raise HTTPException(status_code=400, detail="File too large (50 MB max)")

    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    w, h = img.size
    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({w}x{h}). Max dimension: {MAX_IMAGE_DIM}px",
        )
    return img


def _validate_model(model: str) -> str:
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{model}'. Available: {sorted(VALID_MODELS)}",
        )
    return model


def _depth_to_png(depth: np.ndarray) -> bytes:
    """Convert float32 depth array to 8-bit grayscale PNG."""
    depth_u8 = (depth * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(depth_u8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _estimate_cached(img: Image.Image, model: str) -> tuple[np.ndarray, bool]:
    """Estimate depth with cache. Returns (depth_array, was_cached)."""
    cached = cache.get(img, model)
    if cached is not None:
        return cached, True
    depth = estimator.estimate(img, model=model)
    cache.put(img, model, depth)
    return depth, False


# ---------------------------------------------------------------------------
# UI route
# ---------------------------------------------------------------------------


@app.get("/ui")
def ui_page():
    """Serve the depth estimation UI."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# Public API endpoints (no auth)
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "app": "ninja-depth",
        "version": "1.0.0",
        "description": "Monocular depth estimation — depth maps & 3D meshes",
        "endpoints": ["/health", "/models", "/depth", "/mesh", "/ui"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_available": sorted(VALID_MODELS),
        "models_loaded": estimator.loaded_models(),
        "default_model": DEFAULT_MODEL,
        "cache": cache.stats(),
    }


@app.get("/models")
def models():
    return {
        "models": [
            {
                "id": "depth-anything-v2",
                "name": "Depth Anything V2 Small",
                "params": "24M",
                "description": "Fast and accurate monocular depth estimation",
            },
            {
                "id": "midas",
                "name": "DPT-Hybrid MiDaS",
                "params": "123M",
                "description": "Intel DPT-Hybrid monocular depth estimation",
            },
        ],
        "default": DEFAULT_MODEL,
    }


# ---------------------------------------------------------------------------
# Authenticated endpoints
# ---------------------------------------------------------------------------


@app.post("/depth")
async def depth_endpoint(
    file: UploadFile = File(...),
    model: str = Query(default=None),
    _auth: dict = Depends(require_auth),
):
    """Upload image, receive grayscale depth map PNG."""
    model = model or DEFAULT_MODEL
    _validate_model(model)

    img = await _read_image(file)
    log.info("Depth request: %dx%d, model=%s", img.width, img.height, model)

    t0 = time.perf_counter()
    depth, was_cached = _estimate_cached(img, model)
    elapsed = time.perf_counter() - t0
    log.info("Depth estimated in %.2fs (cached=%s)", elapsed, was_cached)

    png_bytes = _depth_to_png(depth)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": "attachment; filename=depth.png",
            "X-Depth-Model": model,
            "X-Processing-Time": f"{elapsed:.3f}s",
            "X-Cache-Hit": str(was_cached).lower(),
        },
    )


@app.post("/mesh")
async def mesh_endpoint(
    file: UploadFile = File(...),
    model: str = Query(default=None),
    resolution: int = Query(default=None, ge=32, le=1024),
    _auth: dict = Depends(require_auth),
):
    """Upload image, receive textured GLB mesh."""
    model = model or DEFAULT_MODEL
    resolution = resolution or DEFAULT_MESH_RES
    resolution = min(resolution, MAX_MESH_RES)
    _validate_model(model)

    img = await _read_image(file)
    log.info(
        "Mesh request: %dx%d, model=%s, resolution=%d",
        img.width, img.height, model, resolution,
    )

    t0 = time.perf_counter()
    depth, was_cached = _estimate_cached(img, model)
    t1 = time.perf_counter()
    glb_bytes = depth_to_glb(img, depth, resolution=resolution)
    t2 = time.perf_counter()

    log.info(
        "Mesh generated: depth=%.2fs (cached=%s), mesh=%.2fs",
        t1 - t0, was_cached, t2 - t1,
    )

    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={
            "Content-Disposition": "attachment; filename=scene.glb",
            "X-Depth-Model": model,
            "X-Mesh-Resolution": str(resolution),
            "X-Processing-Time": f"{t2 - t0:.3f}s",
            "X-Cache-Hit": str(was_cached).lower(),
        },
    )


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache", dependencies=[Depends(require_auth)])
def cache_clear():
    n = cache.clear()
    return {"cleared": n}


# ---------------------------------------------------------------------------
# Static files (must be last — catch-all mount)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
