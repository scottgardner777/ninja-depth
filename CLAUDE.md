# ninja-depth — Monocular Depth Estimation API

## What This Is

Headless API service that takes a single image and returns either a grayscale depth map (PNG) or a textured 3D mesh (GLB). Part of the ninja.ing ecosystem.

## Stack

- **FastAPI** on port 18025
- **Depth Anything V2** (default) and **MiDaS v3.1** (switchable)
- **trimesh** for GLB mesh generation
- No Neo4j, no UI — API-only service

## Key Files

| File | Purpose |
|------|---------|
| `depth_app.py` | FastAPI application with /depth and /mesh endpoints |
| `core/depth.py` | Depth estimation (Depth Anything V2 + MiDaS) |
| `core/mesh.py` | Depth map to textured GLB mesh conversion |

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/health` | No | Health check |
| `GET` | `/` | No | Platform info |
| `GET` | `/models` | No | List available models |
| `POST` | `/depth` | Yes | Image to depth map PNG |
| `POST` | `/mesh` | Yes | Image to textured GLB mesh |

## Auth

Supports both:
- **Cookie**: `ninja-depth-token` JWT (ecosystem SSO)
- **API Key**: `X-API-Key` header (external consumers)

## Running

```bash
# Docker
docker compose up

# Local dev
uvicorn depth_app:app --host 0.0.0.0 --port 18025

# Test
curl -X POST http://localhost:18025/depth -H "X-API-Key: <key>" -F "file=@image.jpg" -o depth.png
curl -X POST http://localhost:18025/mesh -H "X-API-Key: <key>" -F "file=@image.jpg" -o scene.glb
```

## Ecosystem

- Port: 18025
- Cookie: ninja-depth-token
- Local dir: C:\ninja-depth
- Git: scottgardner777/ninja-depth
