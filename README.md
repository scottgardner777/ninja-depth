# ninja-depth

**Turn any image into a depth map or 3D mesh. One command to run.**

Upload a photo, AI-generated scene, or any image — get back a grayscale depth map (PNG) or a textured 3D model (GLB) you can open in Blender, Three.js, or any VR viewer.

Powered by [Depth Anything V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) and [DPT-MiDaS](https://huggingface.co/Intel/dpt-hybrid-midas).

---

## Quick Start (Docker)

```bash
docker run -p 18025:18025 ghcr.io/scottgardner777/ninja-depth:latest
```

That's it. Open **http://localhost:18025/ui** in your browser.

### Or with docker compose:

```bash
git clone https://github.com/scottgardner777/ninja-depth.git
cd ninja-depth
docker compose up
```

Open **http://localhost:18025/ui**

---

## How to Use

### Browser UI

Go to **http://localhost:18025/ui** and:

1. Drag and drop any image (JPEG, PNG, WebP)
2. Pick a model — **Depth Anything V2** is fast, **MiDaS** is an alternative
3. Adjust mesh resolution if you want (higher = more detail)
4. See the depth map next to your original image
5. Spin the 3D mesh around in the viewer
6. Download the depth map (PNG) or 3D mesh (GLB)

### API

```bash
# Get a depth map
curl -X POST http://localhost:18025/depth \
  -F "file=@your-image.jpg" \
  -o depth.png

# Get a 3D mesh
curl -X POST http://localhost:18025/mesh \
  -F "file=@your-image.jpg" \
  -o scene.glb

# Use MiDaS instead of the default model
curl -X POST http://localhost:18025/depth?model=midas \
  -F "file=@your-image.jpg" \
  -o depth.png

# Set mesh resolution (32-1024, default 512)
curl -X POST http://localhost:18025/mesh?resolution=256 \
  -F "file=@your-image.jpg" \
  -o scene.glb
```

### Python

```python
import requests

# Depth map
with open("photo.jpg", "rb") as f:
    r = requests.post("http://localhost:18025/depth", files={"file": f})
with open("depth.png", "wb") as f:
    f.write(r.content)

# 3D mesh
with open("photo.jpg", "rb") as f:
    r = requests.post("http://localhost:18025/mesh", files={"file": f})
with open("scene.glb", "wb") as f:
    f.write(r.content)
```

### JavaScript / Three.js

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

// Get depth map
const depthRes = await fetch("http://localhost:18025/depth", { method: "POST", body: form });
const depthBlob = await depthRes.blob();

// Get GLB mesh and load into Three.js
const meshRes = await fetch("http://localhost:18025/mesh", { method: "POST", body: form });
const meshBlob = await meshRes.blob();
const url = URL.createObjectURL(meshBlob);
new GLTFLoader().load(url, (gltf) => scene.add(gltf.scene));
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check + loaded models |
| `GET` | `/models` | List available models |
| `GET` | `/ui` | Browser UI |
| `POST` | `/depth` | Image → depth map PNG |
| `POST` | `/mesh` | Image → textured GLB mesh |

### Query parameters

| Param | Default | Description |
|-------|---------|-------------|
| `model` | `depth-anything-v2` | Model to use (`depth-anything-v2` or `midas`) |
| `resolution` | `512` | Mesh vertex grid size (32-1024, `/mesh` only) |

### Response headers

| Header | Description |
|--------|-------------|
| `X-Depth-Model` | Model used for estimation |
| `X-Processing-Time` | Time taken |
| `X-Cache-Hit` | `true` if result was cached |

---

## Models

| Model | ID | Size | Speed | Notes |
|-------|----|------|-------|-------|
| **Depth Anything V2 Small** | `depth-anything-v2` | 24M params | Fast (~2-5s) | Default. Best quality/speed ratio |
| **DPT-Hybrid MiDaS** | `midas` | 123M params | Slower (~5-15s) | Intel's DPT architecture |

Both produce **relative** depth (near vs far), not absolute metric depth.

---

## What You Get

**Depth map** (`/depth`): 8-bit grayscale PNG, same resolution as input. White = near, black = far.

**3D mesh** (`/mesh`): Binary glTF (.glb) file containing:
- Plane geometry displaced by depth values
- Original image as PBR base color texture
- Proper UV mapping
- Ready for Three.js, Blender, Unity, Unreal, WebXR

---

## Configuration

Set via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `JWT_SECRET` | _(empty)_ | JWT secret for cookie auth. If empty + no API_KEY, **dev mode** (no auth) |
| `API_KEY` | _(empty)_ | Static API key for `X-API-Key` header auth |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `DEFAULT_MODEL` | `depth-anything-v2` | Default model |
| `MAX_IMAGE_DIM` | `4096` | Max image dimension in pixels |
| `MAX_CACHE_MB` | `500` | Max cache size before LRU eviction |

**Dev mode**: If neither `JWT_SECRET` nor `API_KEY` is set, all endpoints are open — perfect for local use.

---

## Limits

- Max image size: 4096 x 4096 pixels
- Max upload: 50 MB
- First request per model is slow (downloads model weights ~100-500MB)
- Subsequent requests are fast (cached in memory)
- Identical images are cached to disk (instant on repeat)

---

## License

Copyright 2026 Scott Gardner. All rights reserved.
