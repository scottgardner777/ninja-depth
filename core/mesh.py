"""Convert depth map + image into a textured GLB mesh."""

from __future__ import annotations

import io
import logging

import numpy as np
import trimesh
from PIL import Image

log = logging.getLogger("ninja-depth")

# Depth displacement scale (world units)
DEPTH_SCALE = 10.0


def depth_to_glb(
    image: Image.Image,
    depth: np.ndarray,
    resolution: int = 512,
) -> bytes:
    """Generate a textured GLB mesh from image + depth map.

    Args:
        image: Original RGB image (used as diffuse texture).
        depth: HxW float32 depth array (0-1, where 1 = far).
        resolution: Vertex grid resolution (NxN). Max 1024.

    Returns:
        GLB file bytes.
    """
    resolution = min(max(resolution, 32), 1024)
    h, w = depth.shape

    # Build vertex grid
    rows = np.linspace(0, 1, resolution)
    cols = np.linspace(0, 1, resolution)
    grid_u, grid_v = np.meshgrid(cols, rows)

    # Sample depth at grid positions
    sample_y = (grid_v * (h - 1)).astype(np.int32)
    sample_x = (grid_u * (w - 1)).astype(np.int32)
    grid_depth = depth[sample_y, sample_x]

    # Vertices: X right, Y up (flip V), Z = -depth (into screen)
    aspect = w / h
    vx = (grid_u - 0.5) * aspect * DEPTH_SCALE
    vy = (0.5 - grid_v) * DEPTH_SCALE
    vz = -grid_depth * DEPTH_SCALE

    vertices = np.stack([vx, vy, vz], axis=-1).reshape(-1, 3)

    # UV coordinates (flip V for OpenGL convention)
    uv = np.stack([grid_u, 1.0 - grid_v], axis=-1).reshape(-1, 2)

    # Build face indices (two triangles per quad)
    faces = []
    for r in range(resolution - 1):
        for c in range(resolution - 1):
            i = r * resolution + c
            # Triangle 1
            faces.append([i, i + resolution, i + 1])
            # Triangle 2
            faces.append([i + 1, i + resolution, i + resolution + 1])
    faces = np.array(faces, dtype=np.int32)

    # Create texture image (resize to power-of-2 for GPU compat)
    tex_size = min(1024, max(w, h))
    tex_img = image.convert("RGB").resize((tex_size, tex_size), Image.LANCZOS)

    # Build trimesh with texture
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=tex_img,
        metallicFactor=0.0,
        roughnessFactor=0.8,
    )
    color_visuals = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=color_visuals,
        process=False,
    )

    # Export to GLB
    glb_bytes = mesh.export(file_type="glb")
    log.info(
        "Generated GLB: %d vertices, %d faces, %.1f KB",
        len(vertices),
        len(faces),
        len(glb_bytes) / 1024,
    )
    return glb_bytes
