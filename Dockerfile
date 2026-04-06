FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/scottgardner777/ninja-depth"
LABEL org.opencontainers.image.description="Monocular depth estimation API — depth maps and 3D meshes"
LABEL org.opencontainers.image.licenses="All rights reserved"

WORKDIR /app

# Install torch CPU-only first (smaller image, ~800MB vs ~2GB)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Skip torch/torchvision since already installed above
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/cache

EXPOSE 18025
CMD ["uvicorn", "depth_app:app", "--host", "0.0.0.0", "--port", "18025", "--workers", "2"]
