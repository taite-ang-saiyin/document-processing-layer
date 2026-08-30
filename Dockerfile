FROM python:3.10-slim

ARG DEBIAN_MIRROR=https://mirror.rackspace.com/debian
ARG DEBIAN_SECURITY_MIRROR=https://mirror.rackspace.com/debian-security

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and PyTorch
RUN sed -E -i \
      -e "s|https?://deb.debian.org/debian-security|${DEBIAN_SECURITY_MIRROR}|g" \
      -e "s|https?://deb.debian.org/debian|${DEBIAN_MIRROR}|g" \
      /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies.
# Install a CPU-only PyTorch wheel first (no CUDA) to keep the image small and
# the build fast; the app runs on DEVICE=cpu. Installing torch before the rest
# of requirements satisfies the "torch>=2.2.0" pin from the requirements file.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir paddlepaddle==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install --no-cache-dir -r requirements.txt

# Keep PDF rasterization outside the Python/torch dependency layer so adding
# document support does not cause pip to resolve a new CUDA-enabled torch build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy application source code
COPY . .

# Expose FastAPI server port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
