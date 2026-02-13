FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Python deps
RUN pip install --no-cache-dir \
    runpod \
    "huggingface_hub[cli]" \
    diffusers>=0.31.0 \
    "transformers>=4.49.0,<=4.51.3" \
    accelerate>=1.1.1 \
    tokenizers>=0.20.3 \
    tqdm imageio[ffmpeg] imageio-ffmpeg \
    easydict ftfy \
    "numpy>=1.23.5,<2" scipy einops Pillow \
    opencv-python-headless>=4.9.0.80

RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Clone lingbot-world source code (not the model, just code)
RUN git clone --depth 1 https://github.com/Robbyant/lingbot-world.git /app/lingbot-world

# Copy handler
COPY src/handler.py /app/handler.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "handler.py"]
