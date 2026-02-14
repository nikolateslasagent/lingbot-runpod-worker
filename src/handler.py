"""
RunPod Serverless handler for LingBot World (Wan2.2 Image-to-Video).
Downloads model on first cold start, then caches.
"""

import os
import sys
import time
import base64
import tempfile
import shutil

# Add lingbot-world source to Python path BEFORE any imports
sys.path.insert(0, "/app/lingbot-world")

import runpod

MODEL_REPO = "robbyant/lingbot-world-base-cam"
# Use /runpod-volume if available (persistent), else container disk
_vol = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp"
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_vol}/model")
DEVICE = 0  # GPU device ID

# Global model reference
wan_i2v = None


def download_model():
    """Download model from HuggingFace if not already cached."""
    # Log disk space
    for path in ["/", "/tmp", "/runpod-volume"]:
        try:
            usage = shutil.disk_usage(path)
            print(f"[disk] {path}: {usage.free/1024**3:.1f}GB free / {usage.total/1024**3:.1f}GB total")
        except Exception:
            pass

    marker = os.path.join(MODEL_DIR, ".download_complete")
    if os.path.exists(marker):
        print(f"[init] Model already cached at {MODEL_DIR}")
        return

    print(f"[init] Downloading model {MODEL_REPO} to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    os.environ["HF_HOME"] = os.path.join(os.path.dirname(MODEL_DIR), ".hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    with open(marker, "w") as f:
        f.write("ok")
    print("[init] Model download complete.")


def load_model():
    """Load the WanI2V pipeline using the actual wan API."""
    global wan_i2v

    if wan_i2v is not None:
        return

    download_model()

    print("[init] Loading WanI2V model...")
    t0 = time.time()

    import wan
    from wan.configs import WAN_CONFIGS

    cfg = WAN_CONFIGS["i2v-A14B"]

    wan_i2v = wan.WanI2V(
        config=cfg,
        checkpoint_dir=MODEL_DIR,
        device_id=DEVICE,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,  # A100 80GB has enough VRAM
        convert_model_dtype=True,
    )

    print(f"[init] Model loaded in {time.time() - t0:.1f}s")


def handler(job):
    """Handle a generation request."""
    load_model()

    inp = job["input"]
    prompt = inp.get("prompt", "A beautiful landscape")
    image_b64 = inp.get("image_base64")
    size = inp.get("size", "480*832")
    frame_num = int(inp.get("frame_num", 81))
    seed = int(inp.get("seed", 42))

    import torch
    from PIL import Image
    import io
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.utils import save_video

    # Parse size
    h, w = [int(x) for x in size.split("*")]

    # Load reference image (REQUIRED for image-to-video)
    if not image_b64:
        # Generate a simple gradient image as placeholder
        img = Image.new("RGB", (w, h), (135, 206, 235))  # sky blue
        print("[gen] No input image provided, using placeholder")
    else:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = img.resize((w, h))

    print(f"[gen] prompt={prompt[:80]}... size={w}x{h} frames={frame_num} seed={seed}")
    t0 = time.time()

    # Generate video using the actual wan API
    video = wan_i2v.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS.get(size, h * w),
        frame_num=frame_num,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=20,
        guide_scale=5.0,
        seed=seed,
        offload_model=False,
    )

    duration = time.time() - t0
    print(f"[gen] Generated in {duration:.1f}s")

    # Save to temp MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    save_video(video, tmp_path, fps=16, quality=8)

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    os.unlink(tmp_path)

    return {
        "video_base64": video_b64,
        "duration_seconds": round(duration, 1),
        "frames": frame_num,
        "size": f"{w}x{h}",
    }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
