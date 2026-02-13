"""
RunPod Serverless handler for LingBot World (Wan2.2 Image-to-Video).
Downloads model on first cold start, then caches in /runpod-volume.
"""

import os
import sys
import time
import base64
import tempfile
import runpod

MODEL_REPO = "robbyant/lingbot-world-base-cam"
# Use /runpod-volume if available (persistent), else /tmp (container disk)
_vol = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp"
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_vol}/model")
DEVICE = "cuda"

# Global model references (loaded once)
pipe = None


def download_model():
    """Download model from HuggingFace if not already cached."""
    import shutil
    
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

    # Set HF cache to same disk to avoid double storage
    os.environ["HF_HOME"] = os.path.join(os.path.dirname(MODEL_DIR), ".hf_cache")
    os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_HOME"]

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
    )

    # Mark as complete
    with open(marker, "w") as f:
        f.write("ok")
    print("[init] Model download complete.")


def load_model():
    """Load the Wan2.2 pipeline."""
    global pipe

    if pipe is not None:
        return

    download_model()

    print("[init] Loading model into GPU...")
    t0 = time.time()

    # Add lingbot-world source to path
    sys.path.insert(0, "/app/lingbot-world")

    import torch
    from wan.configs import WAN_CONFIGS
    from wan.pipelines.pipeline_wan_i2v import WanI2VPipeline

    cfg = WAN_CONFIGS["i2v-14B"]

    pipe = WanI2VPipeline.from_pretrained(
        MODEL_DIR,
        config=cfg,
        device=DEVICE,
        torch_dtype=torch.float16,
        offload_model=True,  # CPU offload to fit in 48GB
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
    import imageio

    # Parse size
    h, w = [int(x) for x in size.split("*")]

    # Load reference image if provided
    ref_image = None
    if image_b64:
        img_bytes = base64.b64decode(image_b64)
        ref_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ref_image = ref_image.resize((w, h))

    print(f"[gen] prompt={prompt[:80]}... size={w}x{h} frames={frame_num} seed={seed}")
    t0 = time.time()

    # Generate
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    if ref_image is not None:
        video_frames = pipe(
            prompt=prompt,
            image=ref_image,
            num_frames=frame_num,
            height=h,
            width=w,
            generator=generator,
        )
    else:
        video_frames = pipe(
            prompt=prompt,
            num_frames=frame_num,
            height=h,
            width=w,
            generator=generator,
        )

    duration = time.time() - t0
    print(f"[gen] Generated {len(video_frames)} frames in {duration:.1f}s")

    # Encode to MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    # video_frames is a list of PIL Images or numpy arrays
    import numpy as np
    frames_np = []
    for frame in video_frames:
        if hasattr(frame, "numpy"):
            frames_np.append(frame.numpy())
        elif hasattr(frame, "convert"):
            frames_np.append(np.array(frame))
        else:
            frames_np.append(np.array(frame))

    imageio.mimwrite(tmp_path, frames_np, fps=16, quality=8)

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    os.unlink(tmp_path)

    return {
        "video_base64": video_b64,
        "duration_seconds": round(duration, 1),
        "frames": len(video_frames),
        "size": f"{w}x{h}",
    }


# Start the serverless worker
runpod.serverless.start({"handler": handler})
