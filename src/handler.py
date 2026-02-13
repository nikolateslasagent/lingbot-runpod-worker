"""
RunPod Serverless handler for LingBot World (Wan2.2 I2V-A14B).
"""

import os
import sys
import time
import base64
import tempfile
import shutil

# Add lingbot-world source to Python path
sys.path.insert(0, "/app/lingbot-world")

import runpod

MODEL_REPO = "robbyant/lingbot-world-base-cam"
_vol = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp"
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_vol}/model")
DEVICE = 0  # GPU device id

# Global model reference
wan_i2v = None


def download_model():
    """Download model from HuggingFace if not already cached."""
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
    """Load the WanI2V model."""
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
        t5_cpu=False,
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

    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.utils import save_video
    import io

    # Load reference image if provided
    img = None
    if image_b64:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    print(f"[gen] prompt={prompt[:80]}... size={size} frames={frame_num} seed={seed} img={'yes' if img else 'no'}")
    t0 = time.time()

    # Generate video
    video = wan_i2v.generate(
        prompt,
        img,
        max_area=MAX_AREA_CONFIGS[size],
        frame_num=frame_num,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=40,
        guide_scale=5.0,
        seed=seed,
        offload_model=True,
    )

    duration = time.time() - t0
    print(f"[gen] Generated in {duration:.1f}s")

    # Save to MP4
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name

    save_video(video, tmp_path, fps=16)

    with open(tmp_path, "rb") as f:
        video_b64 = base64.b64encode(f.read()).decode()

    os.unlink(tmp_path)

    return {
        "video_base64": video_b64,
        "duration_seconds": round(duration, 1),
        "frames": frame_num,
        "size": size,
    }


runpod.serverless.start({"handler": handler})
