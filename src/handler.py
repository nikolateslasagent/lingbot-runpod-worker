"""
RunPod Serverless handler for LingBot World — Interactive World Model.

Supports two modes:
1. "generate" — Standard video generation (image + prompt + optional poses → video)
2. "generate_frames" — Interactive chunked generation (image + prompt + poses → individual JPEG frames)
   Used for real-time-ish interactive exploration with WASD controls.
"""

import os
import sys
import time
import base64
import tempfile
import shutil
import io
import json

sys.path.insert(0, "/app/lingbot-world")

import runpod

MODEL_REPO = "robbyant/lingbot-world-base-cam"
_vol = "/runpod-volume" if os.path.isdir("/runpod-volume") else "/tmp"
MODEL_DIR = os.environ.get("MODEL_DIR", f"{_vol}/model")
DEVICE = 0

wan_i2v = None


def download_model():
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
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)

    with open(marker, "w") as f:
        f.write("ok")
    print("[init] Model download complete.")


def load_model():
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
        convert_model_dtype=True,
    )
    print(f"[init] Model loaded in {time.time() - t0:.1f}s")


def make_camera_poses_from_directions(directions, num_frames=17, step_size=0.15):
    """
    Build camera pose matrices from direction commands.

    Args:
        directions: list of {"forward": float, "right": float, "up": float, "yaw": float, "pitch": float}
                    One per frame, or a single dict applied to all frames.
        num_frames: number of frames to generate
        step_size: base translation magnitude per frame

    Returns:
        poses: numpy array [num_frames, 4, 4] — camera-to-world matrices (OpenCV convention)
        intrinsics: numpy array [num_frames, 4] — [fx, fy, cx, cy]
    """
    import numpy as np

    if isinstance(directions, dict):
        directions = [directions] * num_frames

    # Pad directions to num_frames
    while len(directions) < num_frames:
        directions.append(directions[-1] if directions else {"forward": 0, "right": 0, "up": 0, "yaw": 0, "pitch": 0})

    poses = np.zeros((num_frames, 4, 4))
    poses[:, 3, 3] = 1.0

    # Start at identity
    current_pos = np.array([0.0, 0.0, 0.0])
    current_yaw = 0.0  # rotation around Y axis
    current_pitch = 0.0  # rotation around X axis

    for i in range(num_frames):
        d = directions[i] if i < len(directions) else directions[-1]

        # Accumulate rotation
        current_yaw += d.get("yaw", 0.0) * 0.02  # radians per frame
        current_pitch += d.get("pitch", 0.0) * 0.02
        current_pitch = np.clip(current_pitch, -np.pi / 4, np.pi / 4)

        # Build rotation matrix (Y then X)
        cy, sy = np.cos(current_yaw), np.sin(current_yaw)
        cp, sp = np.cos(current_pitch), np.sin(current_pitch)

        # Rotation: Ry * Rx
        R = np.array([
            [cy, sy * sp, sy * cp],
            [0, cp, -sp],
            [-sy, cy * sp, cy * cp]
        ])

        # Forward direction in world space (camera looks along +Z in OpenCV)
        forward = R @ np.array([0, 0, 1])
        right = R @ np.array([1, 0, 0])
        up = R @ np.array([0, -1, 0])  # OpenCV: Y points down

        # Translate
        move = (
            forward * d.get("forward", 0.0) * step_size
            + right * d.get("right", 0.0) * step_size
            + up * d.get("up", 0.0) * step_size
        )
        current_pos += move

        poses[i, :3, :3] = R
        poses[i, :3, 3] = current_pos

    # Default intrinsics for 480p (480x832)
    fx = fy = 400.0
    cx, cy_val = 416.0, 240.0
    intrinsics = np.tile(np.array([fx, fy, cx, cy_val]), (num_frames, 1))

    return poses, intrinsics


def handler(job):
    """Handle a generation request."""
    load_model()

    inp = job["input"]
    mode = inp.get("mode", "generate")

    import torch
    import numpy as np
    from PIL import Image
    from wan.configs import MAX_AREA_CONFIGS
    from wan.utils.utils import save_video

    prompt = inp.get("prompt", "A beautiful landscape")
    image_b64 = inp.get("image_base64")
    size = inp.get("size", "480*832")
    seed = int(inp.get("seed", 42))

    # Parse size
    parts = size.split("*")
    h_target, w_target = int(parts[0]), int(parts[1])

    # Load image
    if not image_b64:
        img = Image.new("RGB", (w_target, h_target), (135, 206, 235))
        print("[gen] No input image, using placeholder")
    else:
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    if mode == "generate_frames":
        # Interactive chunked mode — generate frames with camera control
        frame_num = int(inp.get("frame_num", 17))  # 17 = 4*4+1
        # Ensure 4n+1
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        frame_num = max(5, min(frame_num, 81))

        sampling_steps = int(inp.get("sampling_steps", 15))  # Fewer steps for speed
        directions = inp.get("directions", {"forward": 1.0})
        step_size = float(inp.get("step_size", 0.15))

        # Build camera poses
        poses, intrinsics = make_camera_poses_from_directions(
            directions, num_frames=frame_num, step_size=step_size
        )

        # Write poses to temp dir for the model
        tmp_action = tempfile.mkdtemp()
        np.save(os.path.join(tmp_action, "poses.npy"), poses)
        np.save(os.path.join(tmp_action, "intrinsics.npy"), intrinsics)

        print(f"[gen-frames] frames={frame_num} steps={sampling_steps} seed={seed}")
        t0 = time.time()

        video = wan_i2v.generate(
            prompt,
            img,
            action_path=tmp_action,
            max_area=MAX_AREA_CONFIGS.get(size, h_target * w_target),
            frame_num=frame_num,
            shift=3.0,  # 3.0 recommended for 480p
            sample_solver="unipc",
            sampling_steps=sampling_steps,
            guide_scale=5.0,
            seed=seed,
            offload_model=False,
        )

        duration = time.time() - t0
        print(f"[gen-frames] Generated {frame_num} frames in {duration:.1f}s")

        # Clean up temp
        shutil.rmtree(tmp_action, ignore_errors=True)

        # Convert video tensor to individual base64 JPEG frames
        # video shape: [C, F, H, W] range [-1, 1]
        frames_b64 = []
        video_np = ((video.cpu().float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 3, 0).numpy()
        # video_np shape: [F, H, W, C]

        for i in range(video_np.shape[0]):
            frame_img = Image.fromarray(video_np[i])
            buf = io.BytesIO()
            frame_img.save(buf, format="JPEG", quality=85)
            frames_b64.append(base64.b64encode(buf.getvalue()).decode())

        # Also return last frame as the "continuation image" for next chunk
        last_frame_buf = io.BytesIO()
        Image.fromarray(video_np[-1]).save(last_frame_buf, format="JPEG", quality=95)
        last_frame_b64 = base64.b64encode(last_frame_buf.getvalue()).decode()

        return {
            "mode": "frames",
            "frames": frames_b64,
            "last_frame": last_frame_b64,
            "frame_count": len(frames_b64),
            "duration_seconds": round(duration, 1),
            "fps": 16,
        }

    else:
        # Standard video generation mode
        frame_num = int(inp.get("frame_num", 33))
        frame_num = ((frame_num - 1) // 4) * 4 + 1

        # Optional camera poses
        action_path = None
        poses_data = inp.get("poses")
        intrinsics_data = inp.get("intrinsics")

        if poses_data and intrinsics_data:
            tmp_action = tempfile.mkdtemp()
            poses_arr = np.array(poses_data, dtype=np.float64)
            intrinsics_arr = np.array(intrinsics_data, dtype=np.float64)
            np.save(os.path.join(tmp_action, "poses.npy"), poses_arr)
            np.save(os.path.join(tmp_action, "intrinsics.npy"), intrinsics_arr)
            action_path = tmp_action

        # Check for direction-based poses
        directions = inp.get("directions")
        if directions and not action_path:
            poses_arr, intrinsics_arr = make_camera_poses_from_directions(
                directions, num_frames=frame_num
            )
            tmp_action = tempfile.mkdtemp()
            np.save(os.path.join(tmp_action, "poses.npy"), poses_arr)
            np.save(os.path.join(tmp_action, "intrinsics.npy"), intrinsics_arr)
            action_path = tmp_action

        sampling_steps = int(inp.get("sampling_steps", 20))

        print(f"[gen] prompt={prompt[:80]}... size={size} frames={frame_num} seed={seed}")
        t0 = time.time()

        video = wan_i2v.generate(
            prompt,
            img,
            action_path=action_path,
            max_area=MAX_AREA_CONFIGS.get(size, h_target * w_target),
            frame_num=frame_num,
            shift=3.0 if "480" in size else 5.0,
            sample_solver="unipc",
            sampling_steps=sampling_steps,
            guide_scale=5.0,
            seed=seed,
            offload_model=False,
        )

        duration = time.time() - t0
        print(f"[gen] Generated in {duration:.1f}s")

        if action_path:
            shutil.rmtree(action_path, ignore_errors=True)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            tmp_path = f.name

        save_video(video[None], tmp_path, fps=16, normalize=True, value_range=(-1, 1))

        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        os.unlink(tmp_path)

        return {
            "mode": "video",
            "video_base64": video_b64,
            "duration_seconds": round(duration, 1),
            "frames": frame_num,
            "size": size,
        }


runpod.serverless.start({"handler": handler})
