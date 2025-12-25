"""
RunPod Serverless handler for SCAIL using Wan2GP's optimized implementation.

This version uses:
- Quantized INT8 models (~14GB instead of ~28GB)
- mmgp library for efficient memory management
- Wan2GP's inference pipeline for better VRAM usage

Input (event["input"]):
{
    "prompt": "the girl is dancing",
    "pose_video": "<https url | data:uri | base64 string>",
    "reference_image": "<https url | data:uri | base64 string>",
    "job_id": "optional-id",
    "seed": 123,
    "upload_url": "<https presigned PUT for output mp4>",
    "upload_url_concat": "<https presigned PUT for concat mp4>",  # optional
    "webhook_url": "<https webhook for status callback>",
    "timeout": 900
}
"""

import base64
import os
import sys
import tempfile
import uuid
import subprocess
from pathlib import Path
from typing import Optional

import requests
import runpod
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import cv2
from mmgp import offload

# Add wan2gp_integration to path
sys.path.insert(0, str(Path(__file__).parent / "wan2gp_integration"))

from models.wan.any2video import WanAny2V
from shared.utils import files_locator as fl
from shared.utils.audio_video import save_video
from shared.utils.utils import convert_image_to_tensor

# Enable cuDNN for optimized 3D convolutions
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
print(f"cuDNN enabled: {torch.backends.cudnn.enabled}, available: {torch.backends.cudnn.is_available()}")

# Configuration
MODEL_DIR = Path(os.environ.get("MODEL_BASE_PATH", "/runpod-volume/models"))
SCAIL_CONFIG = {
    "num_train_timesteps": 1000,
    "text_len": 512,
    "t5_dtype": torch.bfloat16,
    "clip_dtype": torch.bfloat16,
    "clip_checkpoint": "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth",
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "param_dtype": torch.bfloat16,
    "sample_neg_prompt": "",  # Default negative prompt
}

# Offload configuration (matches Wan2GP profiles)
MMGP_PROFILE = int(os.environ.get("MMGP_PROFILE", "5"))
MMGP_PRELOAD = int(os.environ.get("MMGP_PRELOAD_IN_VRAM", "0"))
VAE_DTYPE = torch.float16 if os.environ.get("VAE_DTYPE", "fp16") == "fp16" else torch.float32

# Model definition for SCAIL
SCAIL_MODEL_DEF = {
    "URLs": [str(MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors")],
    "scail": True,
}


class SCAILConfig:
    """Configuration class matching Wan2GP's config structure"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


# Global model instance (loaded once, reused across requests)
_model_instance = None
_offload_configured = False


def _configure_offload(model: WanAny2V) -> None:
    """Configure mmgp offload profile to keep VRAM usage low."""
    global _offload_configured
    if _offload_configured:
        return

    # Build pipe dict similar to Wan2GP's offload profile usage.
    pipe = {
        "transformer": model.model,
        "text_encoder": model.text_encoder.model,
        "vae": model.vae.model,
    }
    if hasattr(model, "model2") and model.model2 is not None:
        pipe["transformer2"] = model.model2
    if hasattr(model, "clip") and model.clip is not None:
        pipe["clip"] = model.clip.model

    kwargs = {}
    if MMGP_PROFILE in (2, 4, 5):
        budgets = {
            "transformer": 100 if MMGP_PRELOAD == 0 else MMGP_PRELOAD,
            "text_encoder": 100 if MMGP_PRELOAD == 0 else MMGP_PRELOAD,
            "*": max(1000 if MMGP_PROFILE == 5 else 3000, MMGP_PRELOAD),
        }
        if "transformer2" in pipe:
            budgets["transformer2"] = 100 if MMGP_PRELOAD == 0 else MMGP_PRELOAD
        kwargs["budgets"] = budgets
    elif MMGP_PROFILE == 3:
        kwargs["budgets"] = {"*": "70%"}

    offload.profile(pipe, profile_no=MMGP_PROFILE, quantizeTransformer=False, **kwargs)
    _offload_configured = True


def get_model():
    """Get or initialize the SCAIL model instance"""
    global _model_instance

    if _model_instance is None:
        print("Loading SCAIL model (this may take a few minutes)...")

        # Set up file locator to find models (takes a list of paths)
        fl.set_checkpoints_paths([str(MODEL_DIR)])

        config = SCAILConfig(SCAIL_CONFIG)

        # Initialize Wan2GP model with quantized SCAIL
        _model_instance = WanAny2V(
            config=config,
            checkpoint_dir=str(MODEL_DIR),
            model_filename=[str(MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors")],
            submodel_no_list=[0],
            model_type="scail",
            model_def=SCAIL_MODEL_DEF,
            base_model_type="scail",
            text_encoder_filename=str(MODEL_DIR / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth"),
            quantizeTransformer=False,  # Already quantized
            dtype=torch.bfloat16,
            VAE_dtype=VAE_DTYPE,
        )

        print("SCAIL model loaded successfully!")

        # Configure offload profile to keep most weights off VRAM.
        _configure_offload(_model_instance)

    return _model_instance


def _materialize_source(src: str, dest: Path, binary: bool = True) -> Path:
    """Save a URL/base64/local-path payload to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, stream=True, timeout=60)
        resp.raise_for_status()
        mode = "wb" if binary else "w"
        with open(dest, mode) as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk if binary else chunk.decode("utf-8"))
        return dest

    if src.startswith("data:"):
        _, b64_data = src.split(",", 1)
        data_bytes = base64.b64decode(b64_data)
        dest.write_bytes(data_bytes)
        return dest

    # Treat as base64 string
    try:
        data_bytes = base64.b64decode(src, validate=True)
        dest.write_bytes(data_bytes)
        return dest
    except Exception:
        raise ValueError(f"Invalid input source: {src[:50]}...")


def _upload_file(upload_url: str, file_path: Path, content_type: str) -> None:
    """Upload file to presigned URL"""
    with open(file_path, "rb") as f:
        resp = requests.put(
            upload_url,
            data=f,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(file_path.stat().st_size),
            },
            timeout=300,
        )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Upload failed ({resp.status_code}): {resp.text[:200]}")


def _create_concat_video(ref_image_path: Path, output_video_path: Path) -> Path:
    """Create concatenated video (reference image + output video)"""
    concat_path = output_video_path.parent / f"{output_video_path.stem}_concat.mp4"

    # Read reference image
    ref_img = cv2.imread(str(ref_image_path))
    height, width = ref_img.shape[:2]

    # Get video properties
    cap = cv2.VideoCapture(str(output_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create temporary image video (1 second of reference image)
    temp_ref_video = output_video_path.parent / "temp_ref.mp4"
    ref_duration = 1.0  # 1 second

    # Use ffmpeg to create ref image video and concatenate
    # First, create a 1-second video from the reference image
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(ref_image_path),
        "-t", str(ref_duration),
        "-vf", f"scale={width}:{height}",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        str(temp_ref_video)
    ], check=True, capture_output=True)

    # Concatenate reference video and output video
    concat_list = output_video_path.parent / "concat_list.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{temp_ref_video.name}'\n")
        f.write(f"file '{output_video_path.name}'\n")

    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(concat_path)
    ], check=True, capture_output=True, cwd=str(output_video_path.parent))

    # Cleanup temporary files
    temp_ref_video.unlink()
    concat_list.unlink()

    return concat_path


def _load_pose_video(video_path: Path, target_height: int = 512, target_width: int = 896) -> torch.Tensor:
    """Load pose video and convert to tensor format expected by SCAIL

    Returns:
        torch.Tensor: Video tensor in shape (C, T, H, W), normalized to [-1, 1]
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Could not load any frames from pose video")

    # Convert to tensor: List of (H, W, C) -> (T, H, W, C)
    video = torch.from_numpy(np.stack(frames, axis=0))

    # Permute to (T, C, H, W)
    video = video.permute(0, 3, 1, 2).float()

    # Resize to target resolution
    if video.shape[2] != target_height or video.shape[3] != target_width:
        scale = max(target_width / video.shape[3], target_height / video.shape[2])
        new_h = int(video.shape[2] * scale)
        new_w = int(video.shape[3] * scale)
        video = F.interpolate(video, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Center crop
        y1 = (new_h - target_height) // 2
        x1 = (new_w - target_width) // 2
        video = video[:, :, y1:y1+target_height, x1:x1+target_width]

    # Permute to (C, T, H, W) and normalize to [-1, 1]
    video = video.permute(1, 0, 2, 3)
    video = video.div(127.5).sub(1.0)

    return video


def _run_inference(
    model: WanAny2V,
    prompt: str,
    pose_video_path: Path,
    ref_image_path: Path,
    seed: Optional[int],
) -> tuple[Path, Path]:
    """Run SCAIL inference using Wan2GP pipeline

    Returns:
        tuple: (output_path, concat_path)
    """

    # Set seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Load and prepare reference image
    ref_image_pil = Image.open(ref_image_path).convert("RGB")

    # Resize reference image to match output resolution (critical for VRAM usage)
    # Wan2GP does this at wgp.py:5629 with resize_and_remove_background()
    original_size = ref_image_pil.size  # (W, H)
    ref_image_pil = TF.resize(ref_image_pil, (512, 896), interpolation=TF.InterpolationMode.LANCZOS)
    print(f"Reference image resized: {original_size} -> {ref_image_pil.size}")

    # Convert reference image to tensor and add time dimension
    # SCAIL expects: (C, 1, H, W) tensor, not PIL Image
    ref_image_tensor = convert_image_to_tensor(ref_image_pil).unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)
    print(f"Reference image loaded: {ref_image_tensor.shape}")

    # Load pose video as tensor
    print(f"Loading pose video from {pose_video_path}...")
    pose_video_tensor = _load_pose_video(pose_video_path, target_height=512, target_width=896)
    print(f"Pose video loaded: {pose_video_tensor.shape}")

    # Call generate() with correct keyword arguments (matching wgp.py usage)
    print("Running SCAIL inference...")
    output = model.generate(
        input_prompt=prompt,
        input_ref_images=[ref_image_tensor],  # List of tensors: [(C, 1, H, W)]
        input_frames=pose_video_tensor,  # 4D tensor: (C, T, H, W)
        frame_num=81,  # Number of frames to generate (5 seconds at 16fps)
        height=512,
        width=896,
        sampling_steps=50,
        guide_scale=4.0,
        seed=seed or 42,
        model_type="scail",
        VAE_tile_size=256,  # Enable tiled VAE encoding to save VRAM (critical!)
    )

    # Extract video tensor from output dict
    # The generate() method returns {"x": video_tensor, "latent_slice": ...}
    videos = output["x"]
    print(f"Inference complete, output shape: {videos.shape}")

    # Save output video (SCAIL outputs at 16fps)
    output_path = pose_video_path.parent / "output.mp4"
    save_video(videos, str(output_path), fps=16)
    print(f"Video saved to {output_path}")

    # Create concatenated output (ref image + output video)
    concat_path = _create_concat_video(ref_image_path, output_path)

    return output_path, concat_path


def _maybe_post_webhook(webhook_url: Optional[str], payload: dict) -> None:
    """Post webhook callback (best effort)"""
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json=payload, timeout=15)
    except Exception:
        pass


def handler(event):
    """RunPod entrypoint"""
    job_input = event.get("input", event)
    job_id = job_input.get("job_id") or event.get("id") or f"scail-{uuid.uuid4().hex[:8]}"
    prompt = job_input.get("prompt") or ""
    pose_src = job_input.get("pose_video")
    ref_src = job_input.get("reference_image")
    upload_url = job_input.get("upload_url")
    upload_url_concat = job_input.get("upload_url_concat")
    webhook_url = job_input.get("webhook_url")
    seed = job_input.get("seed")

    if not pose_src or not ref_src:
        raise ValueError("Both 'pose_video' and 'reference_image' are required.")

    job_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}_", dir="/tmp"))
    pose_path = job_dir / "pose.mp4"
    ref_path = job_dir / "ref.jpg"

    # Materialize inputs
    _materialize_source(pose_src, pose_path, binary=True)
    _materialize_source(ref_src, ref_path, binary=True)

    result_payload = {
        "job_id": job_id,
        "status": "running",
        "output_path": None,
        "concat_path": None,
    }
    _maybe_post_webhook(webhook_url, result_payload)

    try:
        # Get model instance
        model = get_model()

        # Run inference (returns both output and concat)
        output_path, concat_path = _run_inference(model, prompt, pose_path, ref_path, seed)

        # Upload if requested
        if upload_url:
            _upload_file(upload_url, output_path, content_type="video/mp4")
        if upload_url_concat and concat_path:
            _upload_file(upload_url_concat, concat_path, content_type="video/mp4")

        result_payload.update({
            "status": "succeeded",
            "output_path": str(output_path),
            "concat_path": str(concat_path),
            "seed": seed,
        })
        _maybe_post_webhook(webhook_url, result_payload)
        return result_payload

    except Exception as exc:
        error_msg = str(exc)
        fail_payload = {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
        }
        _maybe_post_webhook(webhook_url, fail_payload)
        raise


def health_check(_event=None):
    """Health probe"""
    return {"status": "ok"}


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "health_check": health_check,
        "concurrency_modifier": lambda _: 1,
    })
