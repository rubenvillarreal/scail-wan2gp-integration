"""
RunPod Serverless handler for SCAIL pose-to-video sampling.

Input (event["input"]):
{
    "prompt": "the girl is dancing",
    "pose_video": "<https url | data:uri | base64 string | local path>",   # rendered pose video (e.g., rendered.mp4)
    "reference_image": "<https url | data:uri | base64 string | local path>",  # reference image (e.g., ref.jpg/png)
    "job_id": "optional-id",                       # default: event["id"] or generated UUID
    "seed": 123,                                   # optional
    "upload_url": "<https presigned PUT for output mp4>",   # optional
    "upload_url_concat": "<presigned PUT for concat mp4>",  # optional
    "webhook_url": "<https webhook for status callback>",   # optional
    "timeout": 900                                 # optional, seconds for sampling process
}

Behavior:
- Downloads pose video + reference image into /tmp/<job>/ as rendered.mp4 + ref.jpg (also symlinks rendered_aligned.mp4).
- Runs SCAIL sampler (Wan2.1 pose latent) via sample_video.py with prompt@@<job_dir>.
- Collects outputs from samples/<job>/<job>_output_000000.mp4 (and *_concat_000000.mp4 if present).
- Optionally uploads outputs to provided presigned URLs and/or posts completion to webhook_url.
"""

from __future__ import annotations

import base64
import os
import random
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import requests
import runpod


ROOT = Path(__file__).resolve().parent
WORKDIR = ROOT  # sample_video.py expects to run from repo root
ASSET_BASE_URL = os.environ.get("ASSET_BASE_URL")  # Optional, e.g., https://three-pose-viewer-395443-ba158.web.app

# Config paths for the official pose-conditioned sampler
BASE_CONFIGS = [
    "configs/video_model/Wan2.1-i2v-14Bsc-pose-xc-latent.yaml",
    "configs/sampling/wan_pose_14Bsc_xc_cli.yaml",
]


def _materialize_source(src: str, dest: Path, binary: bool = True) -> Path:
    """Save a URL/base64/local-path payload to dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    # If it's a rooted path but not present locally, try asset base URL if provided
    if src.startswith("/") and not Path(src).exists() and ASSET_BASE_URL:
        src = f"{ASSET_BASE_URL.rstrip('/')}{src}"

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

    # Treat as base64 string if it decodes cleanly
    try:
        data_bytes = base64.b64decode(src, validate=True)
        dest.write_bytes(data_bytes)
        return dest
    except Exception:
        pass

    # Fallback: local file path
    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    shutil.copy(src_path, dest)
    return dest


def _upload_file(upload_url: str, file_path: Path, content_type: str) -> None:
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


def _run_sampler(prompt: str, job_dir: Path, seed: Optional[int], timeout: int) -> Path:
    """Invoke sample_video.py in non-interactive mode by piping the prompt line."""
    prompt_line = f"{prompt}@@{job_dir}\n"
    env = os.environ.copy()
    env.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "HF_HUB_OFFLINE": env.get("HF_HUB_OFFLINE", "1"),
            # Memory optimizations
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "CUDA_LAUNCH_BLOCKING": "0",
        }
    )

    cmd = [
        "python",
        "sample_video.py",
        "--base",
        *BASE_CONFIGS,
        "--distributed-backend",
        "gloo",  # Use gloo instead of nccl for single-GPU inference
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    else:
        cmd.extend(["--seed", str(random.randint(1, 10_000_000))])

    proc = subprocess.run(
        cmd,
        cwd=str(WORKDIR),
        input=prompt_line.encode("utf-8"),
        capture_output=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"SCAIL sampler failed (exit {proc.returncode}): {proc.stderr.decode('utf-8')[-500:]}"
        )

    # Outputs land in samples/<job>/<job>_output_000000.mp4
    output_dir = WORKDIR / "samples" / job_dir.name
    main_mp4 = output_dir / f"{job_dir.name}_output_000000.mp4"
    if not main_mp4.exists():
        raise FileNotFoundError(f"Expected output not found: {main_mp4}")
    return main_mp4


def _maybe_post_webhook(webhook_url: Optional[str], payload: dict) -> None:
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json=payload, timeout=15)
    except Exception:
        # Best-effort; do not raise inside handler
        pass


def handler(event):
    """
    RunPod entrypoint.
    """
    job_input = event.get("input", event)
    job_id = job_input.get("job_id") or event.get("id") or f"scail-job-{uuid.uuid4().hex[:8]}"
    prompt = job_input.get("prompt") or ""
    pose_src = job_input.get("pose_video")
    ref_src = job_input.get("reference_image")
    upload_url = job_input.get("upload_url")
    upload_url_concat = job_input.get("upload_url_concat")
    webhook_url = job_input.get("webhook_url")
    seed = job_input.get("seed")
    timeout = int(job_input.get("timeout", 1200))

    if not pose_src or not ref_src:
        raise ValueError("Both 'pose_video' and 'reference_image' are required.")

    job_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}_", dir="/tmp"))
    pose_path = job_dir / "rendered.mp4"
    ref_path = job_dir / "ref.jpg"

    # Materialize inputs
    _materialize_source(pose_src, pose_path, binary=True)
    # Provide both rendered.mp4 and rendered_aligned.mp4 to satisfy CLI lookup
    aligned_path = job_dir / "rendered_aligned.mp4"
    if not aligned_path.exists():
        aligned_path.symlink_to(pose_path)

    _materialize_source(ref_src, ref_path, binary=True)

    result_payload = {
        "job_id": job_id,
        "status": "running",
        "output_path": None,
        "concat_path": None,
    }
    _maybe_post_webhook(webhook_url, result_payload)

    try:
        main_output = _run_sampler(prompt, job_dir, seed, timeout)

        concat_output = main_output.parent / f"{job_dir.name}_concat_000000.mp4"
        if not concat_output.exists():
            concat_output = None

        # Uploads (optional)
        if upload_url:
            _upload_file(upload_url, main_output, content_type="video/mp4")
        if upload_url_concat and concat_output:
            _upload_file(upload_url_concat, concat_output, content_type="video/mp4")

        result_payload.update(
            {
                "status": "succeeded",
                "output_path": str(main_output),
                "concat_path": str(concat_output) if concat_output else None,
                "seed": seed,
            }
        )
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
    """Lightweight health probe."""
    return {"status": "ok"}


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "health_check": health_check,
            "concurrency_modifier": lambda _: 1,
        }
    )
