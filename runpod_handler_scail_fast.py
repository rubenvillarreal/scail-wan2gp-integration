"""
RunPod Serverless handler for SCAIL using ComfyUI-WanVideoWrapper fast path.

Requires:
- ComfyUI core (for comfy.* and folder_paths)
- ComfyUI-WanVideoWrapper (this repo's nodes + wanvideo modules)

Input (event["input"]):
{
    "prompt": "the girl is dancing",
    "pose_video": "<https url | data:uri | base64 string>",
    "reference_image": "<https url | data:uri | base64 string>",
    "job_id": "optional-id",
    "seed": 123,
    "upload_url": "<https presigned PUT for output mp4>",
    "upload_url_concat": "<https presigned PUT for concat mp4>",  # optional
    "upload_url_log": "<https presigned PUT for log>",  # optional
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
import shutil
import traceback
import importlib
import importlib.util
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
import runpod
import torch
import torch.nn.functional as F
from PIL import Image

# Avoid tokenizer fork warnings in serverless environments
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/workspace/ComfyUI")
WAN_WRAPPER_PATH = os.environ.get("WAN_WRAPPER_PATH", "/workspace/ComfyUI-WanVideoWrapper")
WAN_WRAPPER_PACKAGE = os.environ.get("WAN_WRAPPER_PACKAGE", "ComfyUI_WanVideoWrapper")

if COMFYUI_PATH and COMFYUI_PATH not in sys.path:
    sys.path.insert(0, COMFYUI_PATH)
if WAN_WRAPPER_PATH and WAN_WRAPPER_PATH not in sys.path:
    # Keep wrapper path out of sys.path to avoid shadowing ComfyUI's utils package.
    pass

_folder_paths = None


def _get_folder_paths():
    global _folder_paths
    if _folder_paths is None:
        try:
            import folder_paths as fp
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "ComfyUI core not found. Set COMFYUI_PATH to the ComfyUI repo root."
            ) from exc
        _folder_paths = fp
    return _folder_paths


def _register_stub_package(package_name: str, path: Optional[Path] = None) -> None:
    if package_name in sys.modules:
        return
    pkg = types.ModuleType(package_name)
    pkg.__package__ = package_name
    pkg.__path__ = [str(path)] if path else []
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    if path:
        spec.submodule_search_locations = [str(path)]
    pkg.__spec__ = spec
    sys.modules[package_name] = pkg

def _load_wrapper_package():
    wrapper_path = Path(WAN_WRAPPER_PATH)
    if not wrapper_path.exists():
        raise RuntimeError(
            "ComfyUI-WanVideoWrapper not found. Set WAN_WRAPPER_PATH to that repo root."
        )
    _register_stub_package(WAN_WRAPPER_PACKAGE, wrapper_path)
    _register_stub_package(f"{WAN_WRAPPER_PACKAGE}.SCAIL", wrapper_path / "SCAIL")
    _register_stub_package(f"{WAN_WRAPPER_PACKAGE}.multitalk", wrapper_path / "multitalk")
    _register_stub_package(f"{WAN_WRAPPER_PACKAGE}.latent_preview", wrapper_path)


try:
    _load_wrapper_package()
    # Stub out multitalk_loop to avoid latent_preview/server imports.
    multitalk_pkg = f"{WAN_WRAPPER_PACKAGE}.multitalk"
    _register_stub_package(multitalk_pkg, Path(WAN_WRAPPER_PATH) / "multitalk")
    multitalk_stub_name = f"{WAN_WRAPPER_PACKAGE}.multitalk.multitalk_loop"
    if multitalk_stub_name not in sys.modules:
        multitalk_stub = types.ModuleType(multitalk_stub_name)
        def _noop_multitalk_loop(*args, **kwargs):
            raise NotImplementedError("multitalk_loop is not available in serverless mode.")
        multitalk_stub.multitalk_loop = _noop_multitalk_loop
        sys.modules[multitalk_stub_name] = multitalk_stub

    latent_preview_name = f"{WAN_WRAPPER_PACKAGE}.latent_preview"
    if latent_preview_name not in sys.modules:
        latent_preview_stub = types.ModuleType(latent_preview_name)

        def _prepare_callback(*args, **kwargs):
            def _noop_callback(*cb_args, **cb_kwargs):
                return None
            return _noop_callback, None

        latent_preview_stub.prepare_callback = _prepare_callback
        sys.modules[latent_preview_name] = latent_preview_stub

    nodes_model_loading = importlib.import_module(f"{WAN_WRAPPER_PACKAGE}.nodes_model_loading")
    nodes_sampler = importlib.import_module(f"{WAN_WRAPPER_PACKAGE}.nodes_sampler")
    nodes_main = importlib.import_module(f"{WAN_WRAPPER_PACKAGE}.nodes")
    scail_nodes = importlib.import_module(f"{WAN_WRAPPER_PACKAGE}.SCAIL.nodes")

    WanVideoModelLoader = nodes_model_loading.WanVideoModelLoader
    WanVideoVAELoader = nodes_model_loading.WanVideoVAELoader
    LoadWanVideoT5TextEncoder = nodes_model_loading.LoadWanVideoT5TextEncoder
    WanVideoLoraSelect = nodes_model_loading.WanVideoLoraSelect
    WanVideoSetLoRAs = nodes_model_loading.WanVideoSetLoRAs
    WanVideoTorchCompileSettings = nodes_model_loading.WanVideoTorchCompileSettings

    WanVideoSamplerv2 = nodes_sampler.WanVideoSamplerv2
    WanVideoSchedulerv2 = nodes_sampler.WanVideoSchedulerv2

    WanVideoDecode = nodes_main.WanVideoDecode
    WanVideoEmptyEmbeds = nodes_main.WanVideoEmptyEmbeds
    WanVideoTextEncodeCached = nodes_main.WanVideoTextEncodeCached

    WanVideoAddSCAILReferenceEmbeds = scail_nodes.WanVideoAddSCAILReferenceEmbeds
    WanVideoAddSCAILPoseEmbeds = scail_nodes.WanVideoAddSCAILPoseEmbeds
except Exception as exc:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "ComfyUI-WanVideoWrapper not found or failed to import. "
        "Set WAN_WRAPPER_PATH to that repo root."
    ) from exc

from wan2gp_integration.shared.utils.audio_video import save_video

# Configuration
MODEL_DIR = Path(os.environ.get("MODEL_BASE_PATH", "/runpod-volume/models"))
COMFY_MODEL_DIR = Path(os.environ.get("COMFY_MODEL_DIR", str(MODEL_DIR)))

SCAIL_FAST_MODEL_NAME = os.environ.get(
    "SCAIL_FAST_MODEL_NAME",
    "Wan21-14B-SCAIL-preview_fp8_e4m3fn_scaled_KJ.safetensors",
)
SCAIL_FAST_VAE_NAME = os.environ.get(
    "SCAIL_FAST_VAE_NAME",
    "Wan2_1_VAE_bf16.safetensors",
)
SCAIL_FAST_T5_NAME = os.environ.get(
    "SCAIL_FAST_T5_NAME",
    "umt5-xxl-enc-bf16.safetensors",
)
SCAIL_FAST_LORA_NAME = os.environ.get(
    "SCAIL_FAST_LORA_NAME",
    "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
)

SCAIL_FAST_WIDTH = int(os.environ.get("SCAIL_FAST_WIDTH", "896"))
SCAIL_FAST_HEIGHT = int(os.environ.get("SCAIL_FAST_HEIGHT", "512"))
SCAIL_FAST_STEPS = int(os.environ.get("SCAIL_FAST_STEPS", "6"))
SCAIL_FAST_CFG = float(os.environ.get("SCAIL_FAST_CFG", "6.0"))
SCAIL_FAST_SHIFT = float(os.environ.get("SCAIL_FAST_SHIFT", "7.0"))
SCAIL_FAST_SCHEDULER = os.environ.get("SCAIL_FAST_SCHEDULER", "dpm++_sde")
SCAIL_FAST_SEED = int(os.environ.get("SCAIL_FAST_SEED", "42"))
SCAIL_FAST_ATTENTION = os.environ.get("SCAIL_FAST_ATTENTION", "sageattn")
SCAIL_FAST_BASE_PRECISION = os.environ.get("SCAIL_FAST_BASE_PRECISION", "fp16_fast")
SCAIL_FAST_LOAD_DEVICE = os.environ.get("SCAIL_FAST_LOAD_DEVICE", "offload_device")
SCAIL_FAST_LORA_STRENGTH = float(os.environ.get("SCAIL_FAST_LORA_STRENGTH", "1.0"))
SCAIL_FAST_LORA_MERGE = os.environ.get("SCAIL_FAST_LORA_MERGE", "0") == "1"
SCAIL_FAST_USE_COMPILE = os.environ.get("SCAIL_FAST_USE_COMPILE", "0") == "1"
SCAIL_FAST_VAE_TILING = os.environ.get("SCAIL_FAST_VAE_TILING", "0") == "1"
SCAIL_FAST_VAE_TILE_X = int(os.environ.get("SCAIL_FAST_VAE_TILE_X", "272"))
SCAIL_FAST_VAE_TILE_Y = int(os.environ.get("SCAIL_FAST_VAE_TILE_Y", "272"))
SCAIL_FAST_VAE_STRIDE_X = int(os.environ.get("SCAIL_FAST_VAE_STRIDE_X", "144"))
SCAIL_FAST_VAE_STRIDE_Y = int(os.environ.get("SCAIL_FAST_VAE_STRIDE_Y", "128"))


# Globals (cached across requests)
_fast_model = None
_fast_vae = None


def _configure_model_paths() -> None:
    folder_paths = _get_folder_paths()
    for name in ("diffusion_models", "vae", "text_encoders", "loras"):
        try:
            folder_paths.add_model_folder_path(name, str(COMFY_MODEL_DIR))
        except Exception:
            pass


def _load_models():
    global _fast_model, _fast_vae
    if _fast_model is not None and _fast_vae is not None:
        return _fast_model, _fast_vae

    _configure_model_paths()

    compile_args = None
    if SCAIL_FAST_USE_COMPILE:
        compile_args, = WanVideoTorchCompileSettings().set_args(
            backend="inductor",
            fullgraph=False,
            mode="default",
            dynamic=False,
            dynamo_cache_size_limit=64,
            compile_transformer_blocks_only=True,
        )

    model_loader = WanVideoModelLoader()
    model, = model_loader.loadmodel(
        model=SCAIL_FAST_MODEL_NAME,
        base_precision=SCAIL_FAST_BASE_PRECISION,
        load_device=SCAIL_FAST_LOAD_DEVICE,
        quantization="disabled",
        attention_mode=SCAIL_FAST_ATTENTION,
        compile_args=compile_args,
    )

    if SCAIL_FAST_LORA_NAME:
        lora_list, = WanVideoLoraSelect().getlorapath(
            lora=SCAIL_FAST_LORA_NAME,
            strength=SCAIL_FAST_LORA_STRENGTH,
            unique_id=None,
            merge_loras=SCAIL_FAST_LORA_MERGE,
        )
        if lora_list:
            model, = WanVideoSetLoRAs().setlora(model=model, lora=lora_list)

    vae_loader = WanVideoVAELoader()
    vae, = vae_loader.loadmodel(
        model_name=SCAIL_FAST_VAE_NAME,
        precision="bf16",
    )

    _fast_model = model
    _fast_vae = vae
    return _fast_model, _fast_vae


def _materialize_source(src: str, dest: Path, binary: bool = True) -> Path:
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

    try:
        data_bytes = base64.b64decode(src, validate=True)
        dest.write_bytes(data_bytes)
        return dest
    except Exception:
        raise ValueError(f"Invalid input source: {src[:50]}...")


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


def _create_concat_video(ref_image_path: Path, output_video_path: Path) -> Path:
    concat_path = output_video_path.parent / f"{output_video_path.stem}_concat.mp4"
    if shutil.which("ffmpeg") is None:
        print("⚠️  ffmpeg not found; skipping concat video generation.")
        return output_video_path

    ref_img = cv2.imread(str(ref_image_path))
    height, width = ref_img.shape[:2]
    cap = cv2.VideoCapture(str(output_video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    temp_ref_video = output_video_path.parent / "temp_ref.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(ref_image_path),
        "-t", "1.0",
        "-vf", f"scale={width}:{height}",
        "-r", str(fps),
        "-pix_fmt", "yuv420p",
        str(temp_ref_video)
    ], check=True, capture_output=True)

    concat_list = output_video_path.parent / "concat_list.txt"
    with open(concat_list, "w") as f:
        f.write(f"file '{temp_ref_video.name}'\n")
        f.write(f"file '{output_video_path.name}'\n")

    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list.name,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            concat_path.name
        ], check=True, capture_output=True, cwd=str(output_video_path.parent))
    except subprocess.CalledProcessError as e:
        print(f"⚠️  ffmpeg concat failed ({e.returncode}); skipping concat video generation.")
        concat_path = output_video_path

    temp_ref_video.unlink(missing_ok=True)
    concat_list.unlink(missing_ok=True)

    return concat_path


def _load_pose_video(video_path: Path, target_height: int, target_width: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError("Could not load any frames from pose video")

    video = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).float()

    if video.shape[2] != target_height or video.shape[3] != target_width:
        scale = max(target_width / video.shape[3], target_height / video.shape[2])
        new_h = int(video.shape[2] * scale)
        new_w = int(video.shape[3] * scale)
        video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
        y1 = (new_h - target_height) // 2
        x1 = (new_w - target_width) // 2
        video = video[:, :, y1:y1 + target_height, x1:x1 + target_width]

    video = video.permute(0, 2, 3, 1).div(255.0)
    return video


def _load_ref_image(image_path: Path, target_height: int, target_width: int) -> torch.Tensor:
    ref_image = Image.open(image_path).convert("RGB")
    ref_image = ref_image.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
    ref_np = np.array(ref_image).astype(np.float32) / 255.0
    ref_tensor = torch.from_numpy(ref_np).unsqueeze(0)
    return ref_tensor


def _run_inference(
    model,
    vae,
    prompt: str,
    pose_video_path: Path,
    ref_image_path: Path,
    seed: Optional[int],
) -> Tuple[Path, Path]:
    if seed is None:
        seed = SCAIL_FAST_SEED

    pose_video = _load_pose_video(pose_video_path, SCAIL_FAST_HEIGHT, SCAIL_FAST_WIDTH)
    ref_image = _load_ref_image(ref_image_path, SCAIL_FAST_HEIGHT, SCAIL_FAST_WIDTH)

    num_frames = int(pose_video.shape[0])

    image_embeds, = WanVideoEmptyEmbeds().process(
        num_frames=num_frames,
        width=SCAIL_FAST_WIDTH,
        height=SCAIL_FAST_HEIGHT,
    )

    image_embeds, = WanVideoAddSCAILPoseEmbeds().add(
        embeds=image_embeds,
        vae=vae,
        pose_images=pose_video,
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    )

    image_embeds, = WanVideoAddSCAILReferenceEmbeds().add(
        embeds=image_embeds,
        vae=vae,
        ref_image=ref_image,
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
        clip_embeds=None,
    )

    text_embeds, _, _ = WanVideoTextEncodeCached().process(
        model_name=SCAIL_FAST_T5_NAME,
        precision="bf16",
        positive_prompt=prompt or "",
        negative_prompt="",
        quantization="disabled",
        use_disk_cache=True,
        device="gpu",
    )

    scheduler, = WanVideoSchedulerv2().process(
        scheduler=SCAIL_FAST_SCHEDULER,
        steps=SCAIL_FAST_STEPS,
        shift=SCAIL_FAST_SHIFT,
        start_step=0,
        end_step=-1,
        sigmas=None,
        enhance_hf=False,
        unique_id=None,
    )

    samples, _ = WanVideoSamplerv2().process(
        model=model,
        image_embeds=image_embeds,
        text_embeds=text_embeds,
        cfg=SCAIL_FAST_CFG,
        seed=seed,
        force_offload=False,
        scheduler=scheduler,
    )

    decoded, = WanVideoDecode().decode(
        vae=vae,
        samples=samples,
        enable_vae_tiling=SCAIL_FAST_VAE_TILING,
        tile_x=SCAIL_FAST_VAE_TILE_X,
        tile_y=SCAIL_FAST_VAE_TILE_Y,
        tile_stride_x=SCAIL_FAST_VAE_STRIDE_X,
        tile_stride_y=SCAIL_FAST_VAE_STRIDE_Y,
    )

    video = decoded.permute(3, 0, 1, 2).mul(2).sub(1)

    output_path = pose_video_path.parent / "output.mp4"
    save_video(video, str(output_path), fps=16)

    concat_path = _create_concat_video(ref_image_path, output_path)
    return output_path, concat_path


def _maybe_post_webhook(webhook_url: Optional[str], payload: dict) -> None:
    if not webhook_url:
        return
    try:
        requests.post(webhook_url, json=payload, timeout=15)
    except Exception:
        pass


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def _capture_logs(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        tee_out = _Tee(sys.stdout, f)
        tee_err = _Tee(sys.stderr, f)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = tee_out, tee_err
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _tail_file(path: Path, max_bytes: int = 4096) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def handler(event):
    job_input = event.get("input", event)
    job_id = job_input.get("job_id") or event.get("id") or f"scail-fast-{uuid.uuid4().hex[:8]}"
    prompt = job_input.get("prompt") or ""
    pose_src = job_input.get("pose_video")
    ref_src = job_input.get("reference_image")
    upload_url = job_input.get("upload_url")
    upload_url_concat = job_input.get("upload_url_concat")
    upload_url_log = job_input.get("upload_url_log")
    webhook_url = job_input.get("webhook_url")
    seed = job_input.get("seed")

    if not pose_src or not ref_src:
        raise ValueError("Both 'pose_video' and 'reference_image' are required.")

    job_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}_", dir="/tmp"))
    pose_path = job_dir / "pose.mp4"
    ref_path = job_dir / "ref.jpg"

    _materialize_source(pose_src, pose_path, binary=True)
    _materialize_source(ref_src, ref_path, binary=True)

    result_payload = {
        "job_id": job_id,
        "status": "running",
        "output_path": None,
        "concat_path": None,
    }
    _maybe_post_webhook(webhook_url, result_payload)

    log_path = job_dir / "run.log"
    with _capture_logs(log_path):
        try:
            print(f"[job] id={job_id} prompt_len={len(prompt)} seed={seed}")
            print(f"[job] pose={pose_path} ref={ref_path}")
            print(f"[job] steps={SCAIL_FAST_STEPS} cfg={SCAIL_FAST_CFG} model={SCAIL_FAST_MODEL_NAME}")

            model, vae = _load_models()

            output_path, concat_path = _run_inference(
                model,
                vae,
                prompt,
                pose_path,
                ref_path,
                seed,
            )

            if upload_url:
                _upload_file(upload_url, output_path, content_type="video/mp4")
            if upload_url_concat and concat_path:
                _upload_file(upload_url_concat, concat_path, content_type="video/mp4")
            if upload_url_log:
                _upload_file(upload_url_log, log_path, content_type="text/plain")

            result_payload.update({
                "status": "succeeded",
                "output_path": str(output_path),
                "concat_path": str(concat_path),
                "seed": seed,
                "log_path": str(log_path),
            })
            _maybe_post_webhook(webhook_url, result_payload)
            return result_payload

        except Exception as exc:
            error_msg = str(exc)
            fail_payload = {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg,
                "traceback": traceback.format_exc(),
                "log_tail": _tail_file(log_path),
            }
            _maybe_post_webhook(webhook_url, fail_payload)
            if upload_url_log:
                _upload_file(upload_url_log, log_path, content_type="text/plain")
            raise


def health_check(_event=None):
    return {"status": "ok"}


if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "health_check": health_check,
        "concurrency_modifier": lambda _: 1,
    })
