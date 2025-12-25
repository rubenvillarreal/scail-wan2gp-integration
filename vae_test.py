#!/usr/bin/env python3
"""Test VAE within full SCAIL model to debug slow_conv3d_forward error"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "wan2gp_integration"))

import torch
from models.wan.any2video import WanAny2V
from shared.utils import files_locator as fl

# Config class (same as in runpod_handler_wan2gp.py)
class SCAILConfig:
    """Configuration class matching Wan2GP's config structure"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

# Set up file locator
MODEL_DIR = Path("/workspace/models")
fl.set_checkpoints_paths([str(MODEL_DIR)])

# SCAIL configuration
SCAIL_CONFIG = {
    "num_train_timesteps": 1000,
    "text_len": 512,
    "t5_dtype": torch.bfloat16,
    "clip_dtype": torch.bfloat16,
    "clip_checkpoint": "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth",
    "vae_stride": (4, 8, 8),
    "patch_size": (1, 2, 2),
    "param_dtype": torch.bfloat16,
    "sample_neg_prompt": "",
}

SCAIL_MODEL_DEF = {
    "URLs": [str(MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors")],
    "scail": True,
}

print("Loading SCAIL model...")
config = SCAILConfig(SCAIL_CONFIG)

model = WanAny2V(
    config=config,
    checkpoint_dir=str(MODEL_DIR),
    model_filename=[str(MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors")],
    submodel_no_list=[0],
    model_type="scail",
    model_def=SCAIL_MODEL_DEF,
    base_model_type="scail",
    text_encoder_filename=str(MODEL_DIR / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth"),
    quantizeTransformer=False,
    dtype=torch.bfloat16,
    VAE_dtype=torch.float32,
)

print("SCAIL model loaded!")

# Check VAE device and dtype
print(f"\nVAE device: {model.vae.device}")
print(f"VAE dtype: {model.vae.dtype}")
print(f"VAE model device: {next(model.vae.model.parameters()).device}")
print(f"VAE model dtype: {next(model.vae.model.parameters()).dtype}")

# Test VAE encode
print("\nTesting VAE from SCAIL model...")
test_tensor = torch.randn(3, 1, 512, 896).cuda()
try:
    latent = model.vae.encode([test_tensor], tile_size=0)[0]
    print(f"✅ SCAIL's VAE works: {latent.shape}")
except Exception as e:
    print(f"❌ SCAIL's VAE failed: {type(e).__name__}")
    print(f"   {str(e)[:200]}")
    import traceback
    traceback.print_exc()
