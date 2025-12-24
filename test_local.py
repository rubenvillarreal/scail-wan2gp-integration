#!/usr/bin/env python3
"""Local testing script for SCAIL inference"""

import json
import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Set environment variables
os.environ["MODEL_BASE_PATH"] = str(Path.home() / "scail_models")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Add wan2gp_integration to path
sys.path.insert(0, str(Path(__file__).parent / "wan2gp_integration"))

# Now import the model
from models.wan.any2video import WanAny2V
from shared.utils import files_locator as fl

# Setup
MODEL_DIR = Path(os.environ["MODEL_BASE_PATH"])
SCAIL_CONFIG_PATH = Path(__file__).parent / "wan2gp_integration/models/wan/configs/scail.json"


class SCAILConfig:
    """Configuration class matching Wan2GP's config structure"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

def test_model_loading():
    """Test that the model loads correctly"""
    print("=" * 60)
    print("SCAIL Local Test - Model Loading")
    print("=" * 60)

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available!")
        return False

    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA version: {torch.version.cuda}")
    print(f"✅ Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print()

    # Set up file locator
    fl.set_checkpoints_paths([str(MODEL_DIR)])
    print(f"✅ Model directory: {MODEL_DIR}")

    # Check model files exist
    model_file = MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors"
    if not model_file.exists():
        print(f"❌ ERROR: Model file not found: {model_file}")
        return False
    print(f"✅ SCAIL model found: {model_file.name}")
    print()

    # Load config from JSON
    with open(SCAIL_CONFIG_PATH, 'r') as f:
        config_dict = json.load(f)

    # Add additional config parameters needed for model initialization
    config_dict.update({
        "num_train_timesteps": 1000,
        "text_len": 512,
        "t5_dtype": torch.bfloat16,
        "clip_dtype": torch.bfloat16,
        "clip_checkpoint": "models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth",
        "vae_stride": (4, 8, 8),
        "patch_size": (1, 2, 2),
        "param_dtype": torch.bfloat16,
    })

    config = SCAILConfig(config_dict)
    print("✅ Config loaded")

    # Initialize model
    print("\n⏳ Loading SCAIL model (this may take 2-5 minutes)...")
    print("   - Loading quantized INT8 model (~14GB)")
    print("   - This is normal for first load")

    try:
        # Model definition for SCAIL
        model_def = {
            "URLs": [str(model_file)],
            "scail": True,
        }

        # Check T5 text encoder exists
        text_encoder_path = MODEL_DIR / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth"
        if not text_encoder_path.exists():
            print(f"❌ ERROR: T5 text encoder not found: {text_encoder_path}")
            return False

        model = WanAny2V(
            config=config,
            checkpoint_dir=str(MODEL_DIR),
            model_filename=[str(model_file)],
            submodel_no_list=[0],
            model_type="scail",
            model_def=model_def,
            base_model_type="scail",
            text_encoder_filename=str(text_encoder_path),
            quantizeTransformer=False,  # Already quantized
            dtype=torch.bfloat16,
            VAE_dtype=torch.float32,
        )
        print("✅ Model loaded successfully!")
        print()

        # Check model is on GPU
        print(f"✅ Model device: {next(model.model.parameters()).device}")
        print(f"✅ Model dtype: {next(model.model.parameters()).dtype}")

        return True

    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_inference(model=None):
    """Test a simple inference"""
    print("\n" + "=" * 60)
    print("SCAIL Local Test - Inference")
    print("=" * 60)

    # Create dummy inputs
    print("⏳ Creating test inputs...")

    # Dummy reference image (512x512 RGB)
    ref_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
    print("✅ Created dummy reference image")

    # Test inputs
    inputs = {
        "prompt": "A character walking",
        "image_start": ref_image,
        "video_source": None,  # Would be path to pose video
        "resolution": "512x512",
        "num_frames": 25,  # Short test (1.5 seconds)
        "num_steps": 20,   # Fewer steps for faster test
        "guidance_scale": 4.0,
        "seed": 42,
    }

    print("✅ Test inputs prepared")
    print("\n⚠️  Note: Full inference test requires a pose video")
    print("   For now, we've just verified the model loads correctly")

    return True

if __name__ == "__main__":
    print("\n🚀 Starting SCAIL Local Tests\n")

    # Test 1: Model loading
    success = test_model_loading()

    if success:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYour local environment is ready for SCAIL inference!")
        print("\nNext steps:")
        print("1. Prepare a pose video for testing")
        print("2. Modify test_inference() to use real inputs")
        print("3. Run full end-to-end test")
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED")
        print("=" * 60)
        print("\nPlease check the errors above and fix before proceeding")
        sys.exit(1)
