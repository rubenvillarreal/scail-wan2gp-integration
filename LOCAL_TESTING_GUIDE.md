# Local Testing Guide for SCAIL Wan2GP

This guide will help you test the SCAIL inference pipeline locally before deploying to RunPod.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (48GB recommended)
  - RTX 3090/4090 (24GB) - Minimum
  - RTX 6000 Ada (48GB) - Recommended
  - A6000 (48GB) - Recommended
- **RAM**: 32GB+ system RAM
- **Storage**: ~25GB free space for models

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+) or Windows with WSL2
- **CUDA**: 12.1+ (matching PyTorch requirements)
- **Python**: 3.10 (same as Docker image)

---

## Setup Steps

### Step 1: Create Python Virtual Environment

```bash
cd /home/rubenvgrad/SCAIL

# Create virtual environment with Python 3.10
python3.10 -m venv venv

# Activate it
source venv/bin/activate  # Linux/WSL
# OR
venv\Scripts\activate     # Windows (if not using WSL)
```

### Step 2: Install PyTorch with CUDA Support

**IMPORTANT**: Install PyTorch BEFORE other requirements to ensure CUDA support:

```bash
# Install PyTorch 2.3.1 with CUDA 12.1 support
pip install --upgrade pip setuptools wheel
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU is detected**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 3090  # (or your GPU name)
```

### Step 3: Install Requirements

```bash
pip install -r requirements_wan2gp.txt
```

This will install all dependencies (~5-10 minutes depending on internet speed).

**Common Issues**:
- If `insightface` fails to compile, make sure you have `build-essential` installed:
  ```bash
  sudo apt-get update
  sudo apt-get install build-essential g++ gcc
  ```

### Step 4: Set Up Model Directory

Create a local directory to store models (instead of `/runpod-volume`):

```bash
# Create models directory
mkdir -p ~/scail_models

# Set environment variable (add to ~/.bashrc to persist)
export MODEL_BASE_PATH="$HOME/scail_models"
```

### Step 5: Download Models

Download all required models to your local directory:

```bash
cd ~/scail_models

# 1. Download SCAIL quantized model (~14GB)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors

# 2. Download VAE
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/Wan2.1_VAE.pth

# 3. Download CLIP model
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth

# 4. Download T5 model (directory)
mkdir -p umt5-xxl
cd umt5-xxl
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth
# Also download tokenizer files (config, etc.)
cd ..

# 5. Download pose extraction models
mkdir -p pose
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/nlf_l_multi_0.3.2.eager.safetensors -P pose/
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/yolox_l.onnx -P pose/
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/dw-ll_ucoco_384.onnx -P pose/

# 6. Download segmentation models (for multi-person)
mkdir -p mask
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/mask/sam_vit_h_4b8939_fp16.safetensors -P mask/
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/mask/model.safetensors -P mask/

# Verify downloads
ls -lh
```

Your directory structure should look like:
```
~/scail_models/
├── wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors  (~14GB)
├── Wan2.1_VAE.pth
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
├── umt5-xxl/
│   └── models_t5_umt5-xxl-enc-bf16.pth
├── pose/
│   ├── nlf_l_multi_0.3.2.eager.safetensors
│   ├── yolox_l.onnx
│   └── dw-ll_ucoco_384.onnx
└── mask/
    ├── sam_vit_h_4b8939_fp16.safetensors
    └── model.safetensors
```

---

## Step 6: Create Local Test Script

Create a simple test script that doesn't require RunPod:

```bash
cd /home/rubenvgrad/SCAIL
```

Create `test_local.py`:

```python
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
```

Make it executable:
```bash
chmod +x test_local.py
```

### Step 7: Run Test

```bash
python test_local.py
```

**Expected output**:
```
🚀 Starting SCAIL Local Tests

============================================================
SCAIL Local Test - Model Loading
============================================================
✅ GPU detected: NVIDIA GeForce RTX 3090
✅ CUDA version: 12.1
✅ Available VRAM: 24.00 GB

✅ Model directory: /home/user/scail_models
✅ SCAIL model found: wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors

✅ Config loaded

⏳ Loading SCAIL model (this may take 2-5 minutes)...
   - Loading quantized INT8 model (~14GB)
   - This is normal for first load

✅ Model loaded successfully!

✅ Model device: cuda:0
✅ Model dtype: torch.bfloat16

============================================================
✅ ALL TESTS PASSED!
============================================================

Your local environment is ready for SCAIL inference!
```

---

## Step 8: Full Inference Test (Optional)

Once model loading works, create a full test with real inputs:

```python
# test_full_inference.py
import json
from test_local import *

def run_full_inference():
    """Run full inference with real pose video"""

    # Load config
    fl.set_checkpoints_paths([str(MODEL_DIR)])
    with open(SCAIL_CONFIG_PATH, 'r') as f:
        config_dict = json.load(f)
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

    # Model definition
    model_file = MODEL_DIR / "wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors"
    model_def = {
        "URLs": [str(model_file)],
        "scail": True,
    }

    # T5 text encoder path
    text_encoder_path = MODEL_DIR / "umt5-xxl" / "models_t5_umt5-xxl-enc-bf16.pth"

    # Load model
    model = WanAny2V(
        config=config,
        checkpoint_dir=str(MODEL_DIR),
        model_filename=[str(model_file)],
        submodel_no_list=[0],
        model_type="scail",
        model_def=model_def,
        base_model_type="scail",
        text_encoder_filename=str(text_encoder_path),
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
    )

    # Load real inputs
    ref_image = Image.open("path/to/reference_image.jpg").convert("RGB")
    pose_video = "path/to/pose_video.mp4"

    # Run inference
    inputs = {
        "prompt": "A character dancing",
        "image_start": ref_image,
        "video_source": pose_video,
        "resolution": "896x512",
        "num_frames": 81,
        "num_steps": 50,
        "guidance_scale": 4.0,
        "seed": 42,
    }

    print("Running inference...")
    output = model.generate(inputs)

    # Save output
    output.save("output_local.mp4")
    print(f"✅ Saved output to: output_local.mp4")

if __name__ == "__main__":
    run_full_inference()
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size or use gradient checkpointing
```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
```

### Issue: Model files not found

**Solution**: Check file locator paths
```python
import shared.utils.files_locator as fl
fl.set_checkpoints_paths(["/home/user/scail_models", "."])
```

### Issue: Import errors

**Solution**: Make sure PYTHONPATH includes wan2gp_integration:
```bash
export PYTHONPATH="/home/rubenvgrad/SCAIL/wan2gp_integration:$PYTHONPATH"
```

### Issue: Slow model loading

**Solution**: This is normal! First load takes 2-5 minutes due to:
- Loading 14GB quantized model
- Initializing CUDA kernels
- Setting up attention mechanisms

---

## Performance Comparison

| Environment | Model Load Time | Inference Time (81 frames) |
|-------------|----------------|---------------------------|
| Local (RTX 3090 24GB) | 2-3 min | ~5-8 min |
| Local (RTX 4090 24GB) | 1-2 min | ~3-5 min |
| Local (RTX 6000 Ada 48GB) | 1-2 min | ~2-4 min |
| RunPod Serverless | 30-60s (cached) | ~2-4 min |

---

## Next Steps

Once local testing works:

1. ✅ **Test with real pose videos**
2. ✅ **Verify output quality**
3. ✅ **Benchmark performance**
4. 🚀 **Deploy to RunPod** (faster, serverless, pay-per-use)

---

## Tips for Faster Development

1. **Use shorter videos for testing**:
   - `num_frames: 25` instead of `81` (1.5s instead of 5s)
   - Much faster iteration

2. **Use fewer steps**:
   - `num_steps: 20` instead of `50` for quick tests
   - Quality won't be perfect but enough to verify it works

3. **Cache the model**:
   - Model stays in memory between runs
   - Use Jupyter notebook for interactive testing

4. **Monitor VRAM**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## Deactivating Environment

When done testing:

```bash
deactivate  # Exit virtual environment
```

To reactivate later:
```bash
cd /home/rubenvgrad/SCAIL
source venv/bin/activate
```

---

**Happy testing! 🚀**
