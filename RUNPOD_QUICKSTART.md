# SCAIL RunPod Quick Start Guide

## Initial Setup on Fresh RunPod Instance

### 1. Clone the Repository
```bash
cd /workspace
git clone https://github.com/rubenvillarreal/scail-wan2gp-integration.git SCAIL
cd SCAIL
```

### 2. Clone Wan2GP Integration
```bash
git clone https://github.com/Wan-Video/Wan2.1.git wan2gp_integration
```

### 3. Install Dependencies

**IMPORTANT**: Install PyTorch 2.6.0+cu124 FIRST (required for optimum-quanto compatibility):

```bash
# Step 1: Install PyTorch 2.6.0 with CUDA 12.4
pip install --force-reinstall \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 2: Install numpy 2.x (required by optimum-quanto)
pip install "numpy>=2.0,<2.3"

# Step 3: Install all other requirements
pip install -r requirements_wan2gp.txt

# Step 4: Install RunPod SDK
pip install runpod
```

**Or use the automated script**:
```bash
bash install_dependencies.sh
```

### 4. Verify Installation
```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
```

Expected output:
```
PyTorch: 2.6.0+cu124
CUDA: True
cuDNN: 90300
```

### 5. Download Model Checkpoints

Set up model directory (or use RunPod network volume):
```bash
export MODEL_BASE_PATH=/runpod-volume/models
mkdir -p $MODEL_BASE_PATH
```

Download SCAIL quantized model:
```bash
# Download from Hugging Face or ModelScope
# Place the following files in $MODEL_BASE_PATH:
# - wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors
# - Wan2.1_VAE.safetensors
# - umt5-xxl/ (directory with T5 encoder)
# - models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
```

### 6. Test Inference

```bash
python test_inference.py
```

This will run inference on the example files in `examples/001/` and `examples/002/`.

## Troubleshooting

### Conv3d Error
If you see `NotImplementedError: slow_conv3d_forward not available on CUDA backend`:
- Verify CUDA and cuDNN are available
- Check PyTorch version is 2.6.0+cu124
- Try rebuilding PyTorch or using CUDA 12.4 runtime image

### Dependency Conflicts
- **optimum-quanto requires torch>=2.6.0**: Install PyTorch 2.6.0+cu124 first
- **numpy version conflicts**: Install numpy 2.x before other requirements
- **InsightFace compilation fails**: Ensure `python3-dev` is installed

### CUDA Out of Memory
- Reference image should be resized to 512x896 (done automatically in handler)
- VAE tiling must be enabled (`VAE_tile_size=256` in generate call)
- Target resolution: 512x896 for 48GB VRAM

## Running RunPod Serverless Handler

```bash
python runpod_handler_wan2gp.py
```

The handler expects input in the following format:
```json
{
    "prompt": "the girl is dancing",
    "pose_video": "<url or base64>",
    "reference_image": "<url or base64>",
    "seed": 123,
    "upload_url": "<presigned PUT URL for output>",
    "upload_url_concat": "<presigned PUT URL for concat output>",
    "webhook_url": "<status callback URL>"
}
```

## Hardware Requirements

- **Recommended GPU**: RTX 6000 Ada (48GB VRAM) or A100 (40GB+)
- **CUDA**: 12.1+ drivers
- **Storage**: ~30GB for models + dependencies
- **RAM**: 32GB+ recommended

## Key Files

- `runpod_handler_wan2gp.py` - RunPod serverless handler
- `test_inference.py` - Test script for local inference
- `requirements_wan2gp.txt` - Python dependencies
- `install_dependencies.sh` - Automated installation script
- `Dockerfile.wan2gp` - Docker image for deployment
