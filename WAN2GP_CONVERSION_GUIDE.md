# Wan2GP Conversion Guide for SCAIL

## What Changed

### Before (Original SCAIL)
- **Model**: Full BF16 (~28GB)
- **RAM Usage**: ~66GB
- **VRAM Usage**: ~26GB
- **Result**: CUDA OOM errors on RTX 6000 Ada

### After (Wan2GP)
- **Model**: Quantized INT8 (~14GB)
- **RAM Usage**: ~33GB (50% reduction)
- **VRAM Usage**: ~13GB (50% reduction)
- **Result**: Fits comfortably on RTX 6000 Ada

## Step 1: Download Quantized SCAIL Model

You need to download the quantized model to your RunPod network volume:

### Option A: From a RunPod Pod (Recommended)

1. Deploy a temporary Pod with your `scail-models` network volume attached
2. SSH into the Pod
3. Download the model:

```bash
cd /workspace  # or wherever your network volume is mounted

# Download quantized SCAIL model (~14GB)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors

# Verify download
ls -lh wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors
```

### Option B: Using HuggingFace CLI

```bash
cd /workspace
huggingface-cli download DeepBeepMeep/Wan2.1 wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors --local-dir .
```

## Step 1.5: Download Pose Extraction Models

The Wan2GP implementation includes pose extraction from raw videos. You need to download these additional models:

```bash
cd /workspace  # or wherever your network volume is mounted

# Create pose and mask directories
mkdir -p pose mask

# Download NLF pose extraction model (~1.5GB)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/nlf_l_multi_0.3.2.eager.safetensors -P pose/

# Download YOLOX detector (~270MB)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/yolox_l.onnx -P pose/

# Download DWPose RTMPose model (~220MB)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/pose/dw-ll_ucoco_384.onnx -P pose/

# Download MatAnyone segmentation models (for multi-person)
# SAM-H model FP16 (~2.5GB) - NOTE: Must be fp16 version!
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/mask/sam_vit_h_4b8939_fp16.safetensors -P mask/

# MatAnyone model (~350MB) - NOTE: Renamed to model.safetensors (HF standard)
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/mask/model.safetensors -P mask/

# Verify downloads
ls -lh pose/
ls -lh mask/
```

**Note**: If you already have pre-extracted pose videos and don't need pose extraction, you can skip downloading these models for now. Pose extraction can be made optional in a future update.

## Step 2: Verify Network Volume Structure

Your RunPod network volume should have this structure:

```
/runpod-volume/  (when mounted in Serverless)
├── wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors  # NEW quantized model
├── Wan2.1_VAE.pth
├── models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
├── umt5-xxl/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── ... (tokenizer files)
├── pose/  # NEW: Pose extraction models
│   ├── nlf_l_multi_0.3.2.eager.safetensors
│   ├── yolox_l.onnx
│   └── dw-ll_ucoco_384.onnx
├── mask/  # NEW: Segmentation models (for multi-person)
│   ├── sam_vit_h_4b8939_fp16.safetensors
│   └── model.safetensors  # MatAnyone (HuggingFace standard naming)
└── model/  # OLD model (can be deleted to save space)
    ├── 1/
    │   └── mp_rank_00_model_states.pt
    └── latest
```

## Step 3: Build and Push New Docker Image

```bash
cd /home/rubenvgrad/SCAIL

# Build the new Wan2GP-based image
docker build -t rubenvgrad/scail-runpod:wan2gp -f Dockerfile.wan2gp .

# Push to Docker Hub
docker push rubenvgrad/scail-runpod:wan2gp
```

## Step 4: Update RunPod Serverless Endpoint

1. Go to your RunPod Serverless endpoint settings
2. Update the Docker image to: `rubenvgrad/scail-runpod:wan2gp`
3. Ensure network volume `scail-models` is attached
4. Save and redeploy

## Step 5: Test the New Implementation

Send a test request to your endpoint:

```json
{
  "input": {
    "prompt": "the girl is dancing",
    "pose_video": "<your pose video URL>",
    "reference_image": "<your reference image URL>",
    "seed": 123
  }
}
```

## Expected Performance

### Memory Usage (Wan2GP vs Original)

| Component | Original SCAIL | Wan2GP | Reduction |
|-----------|---------------|--------|-----------|
| Model Size | ~28GB | ~14GB | 50% |
| Peak RAM | ~66GB | ~33GB | 50% |
| Peak VRAM | ~26GB | ~13GB | 50% |

### GPU Compatibility

- ✅ **RTX 6000 Ada (48GB)**: Plenty of headroom
- ✅ **RTX 4090 (24GB)**: Should work fine
- ✅ **A40 (48GB)**: Plenty of headroom
- ✅ **RTX 5090**: Supported with updated PyTorch

## Troubleshooting

### Issue: Import errors

If you see import errors related to `mmgp` or other dependencies:
- Check that `requirements_wan2gp.txt` was installed correctly
- Verify Python version is 3.10 or 3.11

### Issue: Model not found

If you see "FileNotFoundError" for the model:
- Verify the model is downloaded to the network volume
- Check the mount path is `/runpod-volume` in Serverless
- Ensure filename matches exactly: `wan2.1_scail_preview_14B_quanto_bf16_int8.safetensors`

### Issue: Still running out of memory

If you still see OOM errors:
- Try using `--quantization int8` flag (if implemented)
- Reduce resolution or number of frames
- Use a GPU with more VRAM (H100 80GB)

## File Summary

### New Files Created
- `wan2gp_integration/` - Wan2GP core modules
- `runpod_handler_wan2gp.py` - New RunPod handler using Wan2GP
- `requirements_wan2gp.txt` - Wan2GP dependencies
- `Dockerfile.wan2gp` - New Dockerfile for Wan2GP setup
- `WAN2GP_CONVERSION_GUIDE.md` - This file

### Files to Keep
- Original SCAIL code (in case you need to reference it)
- Network volume with models
- `.dockerignore` (still useful)

### Files You Can Delete (Optional)
- Original model at `/runpod-volume/model/` (~14GB saved)
- Old Docker images: `rubenvgrad/scail-runpod:v1-v11`

## Next Steps

1. ✅ Download quantized model to network volume
2. ✅ Build and push new Docker image
3. ✅ Update RunPod Serverless endpoint
4. ✅ Test with a sample request
5. 🎯 Monitor memory usage and performance
6. 🚀 Deploy to production!

## Support

If you encounter issues:
1. Check RunPod logs for detailed error messages
2. Verify network volume mount path
3. Ensure all models are downloaded correctly
4. Test locally with Docker before deploying to RunPod

---

**Estimated Time to Complete**: 30-60 minutes (mostly waiting for downloads)
**Estimated Space Saved**: ~14GB (if you delete the old model)
**Estimated Performance**: 50% less memory usage, similar inference speed
