# Fix #4: Reference Image Format Error ✅

**Date:** December 24, 2025
**Error:** `AttributeError: 'Image' object has no attribute 'to'`

---

## The Problem

The SCAIL code at line 731 in `any2video.py` expects `input_ref_images` to be a list of **tensors**, not PIL Images:

```python
# SCAIL expects this to be a tensor
image_ref = input_ref_images[0].to(self.device)
```

But our handler was passing PIL Images:
```python
input_ref_images=[ref_image],  # ❌ PIL Image - wrong!
```

PIL Images don't have a `.to()` method, so it crashed.

---

## The Fix

**File:** `runpod_handler_wan2gp.py` (lines 277-293)

### Added import:
```python
from shared.utils.utils import convert_image_to_tensor
```

### Convert reference image to tensor:
```python
# Load and prepare reference image
ref_image_pil = Image.open(ref_image_path).convert("RGB")

# Convert reference image to tensor and add time dimension
# SCAIL expects: (C, 1, H, W) tensor, not PIL Image
ref_image_tensor = convert_image_to_tensor(ref_image_pil).unsqueeze(1)  # (C, H, W) -> (C, 1, H, W)
print(f"Reference image loaded: {ref_image_tensor.shape}")
```

### Updated generate() call:
```python
output = model.generate(
    input_prompt=prompt,
    input_ref_images=[ref_image_tensor],  # ✅ List of tensors: [(C, 1, H, W)]
    input_frames=pose_video_tensor.unsqueeze(0),
    ...
)
```

---

## Technical Details

### What `convert_image_to_tensor()` does:
1. Converts PIL Image (H, W, C) to numpy array
2. Normalizes to [-1, 1] range: `(pixel / 127.5) - 1`
3. Moves channel dimension to front: (H, W, C) → (C, H, W)

### Why add `unsqueeze(1)`:
- SCAIL expects reference images with time dimension
- Shape: (C, 1, H, W) means "RGB, 1 frame, Height, Width"
- The `1` represents a single reference frame

---

## All Fixes Summary

| Fix # | Issue | File | Status |
|-------|-------|------|--------|
| #1 | Missing `_interrupt` attribute | `any2video.py:95` | ✅ |
| #2 | Wrong `generate()` API usage | `runpod_handler_wan2gp.py` | ✅ |
| #3 | T5 device mismatch | `t5.py:676-678` | ✅ |
| #4 | Reference image format | `runpod_handler_wan2gp.py:277-293` | ✅ |

---

## Expected Progress So Far

With all 4 fixes, the RunPod log should show:

```
Loading SCAIL model (this may take a few minutes)...
t5.py: loading .../umt5-xxl/models_t5_umt5-xxl-enc-bf16.pth
clip.py: loading .../models_clip_open-clip-xlm-roberta-large-vit-huge-14-onlyvisual.pth
vae.py: loading .../Wan2.1_VAE.safetensors
SCAIL model loaded successfully!
Loading pose video from /tmp/scail-xxx/pose.mp4...
Pose video loaded: torch.Size([3, 60, 512, 896])
Reference image loaded: torch.Size([3, 1, H, W])  ⭐ NEW
Running SCAIL inference...
```

Next, the model should actually start running inference! 🎬

---

## Rebuild Required

```bash
cd /mnt/c/Users/ruben/rc_workspace/SCAIL
docker build -f Dockerfile.wan2gp -t <your-username>/scail-wan2gp:latest .
docker push <your-username>/scail-wan2gp:latest
```

Then update your RunPod endpoint and test again!
