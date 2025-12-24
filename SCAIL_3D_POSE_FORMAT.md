# SCAIL 3D Pose Data Format & Pipeline

## Overview

SCAIL is unique among pose-guided video generation models because it uses **3D pose information** rather than just 2D keypoints. This document explains the 3D pose format, rendering pipeline, and how to potentially integrate MediaPipe 3D pose data directly.

## The SCAIL Pose Pipeline

### Stage 1: 3D Pose Extraction (NLF Model)

**Location**: `wan2gp_integration/models/wan/scail/scail_pose_nlf.py:93-95`

```python
pred = model.detect_smpl_batched(frame_batch)
if "joints3d_nonparam" in pred:
    result_list.extend(pred["joints3d_nonparam"])
```

**Output Format**:
- **Shape**: `(24, 3)` per person per frame
- **Data**: 24 SMPL joints with (x, y, z) coordinates in 3D space
- **Coordinate System**: Camera-relative 3D coordinates (not world space)

**SMPL Joint Mapping** (24 joints):
```
0: pelvis           8: left ankle      16: right shoulder
1: right hip        9: left toe        17: left shoulder
2: left hip        10: right toe       18: right elbow
3: spine1          11: right toe       19: left elbow
4: right knee      12: neck            20: right wrist
5: left knee       13: right collar    21: left wrist
6: spine2          14: left collar     22: right hand
7: right ankle     15: head            23: left hand
```

### Stage 2: 3D Rendering (Taichi Cylinders)

**Location**: `wan2gp_integration/models/wan/scail/scail_pose_nlf.py:478-480`

```python
frames_np_rgba = render_whole(
    cylinder_specs_list,
    H=height,           # Output image height
    W=width,            # Output image width
    fx=focal_x,         # Camera focal length X
    fy=focal_y,         # Camera focal length Y
    cx=princpt[0],      # Camera principal point X
    cy=princpt[1],      # Camera principal point Y
    radius=21.5         # Cylinder radius for limbs
)
```

**Camera Intrinsics**:
- Default FOV: 55 degrees
- Focal length calculation: `f = image_size / (2 * tan(FOV/2))`
- Principal point: Usually `(width/2, height/2)`

**Rendering Process**:
1. Convert 3D joints to cylinder specifications (start/end points for each limb)
2. Use Taichi ray-marching to render cylinders with proper depth
3. Apply perspective projection using camera intrinsics
4. Output RGBA images where:
   - **RGB**: Rendered 3D pose with depth-aware shading
   - **Alpha**: Pose visibility mask

**Limb Connections** (drawn as cylinders):
```python
limb_seq = [
    [1, 2],   # Neck to left shoulder
    [1, 5],   # Neck to right shoulder
    [2, 3],   # Left shoulder to elbow
    [3, 4],   # Left elbow to hand
    [5, 6],   # Right shoulder to elbow
    [6, 7],   # Right elbow to hand
    [1, 8],   # Neck to spine
    [8, 9],   # Spine to left hip
    [9, 10],  # Left hip to knee
    [1, 11],  # Spine to right hip
    [11, 12], # Right hip to knee
    [12, 13], # Right knee to ankle
    # ... etc
]
```

### Stage 3: 2D Detail Overlay (DWPose)

**Location**: `wan2gp_integration/models/wan/scail/scail_pose_nlf.py:482-502`

```python
canvas_2d = draw_pose_to_canvas_np(
    aligned_poses,
    show_feet_flag=False,
    show_body_flag=False,   # Body already from 3D render
    show_cheek_flag=True,   # Add face details
    dw_hand=True,           # Add hand details
)
# Overlay 2D on 3D
frame_img[:, :, :3][mask] = canvas_img[mask]
```

**Purpose**: Add fine-grained 2D details that 3D pose doesn't capture:
- Face landmarks (68 points)
- Hand keypoints (21 points per hand)
- Facial expressions

### Stage 4: VAE Encoding

**Location**: `wan2gp_integration/models/wan/any2video.py:748-752`

```python
# Downsample pose video by 0.5x before VAE encoding
pose_pixels_ds = F.interpolate(
    pose_pixels_ds,
    size=(pose_pixels.shape[-2] // 2, pose_pixels.shape[-1] // 2),
    mode="bilinear"
)
pose_latents = self.vae.encode([pose_pixels_ds], VAE_tile_size)[0]
```

**Output Format**:
- **Shape**: `(B, 16, T, H/8, W/8)`
- **B**: Batch size (usually 1)
- **16**: VAE latent channels
- **T**: Temporal frames (downsampled by VAE stride)
- **H/8, W/8**: Spatial downsampling by VAE

**Key Point**: The 3D structure is encoded in these 16 latent channels!

### Stage 5: Diffusion Model Conditioning

**Location**: `wan2gp_integration/models/wan/any2video.py:755`

```python
kwargs.update({
    "scail_pose_latents": pose_latents,
    "ref_images_count": 1
})
```

The diffusion model receives these latents as conditioning and learns to:
1. Respect 3D pose geometry
2. Maintain temporal consistency
3. Apply character appearance from reference image

## Data Format Summary

| Stage | Format | Shape | Description |
|-------|--------|-------|-------------|
| NLF Output | 3D Joints | `(T, N, 24, 3)` | T frames, N people, 24 joints, xyz |
| 3D Render | RGBA Video | `(T, H, W, 4)` | RGB + Alpha mask |
| VAE Input | RGB Video | `(T, H/2, W/2, 3)` | Downsampled by 0.5x |
| VAE Output | Latents | `(1, 16, T', H/8, W/8)` | Encoded pose |
| Model Input | Latents | `(1, 16, T', H/8, W/8)` | Conditioning |

**T'**: Temporal dimension after VAE stride (usually `T/4 + 1`)

## MediaPipe Integration Path

### Option 1: Replace NLF with MediaPipe

If you have MediaPipe 3D pose data, you can bypass NLF extraction:

**MediaPipe Output Format**:
- 33 landmarks with (x, y, z, visibility)
- Normalized coordinates [0, 1] for x,y
- Relative depth for z

**Required Conversion**:
1. **Map MediaPipe landmarks → SMPL joints**:
   ```python
   # MediaPipe to SMPL mapping (approximate)
   mediapipe_to_smpl = {
       23: 0,  # left hip → pelvis (average with right)
       24: 0,  # right hip → pelvis
       26: 5,  # left knee
       25: 4,  # right knee
       28: 8,  # left ankle
       27: 7,  # right ankle
       12: 17, # left shoulder
       11: 16, # right shoulder
       14: 19, # left elbow
       13: 18, # right elbow
       16: 21, # left wrist
       15: 20, # right wrist
       # ... etc
   }
   ```

2. **Denormalize coordinates**:
   ```python
   # MediaPipe gives normalized [0,1] coords
   joints_3d[:, 0] *= image_width
   joints_3d[:, 1] *= image_height
   joints_3d[:, 2] *= depth_scale  # You'll need to calibrate this
   ```

3. **Adjust coordinate system**:
   MediaPipe uses different origin/scale than SMPL. You may need to:
   - Translate to center pelvis at origin
   - Scale to match SMPL bone lengths
   - Rotate to align with SMPL conventions

4. **Feed to renderer**:
   ```python
   from wan2gp_integration.models.wan.scail.scail_pose_nlf import render_nlf_as_images
   from wan2gp_integration.models.wan.scail.scail_pose_taichi_cylinder import render_whole

   # Create mock data structure matching NLF output
   data = [{
       "video_height": height,
       "video_width": width,
       "bboxes": None,
       "nlfpose": [torch.tensor(smpl_joints_3d)]  # Your converted joints
   } for frame in frames]

   # Render to RGBA
   frames_rgba = render_nlf_as_images(
       data,
       poses=None,  # Skip 2D overlay if you don't have it
       intrinsic_matrix=camera_intrinsics
   )
   ```

### Option 2: Direct Latent Input (Advanced)

If you can generate the RGBA pose videos yourself:

```python
# Your MediaPipe → 3D render pipeline
pose_rgba_video = your_mediapipe_renderer(skeleton_data)  # (T, H, W, 4)

# Convert to tensor
pose_pixels = torch.from_numpy(pose_rgba_video).permute(3, 0, 1, 2)  # (4, T, H, W)
pose_pixels = pose_pixels[:3]  # Take RGB only
pose_pixels = pose_pixels.float() / 127.5 - 1.0  # Normalize to [-1, 1]

# Downsample by 0.5x
pose_pixels_ds = F.interpolate(
    pose_pixels.permute(1, 0, 2, 3),
    size=(H // 2, W // 2),
    mode="bilinear"
).permute(1, 0, 2, 3)

# VAE encode
pose_latents = model.vae.encode([pose_pixels_ds], VAE_tile_size)[0].unsqueeze(0)

# Pass to inference
model.generate({
    "scail_pose_latents": pose_latents,
    "image_start": reference_image,
    # ... other params
})
```

## Key Considerations for MediaPipe Integration

### 1. Joint Count Mismatch
- **MediaPipe**: 33 landmarks (full body + face)
- **SMPL**: 24 joints (body only)
- **Solution**: Map MediaPipe subset to SMPL joints, interpolate missing joints

### 2. Coordinate System Differences
- **MediaPipe**: Normalized [0,1] with relative depth
- **SMPL**: Metric 3D coordinates (millimeters or meters)
- **Solution**: Calibrate depth scale, match coordinate conventions

### 3. Missing Joints
SMPL has some joints (spine1, spine2, collar bones) that MediaPipe doesn't directly provide:
- **Solution**: Interpolate from nearby landmarks
  ```python
  # Example: spine from hips and shoulders
  spine1 = (left_hip + right_hip + left_shoulder + right_shoulder) / 4
  ```

### 4. Camera Calibration
For proper 3D rendering, you need accurate camera intrinsics:
```python
def estimate_intrinsics(image_width, image_height, fov_degrees=55):
    fov_rad = np.radians(fov_degrees)
    focal = max(image_width, image_height) / (2 * np.tan(fov_rad / 2))
    return {
        'fx': focal,
        'fy': focal,
        'cx': image_width / 2,
        'cy': image_height / 2
    }
```

### 5. Temporal Consistency
MediaPipe can be jittery. Consider:
- Temporal smoothing (e.g., Kalman filter)
- Velocity constraints
- Physics-based constraints

## Benefits of Direct 3D Pose Input

1. **Skip expensive NLF inference** (~1.5GB model, significant compute)
2. **Better control** over pose generation
3. **Custom animations** from your skeleton system
4. **Real-time potential** with pre-generated poses
5. **Integration with existing pipeline** (your MediaPipe system)

## File Locations Reference

```
wan2gp_integration/
├── models/wan/
│   ├── any2video.py                          # Main inference, VAE encoding
│   └── scail/
│       ├── __init__.py                        # ScailPoseProcessor class
│       ├── scail_pose_nlf.py                  # 3D pose processing & rendering
│       ├── scail_pose_taichi_cylinder.py      # Taichi cylinder renderer
│       ├── scail_pose_draw.py                 # 2D overlay drawing
│       ├── model_scail.py                     # build_scail_pose_tokens()
│       └── nlf/                               # NLF pose extraction
│           └── multiperson_model.py
└── preprocessing/
    └── matanyone/                             # Multi-person segmentation
```

## Next Steps for Integration

1. **Test MediaPipe → SMPL conversion** with sample data
2. **Validate joint mapping** visually (render and compare)
3. **Calibrate depth scale** to match SMPL conventions
4. **Implement renderer interface** to bypass NLF
5. **Test with SCAIL inference** end-to-end
6. **Optimize for your use case** (real-time vs batch)

## Example Workflow

```python
# 1. Get MediaPipe 3D pose from your system
mediapipe_landmarks = your_mediapipe_system.process(video)  # (T, 33, 4)

# 2. Convert to SMPL format
smpl_joints = mediapipe_to_smpl_converter(mediapipe_landmarks)  # (T, 24, 3)

# 3. Render to RGBA video
camera_intrinsics = estimate_intrinsics(width, height)
pose_rgba = render_smpl_cylinders(smpl_joints, camera_intrinsics)  # (T, H, W, 4)

# 4. Encode with VAE
pose_latents = encode_pose_video(pose_rgba, vae_model)  # (1, 16, T', H/8, W/8)

# 5. Run SCAIL inference
output = scail_model.generate({
    "scail_pose_latents": pose_latents,
    "image_start": reference_image,
    "prompt": "character performing action",
    "num_frames": 81,
    "resolution": "896x512"
})
```

---

**Note**: This is an advanced integration that requires careful calibration. Start with the standard NLF pipeline first, then gradually replace components as you validate the MediaPipe conversion.
