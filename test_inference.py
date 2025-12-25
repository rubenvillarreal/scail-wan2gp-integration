#!/usr/bin/env python3
"""
Test SCAIL inference with example files
Usage: python test_inference.py
"""

import sys
from pathlib import Path

# Add wan2gp_integration to path
sys.path.insert(0, str(Path(__file__).parent / "wan2gp_integration"))

from runpod_handler_wan2gp import get_model, _run_inference

def test_example_001():
    """Test with example 001"""
    print("\n" + "="*60)
    print("Testing SCAIL inference with example 001")
    print("="*60 + "\n")

    # Paths to example files (prefer rendered pose if available)
    pose_video = Path("examples/001/rendered.mp4")
    if not pose_video.exists():
        pose_video = Path("examples/001/driving.mp4")
    ref_image = Path("examples/001/ref.jpg")

    if not pose_video.exists():
        print(f"❌ Error: {pose_video} not found")
        return False
    if not ref_image.exists():
        print(f"❌ Error: {ref_image} not found")
        return False

    print(f"📹 Pose video: {pose_video}")
    print(f"🖼️  Reference image: {ref_image}")
    print(f"💬 Prompt: 'a person dancing'\n")

    try:
        # Load model (this happens once and is cached)
        print("Loading SCAIL model...")
        model = get_model()
        print("✅ Model loaded!\n")

        # Run inference
        print("Running inference...")
        output_path, concat_path = _run_inference(
            model=model,
            prompt="a person dancing",
            pose_video_path=pose_video,
            ref_image_path=ref_image,
            seed=42
        )

        print(f"\n✅ Inference completed!")
        print(f"📁 Output video: {output_path}")
        print(f"📁 Concatenated video: {concat_path}")

        # Check if files exist
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
        if concat_path.exists():
            size_mb = concat_path.stat().st_size / (1024 * 1024)
            print(f"   Concat size: {size_mb:.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Error during inference:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_example_002():
    """Test with example 002"""
    print("\n" + "="*60)
    print("Testing SCAIL inference with example 002")
    print("="*60 + "\n")

    # Paths to example files (prefer rendered pose if available)
    pose_video = Path("examples/002/rendered.mp4")
    if not pose_video.exists():
        pose_video = Path("examples/002/driving.mp4")
    ref_image = Path("examples/002/ref.jpg")

    if not pose_video.exists():
        print(f"❌ Error: {pose_video} not found")
        return False
    if not ref_image.exists():
        print(f"❌ Error: {ref_image} not found")
        return False

    print(f"📹 Pose video: {pose_video}")
    print(f"🖼️  Reference image: {ref_image}")
    print(f"💬 Prompt: 'a woman dancing'\n")

    try:
        # Model should already be loaded from previous test
        print("Getting SCAIL model (should be cached)...")
        model = get_model()
        print("✅ Model ready!\n")

        # Run inference
        print("Running inference...")
        output_path, concat_path = _run_inference(
            model=model,
            prompt="a woman dancing",
            pose_video_path=pose_video,
            ref_image_path=ref_image,
            seed=123
        )

        print(f"\n✅ Inference completed!")
        print(f"📁 Output video: {output_path}")
        print(f"📁 Concatenated video: {concat_path}")

        # Check if files exist
        if output_path.exists():
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.2f} MB")
        if concat_path.exists():
            size_mb = concat_path.stat().st_size / (1024 * 1024)
            print(f"   Concat size: {size_mb:.2f} MB")

        return True

    except Exception as e:
        print(f"\n❌ Error during inference:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import torch

    # Print environment info
    print("\n" + "="*60)
    print("Environment Information")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")

    # Test Conv3d on CUDA
    print("\nTesting Conv3d on CUDA...")
    try:
        conv = torch.nn.Conv3d(3, 16, 3, padding=1).cuda()
        x = torch.randn(1, 3, 4, 32, 32).cuda()
        out = conv(x)
        print(f"✅ Conv3d test passed: {out.shape}")
    except Exception as e:
        print(f"❌ Conv3d test failed: {e}")
        sys.exit(1)

    # Run tests
    print("\n")
    success_001 = test_example_001()

    if success_001:
        print("\n")
        success_002 = test_example_002()
    else:
        success_002 = False

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Example 001: {'✅ PASSED' if success_001 else '❌ FAILED'}")
    print(f"Example 002: {'✅ PASSED' if success_002 else '❌ FAILED'}")

    if success_001 and success_002:
        print("\n🎉 All tests passed! SCAIL inference is working correctly.")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)
