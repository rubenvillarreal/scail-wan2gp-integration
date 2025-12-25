#!/bin/bash
# Install dependencies for SCAIL on RunPod GPU template
# Run this on a fresh RunPod instance with CUDA 12.1+

set -e  # Exit on error

echo "Installing system dependencies..."
apt-get update
apt-get install -y --no-install-recommends ffmpeg
rm -rf /var/lib/apt/lists/*

echo "Installing PyTorch 2.6.0 with CUDA 12.4 (compatible with CUDA 12.1+ drivers)..."
pip install --force-reinstall \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing numpy 2.x (required by optimum-quanto)..."
pip install "numpy>=2.0,<2.3"

echo "Installing requirements from requirements_wan2gp.txt..."
pip install -r requirements_wan2gp.txt

echo "Installing RunPod SDK..."
pip install runpod

echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"

echo "✅ Dependencies installed successfully!"
