#!/bin/bash
# Setup script for RunPod GPU instance
# Run after SSH into the pod

set -e

echo "=== Installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

echo "=== Cloning repo ==="
cd /workspace
if [ ! -d "pulse-whisper" ]; then
    git clone https://github.com/Parassharmaa/pulse-whisper.git
fi
cd pulse-whisper

echo "=== Installing dependencies ==="
uv sync

echo "=== Verifying GPU ==="
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo "=== Setup complete! ==="
echo "Now run experiments with: uv run python scripts/<script>.py"
