#!/bin/bash
# Setup script for sallm environment
# Run this ONCE on a compute node (sintx) to build GPU packages and generate lockfile
#
# Usage:
#   sintx
#   cd ~/masters/sallm
#   bash scripts/setup_env.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export SCRATCH="${SCRATCH:-/scratch/lmbanr001}"
export HOME="${HOME:-/home/lmbanr001}"

# Use scratch for caches to avoid home quota issues
export UV_CACHE_DIR="$SCRATCH/.cache/uv"
export PIP_CACHE_DIR="$SCRATCH/.cache/pip"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

echo "=== Setting up sallm environment ==="
echo "Project dir: $PROJECT_DIR"

# Check we're on a compute node with GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Run this on a compute node (sintx)." >&2
    exit 1
fi
echo "GPU available:"
nvidia-smi -L

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    # Add to bashrc for future sessions
    if ! grep -q 'local/bin' ~/.bashrc; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
fi
echo "uv version: $(uv --version)"

# Load conda
echo "=== Loading conda ==="
module load python/miniconda3-py3.12
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create fresh conda env if it doesn't exist, or recreate if requested
if [[ "${RECREATE_ENV:-}" == "1" ]] || ! conda env list | grep -q "sallm-uv"; then
    echo "=== Creating fresh conda environment: sallm-uv ==="
    conda create -n sallm-uv python=3.12 -y
fi

set +u
conda activate sallm-uv
set -u

echo "Python: $(which python)"
echo "Python version: $(python --version)"

cd "$PROJECT_DIR"

# Install from lockfile (lockfile is pre-generated and committed to git)
echo "=== Installing from lockfile ==="
uv sync --frozen

# Install GPU packages (need torch in environment, use pip not uv pip)
echo "=== Installing GPU packages (this may take a while) ==="
pip install --no-cache-dir "mamba-ssm>=2.2.4" "causal-conv1d>=1.5.0" "kernels>=0.11.0"

# Install project as editable
echo "=== Installing sallm as editable ==="
uv pip install -e .

# Verify installation
echo "=== Verifying installation ==="
python -c "import sallm; print('sallm imported successfully')"
python -c "import kernels; print('kernels package installed')"
python -c "import mamba_ssm; print('mamba-ssm imported successfully')"
python -c "import causal_conv1d; print('causal-conv1d imported successfully')"

echo ""
echo "=== Setup complete ==="
echo "SLURM jobs will automatically sync dependencies using:"
echo "  conda activate sallm-uv"
echo "  uv sync --frozen"
