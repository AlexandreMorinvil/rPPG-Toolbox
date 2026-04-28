#!/bin/bash

# Modernized for Python 3.12. Original toolbox targeted Python 3.8.
# On Windows, run the equivalent commands manually in PowerShell or use WSL;
# in particular, `mamba-ssm` / `causal-conv1d` (PhysMamba) only build on Linux.

# Check if a mode argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 {conda|uv}"
    exit 1
fi

MODE=$1

# Function to set up using conda
conda_setup() {
    echo "Setting up using conda..."
    conda remove --name rppg-toolbox --all -y || exit 1
    conda create -n rppg-toolbox python=3.12 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh" || exit 1
    conda activate rppg-toolbox || exit 1
    pip install --upgrade pip setuptools wheel || exit 1
    pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 || exit 1
    pip install -r requirements.txt || exit 1
    # PhysMamba (Linux + CUDA only). Comment out if not needed.
    if [[ "$(uname -s)" == "Linux" ]]; then
        pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2" || \
            echo "WARNING: mamba-ssm install failed; PhysMamba will be disabled."
    else
        echo "INFO: Skipping mamba-ssm install (non-Linux platform); PhysMamba disabled."
    fi
}

# Function to set up using uv
uv_setup() {
    rm -rf .venv || exit 1
    uv venv --python 3.12 || exit 1
    source .venv/bin/activate || exit 1
    uv pip install --upgrade pip setuptools wheel || exit 1
    uv pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 || exit 1
    uv pip install -r requirements.txt || exit 1
    if [[ "$(uname -s)" == "Linux" ]]; then
        uv pip install "causal-conv1d>=1.4.0" "mamba-ssm>=2.2.2" || \
            echo "WARNING: mamba-ssm install failed; PhysMamba will be disabled."
    else
        echo "INFO: Skipping mamba-ssm install (non-Linux platform); PhysMamba disabled."
    fi
    # Explicitly install PyQt5 to use interactive plotting and avoid non-interactive backends
    # See this relevant issue for more details: https://github.com/astral-sh/uv/issues/6893
    uv pip install PyQt5
}

# Execute the appropriate setup based on the mode
case $MODE in
    conda)
        conda_setup
        ;;
    uv)
        uv_setup
        ;;
    *)
        echo "Invalid mode: $MODE"
        echo "Usage: $0 {conda|uv}"
        exit 1
        ;;
esac
