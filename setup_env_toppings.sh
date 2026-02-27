#!/bin/bash
# Setup script for ae-296 environment
# Assumes: conda is available, repo is cloned, and you are in the ae-296 directory.
# Usage: cd ae-296 && bash setup_env.sh

set -e

apt update && apt install -y numactl

ENV_NAME="ae296"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/models/huggyllama-llama-7b"

echo "=== [1/10] Creating conda environment '${ENV_NAME}' with Python 3.10 ==="
conda create -n "${ENV_NAME}" python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== [2/10] Installing PyTorch 2.1.0 with CUDA 12.1 ==="
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "=== [3/10] Installing CUDA 12.1 toolkit (nvcc) via conda ==="
# Install cuda-toolkit and explicitly pin cuda-nvcc to 12.1 to avoid version drift
conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit cuda-nvcc=12.1 -y

echo "=== [4/10] Installing Python dependencies ==="
# Install requirements but skip packages we handle separately:
#   - torch (already installed with correct CUDA)
#   - punica==0.2.0 (wrong package on PyPI; correct one installed below)
#   - numpy==1.25.1 (needs relaxed version for compatibility)
pip install numpy==1.25.2
grep -v -E '^(torch==|punica==|numpy==)' "${SCRIPT_DIR}/requirements.txt" \
    | grep -v '^\s*#' \
    | grep -v '^\s*$' \
    | pip install -r /dev/stdin || echo "WARNING: Some optional packages failed to install. Continuing..."

echo "=== [5/10] Installing punica (LoRA CUDA kernels) ==="
# The correct punica is punica-ai, NOT the Ontology DApp 'punica' from PyPI
pip install punica -i https://punica-ai.github.io/whl/cu121/ --extra-index-url https://pypi.org/simple

echo "=== [6/10] Building and installing tensor_cp (CUDA extension) ==="
export CUDA_HOME="${CONDA_PREFIX}"
export LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH:-}"
cd "${SCRIPT_DIR}/csrc"
pip install .
cd "${SCRIPT_DIR}"

echo "=== [7/10] Installing lightllm package ==="
pip install -e "${SCRIPT_DIR}"

echo "=== [8/10] Applying code patches ==="
# Fix multiprocessing start method error (context already set by transformers)
sed -i "s/set_start_method('spawn')/set_start_method('spawn', force=True)/" \
    "${SCRIPT_DIR}/lightllm/server/api_server.py" 2>/dev/null || true

echo "=== [9/10] Installing Hugging Face CLI and downloading model ==="
curl -LsSf https://hf.co/cli/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "${MODEL_DIR}"
hf download huggyllama/llama-7b

echo "=== [10/10] Updating config files with model path ==="
for yml_file in "${SCRIPT_DIR}"/config/*.yml; do
    sed -i "s|model_dir:.*|model_dir: '${MODEL_DIR}'|" "${yml_file}"
    echo "  Updated: ${yml_file}"
done

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo ""
echo "Before running the server, activate the environment and set library paths:"
echo ""
echo "  conda activate ${ENV_NAME}"
echo "  export LD_LIBRARY_PATH=\$(python -c 'import torch; print(torch.__path__[0])')/lib:\$LD_LIBRARY_PATH"
echo "  export CUDA_HOME=\$CONDA_PREFIX"
echo ""
echo "Then start the server, e.g.:"
echo "  python3 runServerCPULoRA.py ./config/ali-a10.yml"
echo ""
echo "TIP: Add the export lines to your ~/.bashrc to avoid repeating them."
