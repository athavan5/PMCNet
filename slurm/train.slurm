#!/bin/bash
#SBATCH --job-name=train_PMCNet_idrid-1440x960-seed-87 
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --gpus=2
#SBATCH --output=/scratch/st-ipor-1/lwqz/PMCNet/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/lwqz/PMCNet/hpc_logs/%x-%j.err
#SBATCH --mail-user=lwqz@student.ubc.ca
#SBATCH --mail-type=FAIL,END               

set -euo pipefail

module purge
module load intel-oneapi-compilers/2023.1.0
module load cuda/11.8.0
module load cudnn/8.6.0.163-11.8

# Build runtime library path explicitly for TF dlopen()
for p in \
  "${CUDA_HOME:-}/lib64" \
  "${CUDA_PATH:-}/lib64" \
  "${CUDNN_HOME:-}/lib64" \
  "${EBROOTCUDA:-}/lib64" \
  "${EBROOTCUDNN:-}/lib64" \
  "/usr/local/cuda/lib64"
do
  if [[ -n "$p" && -d "$p" ]]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done

# Navigate to the job's working directory
REPO_DIR="/scratch/st-ipor-1/lwqz/PMCNet"
cd "${REPO_DIR}"

# Conda environment setup
# ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet"
ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Debug: confirm conda/python/tensorflow resolution in this Slurm job
echo "[DEBUG] which python: $(which python)"
"$ENV_PREFIX/bin/python" -V
"$ENV_PREFIX/bin/python" -c "import sys; print('[DEBUG] sys.executable:', sys.executable)"
"$ENV_PREFIX/bin/python" -c "import importlib.util; print('[DEBUG] tensorflow spec:', bool(importlib.util.find_spec('tensorflow')))"
# "$ENV_PREFIX/bin/python" -m pip show tensorflow | head -n 5
###############################################################################
# Device Configuration
###############################################################################
# Ensure we see all GPUs if not set by SLURM
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[DEBUG] CUDA_HOME=${CUDA_HOME:-unset}"
echo "[DEBUG] CUDNN_HOME=${CUDNN_HOME:-unset}"
echo "[DEBUG] EBROOTCUDA=${EBROOTCUDA:-unset}"
echo "[DEBUG] EBROOTCUDNN=${EBROOTCUDNN:-unset}"
echo "[DEBUG] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"

# Fail fast if NVIDIA driver is not visible in this job
echo "[DEBUG] host: $(hostname)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found in PATH. GPU driver/runtime not available on this node."
  exit 1
fi
nvidia-smi || { echo "[ERROR] nvidia-smi failed; GPU driver not usable."; exit 1; }

# Confirm Python can load libcuda.so from the driver
"$ENV_PREFIX/bin/python" - <<'PY'
import ctypes, sys
try:
    ctypes.CDLL("libcuda.so.1")
    print("[DEBUG] libcuda.so.1 load: OK")
except OSError as e:
    print(f"[ERROR] libcuda.so.1 load failed: {e}")
    sys.exit(1)

libs = [
    "libcudart.so",
    "libcublas.so",
    "libcublasLt.so",
    "libcudnn.so",
    "libcufft.so",
    "libcurand.so",
    "libcusolver.so",
    "libcusparse.so",
]
for lib in libs:
    try:
        ctypes.CDLL(lib)
        print(f"[DEBUG] {lib} load: OK")
    except OSError as e:
        print(f"[ERROR] {lib} load failed: {e}")
PY

# Verify TensorFlow sees the GPU in this exact job environment
"$ENV_PREFIX/bin/python" -c "import tensorflow as tf; print('TF version:', tf.__version__); print('built_with_cuda:', tf.test.is_built_with_cuda()); print('GPUs:', tf.config.list_physical_devices('GPU'))"

DATA_DIR_NAME="idrid_1440x960/split_seed_87"
USE_WANDB=0
# Add your executable commands here
"$ENV_PREFIX/bin/python" train.py \
  --epochs 120 \
  --learning_rate 1e-4 \
  --optimizer adam \
  --workers 1 \
  --dataset-name ${DATA_DIR_NAME} \
  --dataset-type IDRiD \
  --output-dir /arc/project/st-ipor-1/lwqz/PMCNet/weights/${DATA_DIR_NAME} \
  --use-wandb ${USE_WANDB}
