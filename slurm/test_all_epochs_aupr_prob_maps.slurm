#!/bin/bash
#SBATCH --job-name=test_all_epochs_aupr_prob_maps
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/rlmacdou/PMCNet/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/rlmacdou/PMCNet/hpc_logs/%x-%j.err
#SBATCH --mail-user=rlmacdou@student.ubc.ca
#SBATCH --mail-type=FAIL,END
#SBATCH --reservation=st-ipor-1-gpu_34

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

REPO_DIR="/scratch/st-ipor-1/rlmacdou/PMCNet"
cd "${REPO_DIR}"

ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# ---------------------------------------------------------------------------
# Configure these via env vars before sbatch if desired.
# ---------------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-idrid-maples-eyepacs-1440x960}"
DATASET_TYPE="${DATASET_TYPE:-IDRiD}"
SPLIT="${SPLIT:-val}"   # test | val | validation

# Root that contains one subfolder per epoch (e.g., epoch_1, epoch_2, ...)
PROB_ROOT="${PROB_ROOT:-/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${DATASET_NAME}/result/unet_plusplus/val_prob_maps}"
PROB_SUBDIR="${PROB_SUBDIR:-prob_maps}" # default is prob_maps, because DSR-U-Net often writes to nested prob_maps/ (e.g. epoch_1/prob_maps/)

# Optional explicit data root containing images/ and masks/ for chosen split.
# If unset, defaults to ./data/${DATASET_NAME}/${SPLIT}/
DATA_DIR="${DATA_DIR:-./data/${DATASET_NAME}/val}"

ORDER="${ORDER:-bg,ex,ma,se,he}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROB_ROOT}/all_epochs_aupr_${SPLIT}.csv}"

# Set to 1 to append Dice/IoU (metrics.py eval_metrics) columns to the same CSV.
INCLUDE_SEG_METRICS="${INCLUDE_SEG_METRICS:-1}"
IGNORE_INDEX="${IGNORE_INDEX:-255}"

CMD=(
  "$ENV_PREFIX/bin/python" eval_all_epochs_prob_maps_aupr.py
  --dataset-name "${DATASET_NAME}"
  --dataset-type "${DATASET_TYPE}"
  --split "${SPLIT}"
  --prob-root "${PROB_ROOT}"
  --order "${ORDER}"
  --output-csv "${OUTPUT_CSV}"
  --prob-subdir "${PROB_SUBDIR}"
)

if [[ -n "${DATA_DIR}" ]]; then
  CMD+=(--data-dir "${DATA_DIR}")
fi

if [[ "${INCLUDE_SEG_METRICS}" == "1" ]]; then
  CMD+=(--include-seg-metrics --ignore-index "${IGNORE_INDEX}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Done. CSV: ${OUTPUT_CSV}"
