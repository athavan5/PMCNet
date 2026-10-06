#!/bin/bash
#SBATCH --job-name=test_prob_maps_single_epoch
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --output=/scratch/st-ipor-1/bc2020ag/PMCNet/hpc_logs/%x-%j.out
#SBATCH --error=/scratch/st-ipor-1/bc2020ag/PMCNet/hpc_logs/%x-%j.err
#SBATCH --mail-user=bc2020ag@student.ubc.ca
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

REPO_DIR="/scratch/st-ipor-1/rlmacdou/PMCNet"
cd "${REPO_DIR}"

ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"

# ---------------------------------------------------------------------------
# Configure these variables before submitting.
#
# DATASET_NAME        — dataset DSR-U-Net was trained on
# DSR_U_NET_EPOCH_NUM — the specific epoch checkpoint you want to evaluate
# TEST_DIR            — dataset used for testing (controls both DATA_DIR and
#                       the folder name inside unet_plusplus/)
# ---------------------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-idrid-maples-eyepacs-1440x960}"
DATASET_TYPE="${DATASET_TYPE:-IDRiD}"
DSR_U_NET_EPOCH_NUM="${DSR_U_NET_EPOCH_NUM:-24}"
TEST_DIR="${TEST_DIR:-idrid-maples-eyepacs-1440x960}"

# Ground truth loaded from the test split of the chosen TEST_DIR dataset (originally ./data/${TEST_DIR}/test)
DATA_DIR="${DATA_DIR:-./data/${TEST_DIR}/test}"
#DATA_DIR="${DATA_DIR:-/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${TEST_DIR}/result/all}"

# Single prob-map folder for the specified epoch and test dataset
PROB_ROOT="${PROB_ROOT:-/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${DATASET_NAME}/result/unet_plusplus/get-prob-maps-${DATASET_NAME}-ep${DSR_U_NET_EPOCH_NUM}-on-${TEST_DIR}}"

# .npy files live under prob_maps/ inside PROB_ROOT (originally -prob_maps)
PROB_SUBDIR="${PROB_SUBDIR:-prob_maps}"

ORDER="${ORDER:-bg,ex,ma,se,he}"

# Output CSV saved directly into the single epoch folder
OUTPUT_CSV="${OUTPUT_CSV:-${PROB_ROOT}/metrics_test_gulce.csv}"

INCLUDE_SEG_METRICS="${INCLUDE_SEG_METRICS:-1}"
IGNORE_INDEX="${IGNORE_INDEX:-255}"

echo "[INFO] DATASET_NAME=${DATASET_NAME}"
echo "[INFO] DSR_U_NET_EPOCH_NUM=${DSR_U_NET_EPOCH_NUM}"
echo "[INFO] TEST_DIR=${TEST_DIR}"
echo "[INFO] DATA_DIR=${DATA_DIR}"
echo "[INFO] PROB_ROOT=${PROB_ROOT}"
echo "[INFO] OUTPUT_CSV=${OUTPUT_CSV}"

CMD=(
  "$ENV_PREFIX/bin/python" eval_single_epoch_prob_maps.py
  --dataset-name "${DATASET_NAME}"
  --dataset-type "${DATASET_TYPE}"
  --prob-root    "${PROB_ROOT}"
  --prob-subdir  "${PROB_SUBDIR}"
  --order        "${ORDER}"
  --output-csv   "${OUTPUT_CSV}"
  --data-dir     "${DATA_DIR}"
)

if [[ "${INCLUDE_SEG_METRICS}" == "1" ]]; then
  CMD+=(--include-seg-metrics --ignore-index "${IGNORE_INDEX}")
fi

echo "[INFO] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[INFO] Done. CSV: ${OUTPUT_CSV}"
