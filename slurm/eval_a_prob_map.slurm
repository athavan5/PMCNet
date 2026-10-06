#!/bin/bash
#SBATCH --job-name=eval_a_prob_map
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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
# Set these (or export before sbatch). This job evaluates each prob dir in the
# list — no fusion; each folder must contain {basename}_probs.npy per test image.
# ---------------------------------------------------------------------------
DATA_DIR_NAME="${DATA_DIR_NAME:-eyepacs_1440x960}" # NOTE: Not necessary if test_dir is provided
TEST_DATASET_NAME="${TEST_DATASET_NAME:-idrid-maples-eyepacs-1440x960}"
DATASET_TYPE="${DATASET_TYPE:-IDRiD}"
TEST_DIR="${TEST_DIR:-${REPO_DIR}/data/${TEST_DATASET_NAME}/test}"
CLASS_MODE="${CLASS_MODE:-all}"
ORDER="${ORDER:-bg,ex,ma,se,he}"

# One or more probability-map directories (edit this list).
# Each directory must hold {basename}_probs.npy for every test image basename.
PROB_DIRS=(
  "/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/idrid-maples-eyepacs-1440x960/result/unet_plusplus/get-prob-maps-idrid-maples-eyepacs-1440x960-ep20-on-${TEST_DATASET_NAME}/prob_maps"
  "/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/eyepacs_1440x960/result/unet_plusplus/get-prob-maps-eyepacs_1440x960-ep18-on-${TEST_DATASET_NAME}/prob_maps"
  "/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/idrid_1440x960/result/unet_plusplus/get-prob-maps-idrid_1440x960-ep22-on-${TEST_DATASET_NAME}/prob_maps"
  "/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/maples_1440x960/result/unet_plusplus/get-prob-maps-maples_1440x960-ep24-on-${TEST_DATASET_NAME}/prob_maps"
  "/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_combined_3_datasets/epoch_106000/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_1440x960_eyepacs/epoch_57000/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_1440x960_idrid/epoch_135000/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_1440x960_maples/epoch_142000/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/PMCNet/result/idrid_1440x960/epoch_100/prob_maps/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/PMCNet/result/maples_1440x960/epoch_92/prob_maps/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/PMCNet/result/eyepacs_1440x960/epoch_52/prob_maps/${TEST_DATASET_NAME}_test"
  "/scratch/st-ipor-1/rlmacdou/PMCNet/result/idrid-maples-eyepacs-1440x960/epoch_32/prob_maps/${TEST_DATASET_NAME}_test"
  # Add more lines here, e.g.:
  # "/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/.../epoch_X/${TEST_DATASET_NAME}_test"
  # "/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/.../prob_maps"
)

LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/eval_prob_only/test-${TEST_DATASET_NAME}}"
mkdir -p "${LOG_ROOT}"

i=0
for PROB_DIR in "${PROB_DIRS[@]}"; do
  if [[ -z "${PROB_DIR}" ]]; then
    continue
  fi
  if [[ ! -d "${PROB_DIR}" ]]; then
    echo "[ERROR] Not a directory: ${PROB_DIR}" >&2
    exit 1
  fi
  # PART_NAME is everything after and including the 5th "/" element of PROB_DIR
  PART_NAME="$(echo "${PROB_DIR}" | awk -F'/' '{for(i=5;i<=NF;++i)printf (i==5?"%s":"\/%s"), $i; print ""}')"
  OUT_DIR="${LOG_ROOT}/${PART_NAME}"
  mkdir -p "${OUT_DIR}"

  echo "[INFO] --- (${i}) prob-dir=${PROB_DIR}"
  echo "[INFO]     log: ${OUT_DIR}/results.txt"

  "$ENV_PREFIX/bin/python" eval_a_prob_map.py \
    --prob-dir "${PROB_DIR}" \
    --order "${ORDER}" \
    --dataset-name "${DATA_DIR_NAME}" \
    --dataset-type "${DATASET_TYPE}" \
    --test-dir "${TEST_DIR}" \
    --class-mode "${CLASS_MODE}" \
    --log-txt "${OUT_DIR}/results.txt"

  i=$((i + 1))
done

echo "[INFO] Done. Logs under: ${LOG_ROOT}"
