#!/bin/bash
#SBATCH --job-name=threshold_k_hard_vote_idrid
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
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

# Navigate to the job's working directory
REPO_DIR="/scratch/st-ipor-1/rlmacdou/PMCNet"
cd "${REPO_DIR}"

# Conda environment setup
set +u  # disable unbound variable check — conda.sh references PS1 which is unset in batch jobs
ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u  # re-enable after activation

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ── Global parameters ──────────────────────────────────────────────────────────
DATA_DIR_NAME="idrid_1440x960"
TEST_DATASET_NAME="maples_1440x960"

# ── Threshold-k parameter ──────────────────────────────────────────────────────
# With 3 models:
#   K_VALUE=1  equivalent to OR  (union)       — highest sensitivity
#   K_VALUE=2  equivalent to majority voting   — balanced
#   K_VALUE=3  equivalent to AND (intersection) — highest specificity
K_VALUE=2

# ── PMCNet ─────────────────────────────────────────────────────────────────────
# Epoch numbers for PMCNet:
#   100 for IDRiD | 32 for combined 3 datasets | 92 for MAPLES | 52 for EyePacs
PMCNET_EPOCH_NUM=100
PMCNET_PROB_DIR="./result/${DATA_DIR_NAME}/epoch_${PMCNET_EPOCH_NUM}/prob_maps/${TEST_DATASET_NAME}_test_new"

# ── HACDR-Net ──────────────────────────────────────────────────────────────────
# Epoch numbers for HACDR-Net:
#   135000 for IDRiD | 10600 for combined 3 datasets | 142000 for MAPLES | 57000 for EyePacs
HACDR_EPOCH_NUM=135000
HACDR_PROB_DIR="/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_1440x960_idrid/epoch_${HACDR_EPOCH_NUM}/${TEST_DATASET_NAME}_test_new"

# ── DSR-U-Net ──────────────────────────────────────────────────────────────────
# Epoch numbers for DSR-U-Net:
#   11 for IDRiD | 24 for combined 3 datasets | 23 for MAPLES | 12 for EyePacs
DSR_U_NET_EPOCH_NUM=11
DSR_U_NET_PROB_DIR="/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${DATA_DIR_NAME}/result/unet_plusplus/get-prob-maps-${DATA_DIR_NAME}-ep${DSR_U_NET_EPOCH_NUM}-on-${TEST_DATASET_NAME}/prob_maps_new"

# ── Output directories ─────────────────────────────────────────────────────────
K_MASK_DIR="./prob_maps_threshold_k_hard_vote/k${K_VALUE}/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-gulce"
EVAL_OUT_DIR="./prob_maps_threshold_k_hard_vote_metrics/k${K_VALUE}/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-gulce"

echo "[INFO] k value      : ${K_VALUE}"
echo "[INFO] K mask dir   : ${K_MASK_DIR}"
echo "[INFO] Eval out dir : ${EVAL_OUT_DIR}"

# ── Step 1: Threshold-k hard voting ───────────────────────────────────────────
# Reads (H, W, C) prob maps from each model dir and writes binary (H, W) masks
# per class into K_MASK_DIR/<class>/<stem>.npy
echo "[INFO] Running threshold-k hard voting (k=${K_VALUE})..."
"$ENV_PREFIX/bin/python" threshold_k_hard_voting.py --prob-dir-a "${PMCNET_PROB_DIR}" --prob-dir-b "${HACDR_PROB_DIR}" --prob-dir-c "${DSR_U_NET_PROB_DIR}" --order-a bg,ex,ma,se,he --order-b bg,ex,ma,se,he --order-c bg,ex,ma,se,he --classes ex,ma,se,he --threshold 0.5 --k "${K_VALUE}" --save-k-dir "${K_MASK_DIR}"

# ── Step 2: Evaluate threshold-k masks (AUPR, DICE, IoU) ──────────────────────
# Reads threshold-k masks from K_MASK_DIR and evaluates against GT using the
# same Evaluator and PR_AUC utilities as eval_fused_prob_maps_metrics.py
echo "[INFO] Evaluating threshold-k hard vote masks (k=${K_VALUE})..."
"$ENV_PREFIX/bin/python" eval_threshold_k_hard_vote_metrics.py --k-mask-dir "${K_MASK_DIR}" --k "${K_VALUE}" --dataset-name "${DATA_DIR_NAME}" --dataset-type IDRiD --test-dir ./data/${TEST_DATASET_NAME}/test_new --classes ex,ma,se,he --log-txt "${EVAL_OUT_DIR}/results.txt" --save-k-dir "${EVAL_OUT_DIR}"

echo "[INFO] Done."
