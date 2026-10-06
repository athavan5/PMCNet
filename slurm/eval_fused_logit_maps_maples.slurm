#!/bin/bash
#SBATCH --job-name=eval_fused_logit_maps_all-on-eyepacs
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

# NOTE: Necessary?
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

# Global parameters
DATA_DIR_NAME="maples_1440x960"
TEST_DATASET_NAME="maples_1440x960"

# PMCNet parameters:
# Epoch numbers for PMCNet:
#   100 for IDRiD
#   32 for combined 3 datasets
#   92 for MAPLES
#   52 for EyePacs
PMCNET_EPOCH_NUM=92
PMCNET_LOGIT_DIR="./result/${DATA_DIR_NAME}/epoch_${PMCNET_EPOCH_NUM}/logits/${TEST_DATASET_NAME}_test/all_logit_maps"

# HACDR-Net parameters:
# Epoch numbers for HACDR-Net:
#   135000 for IDRiD
#   106000 for combined 3 datasets
#   142000 for MAPLES
#   57000 for EyePacs
HACDR_EPOCH_NUM=142000
HACDR_LOGIT_DIR="/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/logit_maps/hacdr_1440x960_maples/epoch_${HACDR_EPOCH_NUM}/${TEST_DATASET_NAME}_test"

# DSR-U-Net parameters:
# Epoch numbers for DSR-U-Net:
#   11 for IDRiD
#   24 for combined 3 datasets
#   23 for MAPLES
#   12 for EyePacs
DSR_U_NET_EPOCH_NUM=23
DSR_U_NET_LOGIT_DIR="/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${DATA_DIR_NAME}/result/unet_plusplus/get-prob-maps-${DATA_DIR_NAME}-ep${DSR_U_NET_EPOCH_NUM}-on-${TEST_DATASET_NAME}/logit_maps/all_logit_maps"

# Use this OUT_DIR if doing AUPR weighted average testing
OUT_DIR="./logit_maps_fused_metrics/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-AUPR-logit-weighted-avg-evaluation"

# Use this OUT_DIR if doing equal weight testing
#OUT_DIR="./logit_maps_fused_metrics/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-equal-weights-evaluation"

mkdir -p "${OUT_DIR}"

#represents equal weights for each model (for regular weighted average testing)
#--weights ${W},${W},${W}
W=$(echo "scale=6; 1/3" | bc)

# NOTE: order flags should reflect the channel order in each model's saved _logits.npy files
# NOTE: weights are normalised automatically — pass raw AUPR scores for AUPR-weighted fusion
# NOTE: remove --logit-dir-c and the third weight to run 2-model fusion
"$ENV_PREFIX/bin/python" eval_fused_logit_maps_metrics.py \
   --save-fused-dir "${OUT_DIR}" \
   --log-txt "${OUT_DIR}/results.txt" \
   --logit-dir-a "${PMCNET_LOGIT_DIR}" \
   --logit-dir-b "${HACDR_LOGIT_DIR}" \
   --logit-dir-c "${DSR_U_NET_LOGIT_DIR}" \
   --order-a bg,ex,ma,se,he \
   --order-b bg,ex,ma,se,he \
   --order-c bg,ex,ma,se,he \
   --test-dir "./data/${TEST_DATASET_NAME}/test" \
   --dataset-name "${DATA_DIR_NAME}" \
   --dataset-type IDRiD \
   --class-mode all \
   --weights 0.7262,0.7346,0.7502
