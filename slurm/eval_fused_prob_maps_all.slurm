#!/bin/bash
#SBATCH --job-name=eval_fused_prob_maps_all-on-eyepacs
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
set +u  #disable unbound variable check — conda.sh references PS1 which is unset in batch jobs
ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u  #re-enable after activation

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# Global parameters:
DATA_DIR_NAME="idrid-maples-eyepacs-1440x960"
TEST_DATASET_NAME="idrid-maples-eyepacs-1440x960"

# PMCNet parameters:
# Epoch numbers for PMCNet:
#   100 for IDRiD
#   32 for combined 3 datasets
#   92 for MAPLES
#   52 for EyePacs
PMCNET_EPOCH_NUM=32
PMCNET_PROB_DIR="./result/${DATA_DIR_NAME}/epoch_${PMCNET_EPOCH_NUM}/prob_maps/${TEST_DATASET_NAME}_test"

# HACDR-Net parameters:
# Epoch numbers for HACDR-Net:
#   135000 for IDRiD
#   106000 for combined 3 datasets
#   142000 for MAPLES
#   57000 for EyePacs
HACDR_EPOCH_NUM=106000
HACDR_PROB_DIR="/scratch/st-ipor-1/rlmacdou/HACDR-Net/work_dirs/prob_maps/hacdr_combined_3_datasets/epoch_${HACDR_EPOCH_NUM}/${TEST_DATASET_NAME}_test"

# DSR-U-Net parameters:
# Epoch numbers for DSR-U-Net:
#   11 for IDRiD
#   24 for combined 3 datasets
#   23 for MAPLES
#   12 for EyePacs
DSR_U_NET_EPOCH_NUM=24
DSR_U_NET_PROB_DIR="/scratch/st-ipor-1/rlmacdou/DSR-U-Net/data/${DATA_DIR_NAME}/result/unet_plusplus/get-prob-maps-${DATA_DIR_NAME}-ep${DSR_U_NET_EPOCH_NUM}-on-${TEST_DATASET_NAME}/prob_maps"

#Use this OUT_DIR if doing AUPR weighted average testing
OUT_DIR="./prob_maps_fused_metrics/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-aupr-weighted-avg-gulce"

#Use this OUT_DIR if doing regular weighted average testing
#OUT_DIR="./prob_maps_fused_metrics/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}_DSR-U-Net-ep${DSR_U_NET_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}-gulce"
#OUT_DIR="./prob_maps_fused_metrics/PMCNet-ep${PMCNET_EPOCH_NUM}_HACDR-Net-ep${HACDR_EPOCH_NUM}/${DATA_DIR_NAME}/test-${TEST_DATASET_NAME}"

#represents equal weights for each model (for regular weighted average testing)
#--weights ${W},${W},${W}
W=$(echo "scale=6; 1/3" | bc)

# Add your executable commands here
# NOTE: a should be PMCNet and b should be HACDR-Net for simpler resizing if sizes differ
# NOTE: c is optional
# NOTE: weights is optional, default is equal weights
# NOTE: Changed from eval_fused_prob_maps.py to eval_fused_prob_maps_metrics.py
"$ENV_PREFIX/bin/python" eval_fused_prob_maps_metrics.py \
   --save-fused-dir ${OUT_DIR} \
   --log-txt ${OUT_DIR}/results.txt \
   --prob-dir-a ${PMCNET_PROB_DIR} \
   --prob-dir-b ${HACDR_PROB_DIR} \
   --prob-dir-c ${DSR_U_NET_PROB_DIR} \
   --order-a bg,ex,ma,se,he \
   --order-b bg,ex,ma,se,he \
   --order-c bg,ex,ma,se,he \
   --test-dir ./data/${TEST_DATASET_NAME}/test \
   --dataset-name ${DATA_DIR_NAME} \
   --dataset-type IDRiD \
   --class-mode all \
   --weights 0.8001,0.7838,0.7794

#"$ENV_PREFIX/bin/python" eval_fused_prob_maps_metrics.py \
#  --save-fused-dir ${OUT_DIR} \
#  --log-txt ${OUT_DIR}/results.txt \
#  --prob-dir-a ${PMCNET_PROB_DIR} \
#  --prob-dir-b ${HACDR_PROB_DIR} \
#  --order-a bg,ex,ma,se,he \
#  --order-b bg,ex,ma,se,he \
#  --test-dir ./data/${TEST_DATASET_NAME}/test \
#  --dataset-name ${DATA_DIR_NAME} \
#  --dataset-type IDRiD \
#  --class-mode all 
