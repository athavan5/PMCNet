#!/bin/bash
#SBATCH --job-name=PMCNet_get_prob_maps_logits_eyepacs-on-idrid
#SBATCH --account=st-ipor-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gpus=2
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
set +u  #disable unbound variable check — conda.sh references PS1 which is unset in batch jobs
ENV_PREFIX="/arc/project/st-ipor-1/rlmacdou/envs/pmcnet-env"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u  #re-enable after activation

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

DATA_DIR_NAME="maples_1440x960"

# Add your executable commands here
"$ENV_PREFIX/bin/python" get_prob_maps_logits.py \
  --ckpt_epoch 92 \
  --dataset-name ${DATA_DIR_NAME} \
  --dataset-type IDRiD \
  --test-dir ./data/eyepacs_1440x960/test_new
