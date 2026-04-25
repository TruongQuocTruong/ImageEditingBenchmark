#!/bin/bash
#SBATCH --job-name=qwen3_lora
#SBATCH --output=logs/train128-1e-5_%j.out
#SBATCH --error=logs/train128-1e-5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --gres=mps:2

REQUIRED_VRAM=40000

# Load môi trường
source /datastore/raccoon/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl
mkdir -p logs

# --- LOGIC TÌM GPU (KHÔNG ĐỔI) ---
unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ]; then
    echo "WAITING GPU: $CHECK_OUT"
    exit 0 
elif [ $EXIT_CODE -eq 11 ]; then
    echo "ERROR GPU: $CHECK_OUT"
    exit 1 
fi
BEST_GPU=$CHECK_OUT
echo "✅ Job $SLURM_JOB_ID running on GPU: $BEST_GPU"

# Setup MPS
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU

# --- CHẠY TRAINING QWEN3-VL LORA ---
cd LlamaFactory
# Định nghĩa thư mục Cache chung trên datastore
export HF_HOME="../hf_cache"
export HF_DATASETS_CACHE="../hf_cache/datasets"
# Tạo thư mục nếu chưa có
mkdir -p $HF_DATASETS_CACHE

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
llamafactory-cli train ../lora_scripts/lora_all_metrics/lorarank128_lr2e-5.yaml
# Dọn dẹp MPS sau khi xong
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY