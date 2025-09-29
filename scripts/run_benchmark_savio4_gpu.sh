#!/bin/bash
#SBATCH --job-name=mmethyl-bench
#SBATCH --partition=savio4_gpu
#SBATCH --qos=savio_normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] CUDA devices: $CUDA_VISIBLE_DEVICES"

module load python/3.11.6-gcc-11.4.0 cuda/11.7 || true

# Activate venv if present in job dir
if [ -d .venv ]; then
  source .venv/bin/activate
fi

python3 -V
pip -V || true

# Ensure results and logs dirs
mkdir -p results logs

# DATA ROOT should point to MuLan-Methyl processed_dataset (train/test TSVs)
# Default to repo-internal path if DATA_ROOT not provided
DATA_ROOT=${DATA_ROOT:-"${SLURM_SUBMIT_DIR:-$PWD}/data/benchmark/processed_dataset"}

echo "[INFO] Using DATA_ROOT=$DATA_ROOT"

python3 scripts/finetune_from_tsv.py \
  --data_root "$DATA_ROOT" \
  --models DNABERT-6 Plant-DNABERT-BPE HyenaDNA GENA-LM \
  --marks 4mC 5hmC 6mA \
  --num_train_epochs 1 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --max_length 512 \
  --output_dir results \
  --logging_dir logs \
  --output_csv results/benchmark_results.csv \
  --seed 42 \
  --max_train_samples 20000 \
  --max_eval_samples 10000

echo "[INFO] Done. Results in results/benchmark_results.csv"