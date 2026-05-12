#!/usr/bin/env bash
# Train and evaluate DiffDiff on all 7 datasets in parallel, one per GPU.
# Each dataset runs 4 horizons concurrently as subprocesses on its GPU.
#
# Usage:
#   bash scripts/reproduce.sh
#   PYTHON=/path/to/python bash scripts/reproduce.sh
#
# Adjust the GPU mapping below if your machine has a different layout.

set -e

mkdir -p logs

declare -A GPU_OF
GPU_OF[electricity]=0
GPU_OF[ettm2]=1
GPU_OF[exchange_rate]=2
GPU_OF[traffic]=3
GPU_OF[weather]=4
GPU_OF[solar]=5
GPU_OF[wind]=6

for ds in electricity ettm2 exchange_rate traffic weather solar wind; do
    gpu=${GPU_OF[$ds]}
    echo "[launch] dataset=$ds gpu=$gpu"
    nohup "${PYTHON:-python}" scripts/run_dataset.py \
        --dataset "$ds" \
        --gpu "$gpu" \
        --parallel_horizons \
        --seeds 0,1,2,3,4 \
        > "logs/master_${ds}.log" 2>&1 &
done

echo "[reproduce] all 7 datasets launched. Monitor with: tail -f logs/master_*.log"
echo "[reproduce] when done, run: python scripts/compare_all.py"

wait
echo "[reproduce] all datasets finished."
