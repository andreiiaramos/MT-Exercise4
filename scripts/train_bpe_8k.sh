#! /bin/bash
# Train the BPE model with an 8k joint vocabulary.

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs
logs=$base/logs

mkdir -p "$models" "$logs"

num_threads=4

model_name=bpe_8k

mkdir -p "$logs/$model_name"

SECONDS=0

OMP_NUM_THREADS=$num_threads python -m joeynmt train "$configs/$model_name.yaml" \
    > "$logs/$model_name/out" 2> "$logs/$model_name/err"

echo "time taken:"
echo "$SECONDS seconds"
