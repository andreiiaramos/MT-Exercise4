#! /bin/bash

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

models=$base/models
configs=$base/configs

mkdir -p $models

num_threads=4

# measure time

SECONDS=0

logs=$base/logs

model_name=${1:-word_2k}

config_file=$configs/$model_name.yaml

if [ ! -f "$config_file" ]; then
	echo "Missing config file: $config_file"
	exit 1
fi

mkdir -p $logs

mkdir -p $logs/$model_name

OMP_NUM_THREADS=$num_threads python -m joeynmt train "$config_file" > $logs/$model_name/out 2> $logs/$model_name/err

echo "time taken:"
echo "$SECONDS seconds"
