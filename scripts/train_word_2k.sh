#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

# shellcheck disable=SC1091
source "$scripts/_device.sh"

model_name=word_2k
config_file=$base/$CONFIG_DIR/$model_name.yaml
model_dir=$base/models/$model_name
logs_dir=$base/logs/$model_name

if compgen -G "$model_dir/*.ckpt" >/dev/null; then
    echo "skip $model_name (checkpoint already in $model_dir)"
    exit 0
fi

mkdir -p "$base/models" "$logs_dir"

echo "train $model_name on $DEVICE_NAME using $config_file"

SECONDS=0
OMP_NUM_THREADS=4 python3 -m joeynmt train "$config_file" \
    > "$logs_dir/out" 2> "$logs_dir/err"
echo "done in $SECONDS s"
