#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

# shellcheck disable=SC1091
source "$scripts/_device.sh"

model_name=${1:-word_2k}
case "$model_name" in
    word_2k|bpe_2k|bpe_8k) ;;
    *) echo "unknown model: $model_name"; exit 1 ;;
esac

config_file=$base/$CONFIG_DIR/$model_name.yaml
data=$base/data/word
trans_dir=$base/translations/$model_name
hyp_file=$trans_dir/test.$model_name.it

if [ ! -f "$config_file" ]; then
    echo "missing config: $config_file"
    exit 1
fi
if [ ! -f "$data/test.en" ] || [ ! -f "$data/test.it" ]; then
    echo "missing test data in $data (run prepare_data.sh)"
    exit 1
fi
if ! compgen -G "$base/models/$model_name/*.ckpt" >/dev/null; then
    echo "no checkpoint in $base/models/$model_name (train first)"
    exit 1
fi

mkdir -p "$trans_dir"

if [ -f "$hyp_file" ]; then
    echo "skip translate ($hyp_file exists)"
else
    SECONDS=0
    OMP_NUM_THREADS=4 python3 -m joeynmt translate "$config_file" \
        < "$data/test.en" > "$hyp_file"
    echo "translated in $SECONDS s"
fi

echo "BLEU ($model_name on $DEVICE_NAME):"
cat "$hyp_file" | sacrebleu "$data/test.it"
