#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

export HF_HOME="$base/.cache/huggingface"
export HF_DATASETS_CACHE="$base/.cache/huggingface/datasets"
export HF_HUB_CACHE="$base/.cache/huggingface/hub"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"

src=en
trg=it

if [ -f "$base/data/train.$src" ] && [ -f "$base/data/train.$trg" ] \
    && [ -f "$base/data/dev.$src" ] && [ -f "$base/data/test.$src" ]; then
    echo "skip download (data/ already populated)"
else
    echo "download IWSLT 2017 $src-$trg"
    python3 "$scripts/download_huggingface_data.py" --src "$src" --trg "$trg" --out data
fi

bash "$scripts/tokenize.sh"
bash "$scripts/learn_apply_bpe.sh" 2000 data/bpe2k
bash "$scripts/learn_apply_bpe.sh" 8000 data/bpe8k
