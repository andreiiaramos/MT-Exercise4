#! /bin/bash
# End-to-end data preparation:
#   1. Download IWSLT 2017 en-it (capped at 100k train pairs by the downloader).
#   2. Tokenize all splits with sacremoses.
#   3. Learn + apply BPE at two vocabulary sizes (2000 and 8000).
#
# Run from the repository root after activating the virtualenv:
#     ./scripts/prepare_data.sh

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

src=en
trg=it

# 1. Download (skip if files already exist)
if [ ! -f "$base/data/train.$src" ] || [ ! -f "$base/data/train.$trg" ]; then
    echo "Downloading IWSLT 2017 $src-$trg ..."
    python "$scripts/download_huggingface_data.py" --src "$src" --trg "$trg" --out data
else
    echo "Raw data already present in $base/data — skipping download."
fi

# 2. Tokenize
bash "$scripts/tokenize.sh"

# 3. BPE at two sizes
bash "$scripts/learn_apply_bpe.sh" 2000 data/bpe2k
bash "$scripts/learn_apply_bpe.sh" 8000 data/bpe8k

echo ""
echo "All preparation steps complete."
echo "You can now run training:"
echo "    ./scripts/train_word_2k.sh"
echo "    ./scripts/train_bpe_2k.sh"
echo "    ./scripts/train_bpe_8k.sh"
