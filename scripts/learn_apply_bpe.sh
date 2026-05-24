#! /bin/bash
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <vocab_size> <out_dir>"
    exit 1
fi

vocab_size=$1
rel_out_dir=$2

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

tok_dir=$base/data/word
out_dir=$base/$rel_out_dir
codes=$out_dir/bpe.codes
vocab=$out_dir/joint.vocab

mkdir -p "$out_dir"

if [ -f "$codes" ] && [ -f "$vocab" ]; then
    echo "skip BPE $vocab_size ($out_dir already built)"
    exit 0
fi

if [ ! -f "$tok_dir/train.en" ] || [ ! -f "$tok_dir/train.it" ]; then
    echo "Missing tokenized data in $tok_dir (run tokenize.sh)"
    exit 1
fi

echo "learn BPE $vocab_size -> $codes"
cat "$tok_dir/train.en" "$tok_dir/train.it" \
    | subword-nmt learn-bpe -s "$vocab_size" --total-symbols \
    > "$codes"

echo "build joint vocab -> $vocab"
cat "$tok_dir/train.en" "$tok_dir/train.it" \
    | subword-nmt apply-bpe -c "$codes" --separator "@@" \
    | subword-nmt get-vocab \
    | awk '{print $1}' \
    > "$vocab"
