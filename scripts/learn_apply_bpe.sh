#! /bin/bash
# Learn a joint BPE model on tokenized training data, apply it to all splits,
# and build the joint BPE vocabulary file that JoeyNMT expects.
#
# Usage:
#     ./scripts/learn_apply_bpe.sh <vocab_size> <out_dir>
# Example:
#     ./scripts/learn_apply_bpe.sh 2000 data/bpe2k
#     ./scripts/learn_apply_bpe.sh 8000 data/bpe8k

set -euo pipefail

if [ $# -ne 2 ]; then
    echo "Usage: $0 <vocab_size> <out_dir>"
    exit 1
fi

vocab_size=$1
out_dir=$2

scripts=$(dirname "$0")
base=$scripts/..

src=en
trg=it

tok_dir=$base/data/word
out_dir=$base/$out_dir

mkdir -p "$out_dir"

codes_file=$out_dir/bpe.codes
joint_vocab=$out_dir/joint.vocab

# 1. Learn joint BPE on the concatenated tokenized training data
echo "Learning BPE (size=$vocab_size) ..."
cat "$tok_dir/train.$src" "$tok_dir/train.$trg" \
    | subword-nmt learn-bpe -s "$vocab_size" --total-symbols \
    > "$codes_file"

# 2. Apply BPE to every split / language
echo "Applying BPE ..."
for split in train dev test; do
    for lang in $src $trg; do
        in_file=$tok_dir/$split.$lang
        out_file=$out_dir/$split.bpe.$lang
        subword-nmt apply-bpe -c "$codes_file" --separator "@@" \
            < "$in_file" > "$out_file"
    done
done

# 3. Build the joint vocabulary (token per line, no counts; JoeyNMT only needs tokens)
echo "Building joint vocabulary ..."
cat "$out_dir/train.bpe.$src" "$out_dir/train.bpe.$trg" \
    | subword-nmt get-vocab \
    | awk '{print $1}' \
    > "$joint_vocab"

# 4. Sanity check: there should be no <unk> in BPE-encoded training data
unk_count=$(grep -c "<unk>" "$out_dir/train.bpe.$src" "$out_dir/train.bpe.$trg" || true)
echo "BPE training files contain $unk_count occurrences of <unk> (expected: 0)"

echo "Done. Outputs are in $out_dir"
echo "  codes:        $codes_file"
echo "  joint vocab:  $joint_vocab"
echo "  split files:  $out_dir/{train,dev,test}.bpe.{$src,$trg}"
