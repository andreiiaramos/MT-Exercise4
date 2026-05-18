#! /bin/bash
# Tokenize raw IWSLT data with sacremoses.
# Reads:  data/{train,dev,test}.{en,it}
# Writes: data/word/{train,dev,test}.{en,it}

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

src=en
trg=it

in_dir=$base/data
out_dir=$base/data/word

mkdir -p "$out_dir"

for split in train dev test; do
    for lang in $src $trg; do
        in_file=$in_dir/$split.$lang
        out_file=$out_dir/$split.$lang
        if [ ! -f "$in_file" ]; then
            echo "Missing input file: $in_file"
            echo "Run scripts/download_huggingface_data.py first."
            exit 1
        fi
        echo "Tokenizing $in_file -> $out_file"
        sacremoses -l "$lang" tokenize -x < "$in_file" > "$out_file"
    done
done

echo "Done. Tokenized files are in $out_dir"
