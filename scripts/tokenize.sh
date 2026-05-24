#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

in_dir=$base/data
out_dir=$base/data/word
mkdir -p "$out_dir"

for split in train dev test; do
    for lang in en it; do
        in_file=$in_dir/$split.$lang
        out_file=$out_dir/$split.$lang
        if [ ! -f "$in_file" ]; then
            echo "Missing: $in_file (run prepare_data.sh)"
            exit 1
        fi
        if [ -f "$out_file" ] && [ "$out_file" -nt "$in_file" ]; then
            echo "skip $out_file"
            continue
        fi
        echo "tokenize $in_file"
        sacremoses -l "$lang" tokenize -x < "$in_file" > "$out_file"
    done
done
