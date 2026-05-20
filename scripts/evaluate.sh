#! /bin/bash

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

data_root=$base/data
configs=$base/configs

translations=$base/translations

mkdir -p $translations

src=en
trg=it


num_threads=4
device=0

# measure time

SECONDS=0

model_name=${1:-word_2k}

case "$model_name" in
	word_2k)
		data=$data_root/word
		;;
	bpe_2k)
		data=$data_root/bpe2k
		;;
	bpe_8k)
		data=$data_root/bpe8k
		;;
	*)
		echo "Unknown model name: $model_name"
		exit 1
		;;
esac

config_file=$configs/$model_name.yaml

if [ ! -f "$config_file" ]; then
	echo "Missing config file: $config_file"
	exit 1
fi

if [ ! -f "$data/test.$src" ] || [ ! -f "$data/test.$trg" ]; then
	echo "Missing test data in $data"
	echo "Run scripts/prepare_data.sh first."
	exit 1
fi

echo "###############################################################################"
echo "model_name $model_name"

translations_sub=$translations/$model_name

mkdir -p "$translations_sub"

CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python -m joeynmt translate "$config_file" < "$data/test.$src" > "$translations_sub/test.$model_name.$trg"

# compute case-sensitive BLEU 

cat "$translations_sub/test.$model_name.$trg" | sacrebleu "$data/test.$trg"


echo "time taken:"
echo "$SECONDS seconds"
