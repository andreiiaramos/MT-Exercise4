#! /bin/bash
set -euo pipefail
export PYTHONUTF8=1

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

# shellcheck disable=SC1091
source "$scripts/_device.sh"

model_name="bpe_8k"
base_config="$base/$CONFIG_DIR/$model_name.yaml"
data="$base/data/word"
ref_file="$base/data/test.it"
trans_dir="$base/translations/$model_name"
out_csv="$base/beam_search_results.csv"

# Make sure the config damm well exists
if [ ! -f "$base_config" ]; then
    echo "Missing base config: $base_config"
    exit 1
fi

mkdir -p "$trans_dir"


echo "beam_size,bleu,time_s" > "$out_csv"

echo "Starting Beam Search Evaluation for $model_name on $DEVICE_NAME..."
echo "Results will be saved to $out_csv"
echo "------------------------------------------------------"

for k in {1..10}; do
    echo "Translating with beam size $k..."
    
    # 1. Create a temporary config file with the current beam size
    temp_config="$base/$CONFIG_DIR/${model_name}_beam_${k}.yaml"
    sed "s/beam_size: [0-9]*/beam_size: $k/g" "$base_config" > "$temp_config"
    
    hyp_file="$trans_dir/test.${model_name}.beam${k}.it"
    
    # 2. Run translation and time it
    SECONDS=0
    OMP_NUM_THREADS=4 python3 -m joeynmt translate "$temp_config" < "$data/test.en" > "$hyp_file"
    time_taken=$SECONDS
    
    # 3. Calculate detokenized BLEU 
    bleu=$(cat "$hyp_file" | sacremoses -l it detokenize | sacrebleu "$ref_file" -b)
    
    # 4. Save to CSV and display
    echo "$k,$bleu,$time_taken" >> "$out_csv"
    echo "  -> BLEU: $bleu | Time: ${time_taken}s"
    
    # Clean up the temporary config
    rm "$temp_config"
done

echo "------------------------------------------------------"