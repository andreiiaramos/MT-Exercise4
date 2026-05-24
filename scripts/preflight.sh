#! /bin/bash
set -uo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

status=0

required_commands=(python3 bash git sacremoses subword-nmt sacrebleu)
shell_scripts=(
    "$scripts/download_moses.sh"
    "$scripts/evaluate.sh"
    "$scripts/learn_apply_bpe.sh"
    "$scripts/make_virtualenv.sh"
    "$scripts/prepare_data.sh"
    "$scripts/preflight.sh"
    "$scripts/run_all.sh"
    "$scripts/tokenize.sh"
    "$scripts/train.sh"
    "$scripts/train_bpe_2k.sh"
    "$scripts/train_bpe_8k.sh"
    "$scripts/train_word_2k.sh"
    "$scripts/_device.sh"
)
config_files=(
    "$base/configs/word_2k.yaml"
    "$base/configs/bpe_2k.yaml"
    "$base/configs/bpe_8k.yaml"
    "$base/configs/cuda/word_2k.yaml"
    "$base/configs/cuda/bpe_2k.yaml"
    "$base/configs/cuda/bpe_8k.yaml"
)

echo "== Commands =="
for c in "${required_commands[@]}"; do
    if command -v "$c" >/dev/null 2>&1; then
        echo "  ok    $c"
    else
        echo "  MISS  $c"
        status=1
    fi
done

echo "== Shell scripts =="
for s in "${shell_scripts[@]}"; do
    if [ ! -f "$s" ]; then
        echo "  MISS  $s"
        status=1
    elif bash -n "$s" 2>/dev/null; then
        echo "  ok    $s"
    else
        echo "  FAIL  $s"
        status=1
    fi
done

echo "== Configs =="
for c in "${config_files[@]}"; do
    if [ -f "$c" ]; then
        echo "  ok    $c"
    else
        echo "  MISS  $c"
        status=1
    fi
done

echo "== Python modules =="
for m in torch joeynmt; do
    if python3 -c "import $m" >/dev/null 2>&1; then
        echo "  ok    $m"
    else
        echo "  MISS  $m"
        status=1
    fi
done

echo "== Device =="
gpu_name=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
fi
if [ -n "$gpu_name" ]; then
    echo "  GPU detected: $gpu_name"
fi

# shellcheck disable=SC1091
source "$scripts/_device.sh"
echo "  DEVICE_NAME = $DEVICE_NAME"
echo "  CONFIG_DIR  = $CONFIG_DIR"

if [ "$DEVICE_NAME" = "cpu" ] && [ -n "$gpu_name" ]; then
    echo "  WARN: GPU is present but torch.cuda.is_available() == False."
    echo "        Install a CUDA-enabled torch: pip install torch --index-url https://download.pytorch.org/whl/cu121"
fi

echo "== Data artifacts =="
declare -A artifacts=(
    [raw_en]="$base/data/train.en"
    [raw_it]="$base/data/train.it"
    [tok_en]="$base/data/word/train.en"
    [tok_it]="$base/data/word/train.it"
    [bpe2k_codes]="$base/data/bpe2k/bpe.codes"
    [bpe2k_vocab]="$base/data/bpe2k/joint.vocab"
    [bpe8k_codes]="$base/data/bpe8k/bpe.codes"
    [bpe8k_vocab]="$base/data/bpe8k/joint.vocab"
)
for key in raw_en raw_it tok_en tok_it bpe2k_codes bpe2k_vocab bpe8k_codes bpe8k_vocab; do
    if [ -f "${artifacts[$key]}" ]; then
        echo "  ok    $key  ${artifacts[$key]}"
    else
        echo "  miss  $key  ${artifacts[$key]}"
    fi
done

echo "== Models =="
for m in word_2k bpe_2k bpe_8k; do
    if compgen -G "$base/models/$m/*.ckpt" >/dev/null; then
        echo "  trained   $m"
    else
        echo "  pending   $m"
    fi
done

echo ""
if [ "$status" -eq 0 ]; then
    echo "Preflight passed."
else
    echo "Preflight reported issues."
fi
exit "$status"
