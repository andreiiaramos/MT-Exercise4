#! /bin/bash

set -euo pipefail

scripts=$(dirname "$0")
base=$scripts/..

required_commands=(python3 bash sacremoses subword-nmt sacrebleu)
shell_scripts=(
	"$scripts/download_moses.sh"
	"$scripts/evaluate.sh"
	"$scripts/learn_apply_bpe.sh"
	"$scripts/make_virtualenv.sh"
	"$scripts/prepare_data.sh"
	"$scripts/preflight.sh"
	"$scripts/tokenize.sh"
	"$scripts/train.sh"
	"$scripts/train_bpe_2k.sh"
	"$scripts/train_bpe_8k.sh"
	"$scripts/train_word_2k.sh"
)

config_files=(
	"$base/configs/bpe_2k.yaml"
	"$base/configs/bpe_8k.yaml"
	"$base/configs/transformer_sample_config.yaml"
	"$base/configs/word_2k.yaml"
)

status=0

echo "Checking required commands..."
for command in "${required_commands[@]}"; do
	if ! command -v "$command" >/dev/null 2>&1; then
		echo "Missing command: $command"
		status=1
	else
		echo "Found: $command"
	fi
done

echo ""
echo "Checking shell scripts..."
for script in "${shell_scripts[@]}"; do
	if [ ! -f "$script" ]; then
		echo "Missing script: $script"
		status=1
		continue
	fi
	if bash -n "$script"; then
		echo "OK: $script"
	else
		echo "Syntax error in: $script"
		status=1
	fi
done

echo ""
echo "Checking config files..."
for config in "${config_files[@]}"; do
	if [ -f "$config" ]; then
		echo "OK: $config"
	else
		echo "Missing config: $config"
		status=1
	fi
done

echo ""
echo "Checking device assumptions..."
for config in "$base/configs/word_2k.yaml" "$base/configs/bpe_2k.yaml" "$base/configs/bpe_8k.yaml"; do
	if grep -q "use_cuda: False" "$config"; then
		echo "Warning: $config still disables CUDA. Update this for a GPU-backed run if your backend supports it."
	else
		echo "OK: $config"
	fi
done

echo ""
if python3 - <<'PY'
import importlib.util
modules = ["torch", "joeynmt"]
missing = [m for m in modules if importlib.util.find_spec(m) is None]
raise SystemExit(1 if missing else 0)
PY
then
	echo "Python modules found: torch, joeynmt"
else
	echo "Warning: torch and/or joeynmt are not installed in the active Python environment."
	status=1
fi

if [ "$status" -eq 0 ]; then
	echo ""
	echo "Preflight passed."
else
	echo ""
	echo "Preflight completed with issues."
fi

exit "$status"