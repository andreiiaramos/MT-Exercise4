#! /bin/bash
set -euo pipefail
export PYTHONUTF8=1
scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)
cd "$base"

if ! python3 -c "import joeynmt" >/dev/null 2>&1; then
    if [ -x "$base/venvs/torch3/bin/python3" ]; then
        # shellcheck disable=SC1091
        source "$base/venvs/torch3/bin/activate"
    else
        echo "joeynmt not importable and venv missing. Run ./scripts/make_virtualenv.sh first."
        exit 1
    fi
fi

# shellcheck disable=SC1091
source "$scripts/_device.sh"
echo "Device: $DEVICE_NAME  (configs from $CONFIG_DIR)"

if command -v pmset >/dev/null 2>&1; then
    (sleep 10 && pmset displaysleepnow || true) &
fi

if command -v caffeinate >/dev/null 2>&1; then
    runner=(caffeinate -i bash -c)
else
    runner=(bash -c)
fi

"${runner[@]}" '
    set -euo pipefail
    bash scripts/prepare_data.sh
    bash scripts/train_word_2k.sh
    bash scripts/train_bpe_2k.sh
    bash scripts/train_bpe_8k.sh
'
