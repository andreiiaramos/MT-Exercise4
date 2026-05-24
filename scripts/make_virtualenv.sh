#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(cd "$scripts/.." && pwd)

venv_dir=$base/venvs/torch3
tools_dir=$base/tools
hotfix_dir=$tools_dir/joeynmt-hotfixed

mkdir -p "$base/venvs" "$tools_dir"

if [ ! -d "$venv_dir" ]; then
    python3 -m virtualenv -p python3.10 "$venv_dir"
fi

pip="$venv_dir/bin/python -m pip"
$pip install --upgrade pip
$pip install "numpy<2" sacremoses nltk "datasets==3.6.0" subword-nmt sacrebleu

if [ ! -d "$hotfix_dir/.git" ]; then
    git clone https://github.com/moritz-steiner/joeynmt-hotfixed "$hotfix_dir"
fi
$pip install -e "$hotfix_dir"

echo "activate with: source $venv_dir/bin/activate"
