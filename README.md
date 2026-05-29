# MT Exercise 4 — Byte Pair Encoding & Beam Search

Train three en→it transformer NMT systems on IWSLT 2017 (word-level with a 2k vocab threshold; BPE with 2k joint vocab; BPE with 8k joint vocab) and study the impact of beam size on translation quality.

Everything the pipeline writes lives inside this repository: `venvs/`, `tools/`, `.cache/`, `data/`, `models/`, `logs/`, `translations/`. Nothing is written outside the repo.

## Requirements

- Python 3.10 (must be reachable as `python3` — on macOS `python` alone is usually absent)
- `virtualenv` (`pip install virtualenv`)
- `git`
- On Windows: run the shell scripts from **Git Bash** or WSL
- On macOS with Apple Silicon: training runs on CPU (no CUDA on M1/M2)
- On NVIDIA GPUs: install torch with CUDA, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu121` (do this after `make_virtualenv.sh` if `torch.cuda.is_available()` returns `False`)

## One-time setup

```
./scripts/make_virtualenv.sh
source ./venvs/torch3/bin/activate
```

`make_virtualenv.sh` creates the venv at `venvs/torch3/`, installs `numpy<2 sacremoses nltk datasets==3.6.0 subword-nmt sacrebleu`, clones `joeynmt-hotfixed` into `tools/joeynmt-hotfixed/`, and installs it editable. Re-running is safe — every step is idempotent.

## Device detection

`scripts/_device.sh` is sourced by every script that runs JoeyNMT. It checks `torch.cuda.is_available()` and exports:

- `DEVICE_NAME=cuda`, `CONFIG_DIR=configs/cuda` → CUDA path
- `DEVICE_NAME=cpu`,  `CONFIG_DIR=configs`      → CPU path

Both config sets are byte-for-byte identical except for `use_cuda`. Train/eval scripts auto-pick the right set; you don't pass a flag.

`./scripts/preflight.sh` prints which path is active, whether an NVIDIA GPU is visible, and whether all required commands, configs, and Python modules are present. It also lists data artifacts (present/missing) and model checkpoints (trained/pending).

## End-to-end run

```
./scripts/preflight.sh        # optional sanity check
./scripts/run_all.sh
```

`run_all.sh`:

1. activates `venvs/torch3` if a venv isn't already active,
2. sources `_device.sh` and reports the detected device,
3. on macOS, wraps the pipeline in `caffeinate -i` (system stays awake) and fires `pmset displaysleepnow` after 10 s (display goes dark; on other OSes both calls are skipped),
4. runs `prepare_data.sh` → `train_word_2k.sh` → `train_bpe_2k.sh` → `train_bpe_8k.sh`.

Every step skips itself if its output already exists, so re-running after interruption resumes where it stopped.

## Pipeline stages and their caches

| Script                              | Output(s)                                            | Skip condition                                                       |
| ---                                 | ---                                                  | ---                                                                  |
| `scripts/download_huggingface_data.py` | `data/{train,dev,test}.{en,it}`                  | `data/train.en` + `data/train.it` + `data/dev.en` + `data/test.en` already exist |
| `scripts/tokenize.sh`               | `data/word/{train,dev,test}.{en,it}`                 | Per-file: output exists and is newer than input                      |
| `scripts/learn_apply_bpe.sh N D`    | `D/bpe.codes`, `D/joint.vocab`                       | Both files already exist in `D`                                      |
| `scripts/train_*.sh`                | `models/<name>/*.ckpt`, `logs/<name>/{out,err}`     | Any `*.ckpt` already in `models/<name>/`                             |
| `scripts/evaluate.sh <name>`        | `translations/<name>/test.<name>.it`                 | Hypothesis file already exists (sacrebleu still runs over it)        |

The HuggingFace cache is forced into `./.cache/huggingface/` via `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, so nothing lands in `~/.cache/`.

## Individual steps

```
./scripts/prepare_data.sh        # download + tokenize + learn BPE @ 2k & 8k
./scripts/train_word_2k.sh
./scripts/train_bpe_2k.sh
./scripts/train_bpe_8k.sh
./scripts/evaluate.sh word_2k    # translate test set + sacrebleu
./scripts/evaluate.sh bpe_2k
./scripts/evaluate.sh bpe_8k
```

All three configs point `train`/`dev`/`test` at `data/word/` because JoeyNMT applies BPE internally via `tokenizer_cfg.codes` (per the exercise sheet). `data/bpe2k/` and `data/bpe8k/` only hold the BPE codes and joint vocab files.

## Configs

| File                          | Vocabulary                          | `use_cuda` |
| ---                           | ---                                 | ---        |
| `configs/word_2k.yaml`        | `voc_limit: 2000`, untied embeddings | False      |
| `configs/bpe_2k.yaml`         | joint `data/bpe2k/joint.vocab`       | False      |
| `configs/bpe_8k.yaml`         | joint `data/bpe8k/joint.vocab`       | False      |
| `configs/cuda/word_2k.yaml`   | same as above                        | True       |
| `configs/cuda/bpe_2k.yaml`    | same as above                        | True       |
| `configs/cuda/bpe_8k.yaml`    | same as above                        | True       |

Other settings are identical to `configs/transformer_sample_config.yaml` (transformer, 4 encoder / 1 decoder layers, 256-dim, batch 2048 tokens, label smoothing 0.3, plateau LR schedule, 10 epochs). The word-level config sets `tied_embeddings: False` and `tied_softmax: False`; the BPE configs keep them `True` because vocab is joint.

## Re-running and overrides

- Re-running any script is safe; cached outputs are skipped.
- To force a step to re-run, delete its output (`rm -rf models/word_2k`, `rm data/bpe2k/bpe.codes`, etc.).
- To switch device after a CPU run, delete the relevant `models/<name>/` then re-run; `_device.sh` will pick `configs/cuda/` automatically once `torch.cuda.is_available()` flips.

## Exercise 2 — beam search

Beam search experiments are implemented using the best-performing model (`bpe_8k`).

The script

```bash
./scripts/evaluate_beam.sh
```

evaluates beam sizes 1–10, records BLEU scores and translation times in `beam_search_results.csv`, and stores the generated translations in:

```text
translations/bpe_8k/
```

The script

```bash
python scripts/plot_graphs.py
```

creates the following graphs:

```text
graphs/beam_vs_bleu.png
graphs/beam_vs_time.png
```

All numerical results and analysis are documented in `README_Results.md`.
## Repository layout

```
configs/                  # CPU configs (use_cuda: False)
configs/cuda/             # CUDA configs (use_cuda: True)
configs/transformer_sample_config.yaml   # original template (unmodified)
scripts/                  # bash + python entry points
scripts/_device.sh        # sourced helper (sets DEVICE_NAME, CONFIG_DIR)
venvs/torch3/             # local virtualenv (created by make_virtualenv.sh)
tools/joeynmt-hotfixed/   # cloned dependency (editable install)
.cache/huggingface/       # HF datasets cache
data/                     # raw + tokenized + BPE artifacts
models/                   # trained checkpoints
logs/                     # stdout/stderr per model
translations/             # test-set translations and beam-search outputs
graphs/                   # beam-search plots
beam_search_results.csv   # BLEU/time measurements for beam sizes 1–10
README_Results.md         # experiment results and analysis
```

## Submission

The repository contains:

- all scripts required to reproduce preprocessing, training, evaluation, and beam-search experiments,
- trained model artifacts,
- beam-search measurements and plots,
- experimental results and discussion in `README_Results.md`.

**AI Disclosure:** GenAI was used to create this README.
