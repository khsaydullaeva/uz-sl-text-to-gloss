# uzsl-text-to-gloss — Uzbek Text → Sign Gloss

Translates an **Uzbek sentence** into a **sign-language gloss** sequence
(space-separated lemma-like tokens). Built around fine-tuned NLLB / mBART and
optional LoRA adapters (Qwen, Gemma).

## Repository layout

```
notebooks/     THE training pipeline — staged, documented, runnable notebooks
src/ttg/       small library the notebooks + inference import
inference/     CLI predict + local metrics dashboard
scripts/       optional corpus builders / lexicon baseline
docs/          short overview
data/          raw annotations + frozen train/dev/test splits
artifacts/     predictions, comparison.json, ckpts/ (weights — gitignored)
```

## Setup

```bash
conda create -n uzsl-ttg python=3.11 -y
conda activate uzsl-ttg
pip install -r requirements.txt
```

Gemma models on Hugging Face may require accepting the model license and
`huggingface-cli login` before training or evaluation.

## How to train

Everything is in `notebooks/` — run them **in order**. Each notebook states its
inputs, outputs, and knobs (`SMOKE_TEST`, epochs, …) in the first cells.

| # | Notebook | What it does |
|---|---|---|
| 01 | `dataset_exploration` | peek at raw annotations (optional) |
| 02 | `prepare_data` | build train/dev/test + mbart jsonl |
| 03 | `train_mbart` | fine-tune mBART-50 → `artifacts/ckpts/mbart/best` |
| 04 | `train_nllb` | fine-tune NLLB-200 → `artifacts/ckpts/nllb/best` |
| 05 | `train_qwen` | LoRA Qwen2.5 → `artifacts/ckpts/qwen/best` |
| 06 | `train_gemma` | LoRA Gemma-2 → `artifacts/ckpts/gemma/best` |
| 07 | `evaluate` | BLEU / chrF / gloss-F1 → `artifacts/comparison.json` |

Set `SMOKE_TEST = True` in a train notebook for a 1-epoch plumbing check.
Notebooks 03–06 each have a `DIRECTION = "text2gloss" | "gloss2text"` toggle;
`gloss2text` trains the reverse direction into a sibling `*_g2t` checkpoint
dir (e.g. `artifacts/ckpts/nllb_g2t/best`) using the same train/dev split
with input/target swapped.

## Results

Both directions are scored on the same 130-example held-out test split
(`data/mbart/test.jsonl`) — gloss → text just swaps which field is model
input vs. scoring reference, not a separate test set. Sockeye has no
checkpoint in either direction (never trained) and is omitted below.

### Text → Gloss (`artifacts/comparison.json`)

| Model | BLEU | chrF | Gloss-token F1 | Exact match |
|---|---|---|---|---|
| **NLLB-200** | **27.52** | **61.35** | **60.24%** | 0.77% |
| Gemma 2 (LoRA) | 24.51 | 57.08 | 56.26% | **1.54%** |
| Qwen2.5 (LoRA) | 21.53 | 54.90 | 52.79% | 0.00% |
| mBART-50 | 20.10 | 55.46 | 52.58% | 0.77% |

### Gloss → Text (`artifacts/comparison_gloss2text.json`)

| Model | BLEU | chrF | Gloss-token F1 | Exact match |
|---|---|---|---|---|
| **NLLB-200** | **17.29** | **53.81** | **40.90%** | 0.00% |
| Gemma 2 (LoRA) | 5.41 | 43.22 | 26.81% | 0.00% |
| Qwen2.5 (LoRA) | 4.88 | 42.80 | 25.55% | 0.00% |
| mBART-50 | 2.45 | 30.38 | 18.38% | 0.00% |

NLLB-200 is the strongest backbone in both directions by a wide margin.
Gloss → text scores are lower across the board than text → gloss — going
from a handful of gloss keywords to a fully fluent sentence is a harder
generation task than the reverse, especially fine-tuned from only ~1,040
sentence pairs. mBART-50 struggles most at this direction: its language
tags are proxies (Turkish/English stand-ins, since mBART-50 has no real
Uzbek code) rather than a real fit for generating fluent Uzbek.

Re-run `notebooks/07_evaluate.ipynb` (set `DIRECTION = "gloss2text"` for the
second table) to refresh these after retraining.

## How to run inference

Activate the conda env first, then run from the `inference/` folder:

```bash
conda activate uzsl-ttg
cd inference

PYTHONPATH=../src python main.py --text "Men uyda o'tiribman." --model nllb
PYTHONPATH=../src python main.py --text "..." --model all --json

# local metrics + try-a-sentence UI, opens http://127.0.0.1:8765
PYTHONPATH=../src python ui.py --open
```

`--model` accepts `nllb`, `mbart`, `qwen`, `gemma`, `sockeye`, or `all`. Each
fine-tuned checkpoint must exist under `artifacts/ckpts/<model>/best/` (see
[Checkpoints](#checkpoints)) — inference for a model whose checkpoint is
missing will fail. Add `--direction gloss2text` to translate the other way
(`--text` is then the gloss); this uses the `*_g2t` checkpoints and is
available for every model except sockeye.

### Pose playback ("Play pose" button in the UI)

Rendering gloss → skeleton video needs a separate env (mediapipe pins numpy/protobuf
versions that conflict with the training stack) and lives in the sibling
`uzbek-sign-language` repo, not this one:

```bash
conda create -n uzsl-pose python=3.11 -y
conda run -n uzsl-pose pip install -e "../uzbek-sign-language/pose/src/python[mediapipe]"
conda run -n uzsl-pose pip install vidgear opencv-python
```

`ui.py` auto-detects `~/miniforge3/envs/uzsl-pose/bin/python` and the sibling
`uzbek-sign-language` checkout (for `record/tools/text_to_pose.py`) — no flags
needed if both live at their default locations. Override with `--pose-python` /
`--pose-repo-root` (or `$POSE_PYTHON` / `$POSE_REPO_ROOT`) if yours live elsewhere.
You'll also need the full UzSL dataset (`metadata/signs.csv`, `metadata/samples.csv`)
locally — point `--uzsl-data-dir` / `$UZSL_DATA_DIR` at it.

From Python:

```python
import sys
sys.path.insert(0, "src")
from ttg import predict_gloss
print(predict_gloss("Men uyda o'tiribman.", model="nllb"))
```

## Checkpoints

Best weights live under `artifacts/ckpts/<model>/best/` (gitignored; several GB).
They were moved here from `uzbek-sign-language/text_to_gloss/checkpoints/*/best`.
Intermediate epoch dumps (if any) may still sit in the old tree.

Weights aren't committed to git — they're published as GitHub release assets instead.
Fetch them with:

```bash
scripts/fetch_weights.sh          # downloads models-v1, reassembles into models/<model>/
```

`resolve_model_dir()` checks `models/<model>/` as a fallback location, so this works without
needing `artifacts/ckpts/` populated. See the
[models-v1 release](https://github.com/zehnmind/uzsl-text-to-gloss/releases/tag/models-v1)
for the raw assets (large files are split into <2GB parts; the script reassembles and
checksum-verifies them).
