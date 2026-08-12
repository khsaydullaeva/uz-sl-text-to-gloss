#!/usr/bin/env bash
# Prep step before training: rebuilds data/splits + data/mbart from the
# newest file in data/raw/*.jsonl (no need to hand-edit a filename after
# each new annotation export).
#
# This mirrors notebooks/02_prepare_data.ipynb (stage 02 of the pipeline;
# stage 01 is the optional dataset_exploration notebook). Training itself is
# NOT run from here — it stays in notebooks/03_train_mbart.ipynb ..
# 06_train_gemma.ipynb, run manually in VS Code / Jupyter against the
# .venv-text-gloss-mbart kernel.
#
# Usage:
#   scripts/run_training.sh
#
# Data prep uses only the standard library (see src/ttg/data.py), so plain
# python3 is enough -- no venv/ML deps required for this step.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${TTG_PYTHON:-python3}"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - <<'EOF'
import json
from ttg.data import latest_raw_dataset, prepare_splits

input_path = latest_raw_dataset()
print(f"Using latest raw dataset: {input_path}")

meta = prepare_splits(input_path=input_path)
print(json.dumps(meta, indent=2, ensure_ascii=False))
EOF

cat <<'EOF'

Data prepared (equivalent to notebooks/02_prepare_data.ipynb). Now run the
training notebooks in order (VS Code / Jupyter, .venv-text-gloss-mbart
kernel):

  notebooks/01_dataset_exploration.ipynb  -> optional, peek at raw annotations
  notebooks/02_prepare_data.ipynb         -> already done by this script
  notebooks/03_train_mbart.ipynb          -> artifacts/ckpts/mbart/best
  notebooks/04_train_nllb.ipynb           -> artifacts/ckpts/nllb/best
  notebooks/05_train_qwen.ipynb           -> artifacts/ckpts/qwen/best
  notebooks/06_train_gemma.ipynb          -> artifacts/ckpts/gemma/best
  notebooks/07_evaluate.ipynb             -> artifacts/comparison.json
EOF
