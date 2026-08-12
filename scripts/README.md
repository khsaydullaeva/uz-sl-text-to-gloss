# Scripts

- `run_training.sh` — data-prep step before training. Rebuilds `data/splits`
  + `data/mbart` from the newest file in `data/raw/*.jsonl` (no need to
  update a hardcoded filename after each new annotation export). Pure
  stdlib, so plain `python3` is enough — no venv/ML deps needed for this
  step. **Training itself stays in the notebooks** — run
  `notebooks/03_train_mbart.ipynb` .. `06_train_gemma.ipynb` manually in
  VS Code / Jupyter against the `.venv-text-gloss-mbart` kernel, same as
  `notebooks/02_prepare_data.ipynb` (which this script mirrors and which
  also now auto-picks the newest raw dataset).

  ```bash
  scripts/run_training.sh
  ```

- `fetch_weights.sh` — downloads fine-tuned checkpoints from the `models-v1` GitHub
  release and reassembles them under `models/<model>/` (see [Checkpoints](../README.md#checkpoints)).

  ```bash
  scripts/fetch_weights.sh
  ```

Optional corpus / baseline utilities (moved from `text_to_gloss/baselines/`):

- `build_gazeta_text_gloss.py` — build Gazeta text↔gloss pairs
- `filter_gazeta_text_gloss.py` — filter / score the corpus
- `text_to_gloss_uzlexicon.py` — lexicon baseline
