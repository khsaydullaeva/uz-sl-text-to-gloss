"""Run held-out evaluation and write comparison / prediction artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .config import (
    DATA_DIR,
    DEFAULT_GEMMA_BASE,
    DEFAULT_MBART_BASE,
    DEFAULT_MBART_SRC_LANG,
    DEFAULT_MBART_TGT_LANG,
    DEFAULT_NLLB_BASE,
    DEFAULT_NLLB_LANG,
    DEFAULT_QWEN_BASE,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    USER_PROMPT,
    USER_PROMPT_GLOSS_TO_TEXT,
    resolve_model_dir,
)
from .data import load_test
from .metrics import score_predictions
from .predict import (
    predict_gemma_batch,
    predict_mbart_batch,
    predict_nllb_batch,
    predict_qwen_batch,
    predict_sockeye_batch,
)


def _write_preds(
    path: Path, ids: list[str], inputs: list[str], refs: list[str], hyps: list[str]
) -> None:
    """Write id/input/reference/prediction rows — `inputs`/`refs` must be
    whatever was actually fed to the model and scored against (already
    swapped for gloss2text), not blindly `test["text"]`/`test["reference"]`,
    or the file would record the wrong side of the pair as "the input"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row_id, inp, ref, hyp in zip(ids, inputs, refs, hyps):
            f.write(
                json.dumps(
                    {"id": row_id, "text": inp, "reference": ref, "prediction": hyp},
                    ensure_ascii=False,
                )
                + "\n"
            )


def evaluate(
    *,
    data_dir: Path | None = None,
    only: str = "all",
    direction: str = "text2gloss",
    beam_size: int = 5,
    max_length: int = 64,
    qwen_max_new_tokens: int = 64,
    gemma_max_new_tokens: int = 64,
    output: Path | None = None,
    predictions_dir: Path | None = None,
    sockeye_python: str | None = None,
) -> dict:
    """Evaluate selected models on the test split. Returns the comparison dict.

    direction="gloss2text" scores the "*_g2t" checkpoints (gloss in, sentence
    out) against the same held-out rows, just with input/reference swapped —
    it's the same parallel test set, not a separate one.
    """
    if direction not in ("text2gloss", "gloss2text"):
        raise ValueError(f"unknown direction: {direction!r}")
    g2t = direction == "gloss2text"
    ck_suffix = "_g2t" if g2t else ""

    data_dir = data_dir or DATA_DIR
    output = output or (OUTPUTS_DIR / ("comparison_gloss2text.json" if g2t else "comparison.json"))
    pred_dir = predictions_dir or (OUTPUTS_DIR / ("predictions_gloss2text" if g2t else "predictions"))
    sockeye_python = sockeye_python or str(PROJECT_ROOT / ".venv-sockeye" / "bin" / "python")

    test = load_test(data_dir)
    ids = [r["id"] for r in test]
    if g2t:
        texts = [r["reference"] for r in test]
        refs = [r["text"] for r in test]
    else:
        texts = [r["text"] for r in test]
        refs = [r["reference"] for r in test]

    if output.exists():
        results: dict = json.loads(output.read_text(encoding="utf-8"))
        stale_size = results.get("test_size")
        if stale_size is not None and stale_size != len(test) and results.get("models"):
            print(
                f"WARNING: {output} has cached results scored against test_size="
                f"{stale_size}, but the current test split has {len(test)} examples. "
                "Existing entries in 'models' were scored on a different test set and "
                "are not comparable to any new entries this run adds — re-run with "
                "only='all' to rescore every model on the current split.",
                file=sys.stderr,
            )
    else:
        results = {"test_size": len(test), "models": {}}
    results["test_size"] = len(test)
    results.setdefault("models", {})
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Drop cached results for checkpoints that no longer exist, so a deleted/
    # abandoned model doesn't linger in comparison.json forever.
    for pred_name, dirname in (
        ("sockeye3", "sockeye"),
        ("mbart50", "mbart"),
        ("nllb200", "nllb"),
        ("qwen25", "qwen"),
        ("gemma2", "gemma"),
    ):
        key = f"{pred_name}{ck_suffix}"
        if key in results["models"] and not resolve_model_dir(f"{dirname}{ck_suffix}").exists():
            del results["models"][key]

    if not g2t:
        sockeye_model = resolve_model_dir("sockeye")
        if only in ("all", "sockeye") and sockeye_model.exists():
            spm = data_dir / "sockeye" / "spm.model"
            if not spm.is_file():
                from .config import resolve_spm_model

                spm = resolve_spm_model()
            hyps = predict_sockeye_batch(
                sockeye_model, spm, texts, beam_size=beam_size, python=sockeye_python
            )
            results["models"]["sockeye3"] = score_predictions("sockeye3", hyps, refs)
            _write_preds(pred_dir / "sockeye3_test.jsonl", ids, texts, refs, hyps)
        elif only in ("all", "sockeye"):
            print(f"Skipping Sockeye: model not found at {sockeye_model}", file=sys.stderr)
    elif only in ("all", "sockeye"):
        print("Skipping Sockeye: not trained for gloss2text", file=sys.stderr)

    mbart_key = f"mbart50{ck_suffix}"
    mbart_model = resolve_model_dir(f"mbart{ck_suffix}")
    if only in ("all", "mbart") and mbart_model.exists():
        hyps = predict_mbart_batch(
            mbart_model,
            texts,
            base_model=DEFAULT_MBART_BASE,
            src_lang=DEFAULT_MBART_TGT_LANG if g2t else DEFAULT_MBART_SRC_LANG,
            tgt_lang=DEFAULT_MBART_SRC_LANG if g2t else DEFAULT_MBART_TGT_LANG,
            max_length=max_length,
            num_beams=beam_size,
        )
        results["models"][mbart_key] = score_predictions(mbart_key, hyps, refs)
        _write_preds(pred_dir / f"{mbart_key}_test.jsonl", ids, texts, refs, hyps)
    elif only in ("all", "mbart"):
        print(f"Skipping mBART: model not found at {mbart_model}", file=sys.stderr)

    nllb_key = f"nllb200{ck_suffix}"
    nllb_model = resolve_model_dir(f"nllb{ck_suffix}")
    if only in ("all", "nllb") and nllb_model.exists():
        hyps = predict_nllb_batch(
            nllb_model,
            texts,
            base_model=DEFAULT_NLLB_BASE,
            lang=DEFAULT_NLLB_LANG,
            max_length=max_length,
            num_beams=beam_size,
        )
        results["models"][nllb_key] = score_predictions(nllb_key, hyps, refs)
        _write_preds(pred_dir / f"{nllb_key}_test.jsonl", ids, texts, refs, hyps)
    elif only in ("all", "nllb"):
        print(f"Skipping NLLB: model not found at {nllb_model}", file=sys.stderr)

    qwen_key = f"qwen25{ck_suffix}"
    qwen_adapter = resolve_model_dir(f"qwen{ck_suffix}")
    if only in ("all", "qwen") and qwen_adapter.exists():
        hyps = predict_qwen_batch(
            qwen_adapter,
            texts,
            base_model=DEFAULT_QWEN_BASE,
            max_new_tokens=qwen_max_new_tokens,
            prompt_template=USER_PROMPT_GLOSS_TO_TEXT if g2t else USER_PROMPT,
        )
        results["models"][qwen_key] = score_predictions(qwen_key, hyps, refs)
        _write_preds(pred_dir / f"{qwen_key}_test.jsonl", ids, texts, refs, hyps)
    elif only in ("all", "qwen"):
        print(f"Skipping Qwen: adapter not found at {qwen_adapter}", file=sys.stderr)

    gemma_key = f"gemma2{ck_suffix}"
    gemma_adapter = resolve_model_dir(f"gemma{ck_suffix}")
    if only in ("all", "gemma") and gemma_adapter.exists():
        hyps = predict_gemma_batch(
            gemma_adapter,
            texts,
            base_model=DEFAULT_GEMMA_BASE,
            max_new_tokens=gemma_max_new_tokens,
            prompt_template=USER_PROMPT_GLOSS_TO_TEXT if g2t else USER_PROMPT,
        )
        results["models"][gemma_key] = score_predictions(gemma_key, hyps, refs)
        _write_preds(pred_dir / f"{gemma_key}_test.jsonl", ids, texts, refs, hyps)
    elif only in ("all", "gemma"):
        print(f"Skipping Gemma: adapter not found at {gemma_adapter}", file=sys.stderr)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
