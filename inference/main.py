#!/usr/bin/env python3
"""Uzbek text → gloss inference CLI.

    python main.py --text "Men uyda o'tiribman." --model nllb
    python main.py --text "..." --model all --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    from ttg.config import (
        DEFAULT_GEMMA_BASE,
        DEFAULT_MBART_BASE,
        DEFAULT_MBART_SRC_LANG,
        DEFAULT_MBART_TGT_LANG,
        DEFAULT_NLLB_BASE,
        DEFAULT_NLLB_LANG,
        DEFAULT_QWEN_BASE,
        USER_PROMPT,
        USER_PROMPT_GLOSS_TO_TEXT,
        resolve_model_dir,
        resolve_spm_model,
    )
    from ttg.predict import (
        predict_gemma,
        predict_mbart,
        predict_nllb,
        predict_qwen,
        predict_sockeye,
    )

    ap = argparse.ArgumentParser(description="Uzbek sentence <-> sign-language gloss")
    ap.add_argument("--text", required=True, help="Uzbek sentence (or gloss, for --direction gloss2text)")
    ap.add_argument(
        "--model",
        choices=("sockeye", "mbart", "nllb", "qwen", "gemma", "all"),
        default="nllb",
    )
    ap.add_argument(
        "--direction",
        choices=("text2gloss", "gloss2text"),
        default="text2gloss",
        help="gloss2text is not trained for sockeye (no checkpoint exists)",
    )
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args()
    g2t = args.direction == "gloss2text"

    out: dict[str, str] = {"text": args.text}
    models = (
        ("sockeye", "mbart", "nllb", "qwen", "gemma")
        if args.model == "all"
        else (args.model,)
    )

    for name in models:
        # Direction-specific checkpoints live under a "*_g2t" dir; resolve_model_dir
        # naturally reports "not found" for sockeye here since no such checkpoint
        # was ever trained (mbart/nllb/qwen/gemma all have one).
        checkpoint_key = f"{name}_g2t" if g2t else name
        path = resolve_model_dir(checkpoint_key)
        if not path.exists():
            print(f"skip {name}: not found at {path}", file=sys.stderr)
            continue
        if name == "sockeye":
            out["sockeye3"] = predict_sockeye(
                args.text, path, resolve_spm_model(), beam_size=args.beam_size
            )
        elif name == "mbart":
            out[f"mbart50{'_g2t' if g2t else ''}"] = predict_mbart(
                args.text,
                path,
                base_model=DEFAULT_MBART_BASE,
                src_lang=DEFAULT_MBART_TGT_LANG if g2t else DEFAULT_MBART_SRC_LANG,
                tgt_lang=DEFAULT_MBART_SRC_LANG if g2t else DEFAULT_MBART_TGT_LANG,
                max_length=args.max_length,
                num_beams=args.beam_size,
            )
        elif name == "nllb":
            # uzn_Latn is used for both src and tgt, so the direction swap
            # needs no language-code changes here.
            out[f"nllb200{'_g2t' if g2t else ''}"] = predict_nllb(
                args.text,
                path,
                base_model=DEFAULT_NLLB_BASE,
                lang=DEFAULT_NLLB_LANG,
                max_length=args.max_length,
                num_beams=args.beam_size,
            )
        elif name == "qwen":
            out[f"qwen25{'_g2t' if g2t else ''}"] = predict_qwen(
                args.text,
                path,
                base_model=DEFAULT_QWEN_BASE,
                max_new_tokens=args.max_new_tokens,
                prompt_template=USER_PROMPT_GLOSS_TO_TEXT if g2t else USER_PROMPT,
            )
        elif name == "gemma":
            out[f"gemma2{'_g2t' if g2t else ''}"] = predict_gemma(
                args.text,
                path,
                base_model=DEFAULT_GEMMA_BASE,
                max_new_tokens=args.max_new_tokens,
                prompt_template=USER_PROMPT_GLOSS_TO_TEXT if g2t else USER_PROMPT,
            )

    preds = {k: v for k, v in out.items() if k != "text"}
    if not preds:
        print("error: no models available", file=sys.stderr)
        return 2

    if args.json or args.model == "all":
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(next(iter(preds.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
