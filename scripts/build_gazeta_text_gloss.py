#!/usr/bin/env python3
"""Convert sentence JSONL to text+gloss dataset via sign.mt / SLP text-to-gloss pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.slp_pipeline import (  # noqa: E402
    UZ_GLOSSER,
    gloss_to_string,
    guess_domain,
    text_to_gloss,
)

SPOKEN_LANG = "uz"


def convert(
    inp: Path,
    out: Path,
    *,
    id_prefix: str = "uzsl_stc",
    glosser: str = UZ_GLOSSER,
    limit: int | None = None,
) -> None:
    n = 0
    with inp.open(encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if limit is not None and n >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "")).strip()
            if not text:
                continue
            n += 1
            sentences = text_to_gloss(
                text,
                spoken_language=SPOKEN_LANG,
                glosser=glosser,
                fingerspell_unknown=False,
            )
            gloss = gloss_to_string(sentences)
            domain = guess_domain(text, sentences)
            out_rec = {
                "id": f"{id_prefix}_{n:06d}",
                "text": text,
                "gloss": gloss,
                "domain": domain,
                "spoken_language": SPOKEN_LANG,
                "signed_language": "uzs",
            }
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            if n % 50000 == 0:
                print(f"processed {n}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build text+gloss JSONL using sign.mt SLP text-to-gloss pipeline."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "scored_v2_gazeta_final_sentences.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "scored_v2_gazeta_text_gloss.jsonl",
    )
    p.add_argument(
        "--glosser",
        default=UZ_GLOSSER,
        help=f"SLP glosser module name (Uzbek: {UZ_GLOSSER})",
    )
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    convert(args.input, args.output, glosser=args.glosser, limit=args.limit)
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
