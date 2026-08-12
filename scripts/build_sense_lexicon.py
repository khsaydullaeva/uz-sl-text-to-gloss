#!/usr/bin/env python3
"""Build a sense-lexicon skeleton from the UzSL sign dictionary.

A single Uzbek label (`label_uz`) can map to several `sign_id`s that mean
different things (homographs, e.g. `rasm` = картина vs фото). `signs.csv` has no
usable `sense` column (empty in every row), so the only built-in signal that
separates senses is `label_ru`. This script groups isolated signs by normalized
label, keeps every label with >1 sign_id, and emits a JSON skeleton seeded from
`label_ru` for a human to curate (`meaning_uz` + `examples`).

The output feeds src/ttg/disambiguate.py. Run once, then hand-fill the blank
`meaning_uz`/`examples` fields for the senses you care about (the disambiguator
falls back to `label_ru` for any left blank).

Usage:
    python scripts/build_sense_lexicon.py            # writes data/sense_lexicon.json
    python scripts/build_sense_lexicon.py -o out.json
    UZSL_DATA_DIR=/path/to/uzsl_data python scripts/build_sense_lexicon.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Self-contained: this repo and uzsl_data / the pose repo live in separate
# trees, so we do NOT import the sibling repo. Mirror its apostrophe handling —
# collapse every apostrophe-like character to a single canonical form and
# lowercase, so our labels line up with the sign dictionary's keys. The pose
# side re-normalizes override keys with its own normalizer as a safety net.
_APOSTROPHE_VARIANTS = "'‘’ʻʼˈ`´"
_CANONICAL_APOSTROPHE = "'"


def normalize_label(text: str) -> str:
    out = "".join(_CANONICAL_APOSTROPHE if ch in _APOSTROPHE_VARIANTS else ch
                  for ch in text)
    return out.strip().lower()


def default_uzsl_data() -> Path:
    return Path(os.environ.get("UZSL_DATA_DIR", str(Path.home() / "Projects" / "uzsl_data")))


def build(signs_csv: Path) -> dict[str, list[dict]]:
    # label -> [ {sign_id, label_ru}, ... ] for isolated signs only.
    groups: dict[str, list[dict]] = {}
    with open(signs_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("sign_type") or "isolated") != "isolated":
                continue
            label = normalize_label(r.get("label_uz") or "")
            if not label:
                continue
            groups.setdefault(label, []).append(
                {
                    "sign_id": r["sign_id"],
                    "label_ru": (r.get("label_ru") or "").strip(),
                    "meaning_uz": "",
                    "examples": [],
                }
            )
    # Keep only ambiguous labels (>1 sign_id). Sort senses by sign_id for stable
    # diffs; sort labels alphabetically.
    lexicon = {
        label: sorted(senses, key=lambda s: s["sign_id"])
        for label, senses in sorted(groups.items())
        if len(senses) > 1
    }
    return lexicon


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uzsl-data-dir", type=Path, default=default_uzsl_data(),
                        help="dataset root containing metadata/signs.csv")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("-o", "--output", type=Path,
                        default=repo_root / "data" / "sense_lexicon.json",
                        help="where to write the lexicon skeleton")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing lexicon (default: refuse, to protect curation)")
    args = parser.parse_args()

    signs_csv = args.uzsl_data_dir / "metadata" / "signs.csv"
    if not signs_csv.is_file():
        print(f"signs.csv not found at {signs_csv}\n"
              f"Set UZSL_DATA_DIR or pass --uzsl-data-dir.", file=sys.stderr)
        return 1

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists — refusing to overwrite curated data.\n"
              f"Pass --force to regenerate the skeleton (this discards curation).",
              file=sys.stderr)
        return 1

    lexicon = build(signs_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(lexicon, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")

    n_senses = sum(len(v) for v in lexicon.values())
    print(f"Wrote {len(lexicon)} ambiguous labels ({n_senses} senses) to {args.output}")
    print("Next: fill meaning_uz / examples for the senses you care about "
          "(label_ru is used as a fallback for any left blank).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
