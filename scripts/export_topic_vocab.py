#!/usr/bin/env python3
"""Export per-topic isolated-sign vocabulary for manual noun/non-noun review.

`signs.csv` has no part-of-speech field, so grouping isolated signs by category
and feeding every label straight into noun-slot sentence templates produces
nonsense for pronouns, modals, and interjections mixed into the same category
(e.g. "biz" [we], "kerak" [is needed], "rahmat" [thanks] under `tanishuv`).
This dumps one row per (category, label) with a `keep` column defaulted to "y"
for a human to flip to "n" on anything that isn't a usable noun/noun phrase.
The curated file is then read by generate_template_sentences.py.

Usage:
    python scripts/export_topic_vocab.py            # writes data/topic_vocab_review.csv
    UZSL_DATA_DIR=/path/to/uzsl_data python scripts/export_topic_vocab.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Kept identical to generate_template_sentences.py's EXCLUDED_TOPICS. Category
# names in categories.csv are space-separated (e.g. "vaqt taqvim"), not
# underscore-separated.
EXCLUDED_TOPICS = ["alifbo", "kerak emas", "jumla buyicha so'zlar", "jumlalar"]


def default_uzsl_data() -> Path:
    return Path(os.environ.get("UZSL_DATA_DIR", str(Path.home() / "Projects" / "uzsl_data")))


def load_active_categories(categories_csv: Path) -> list[tuple[str, str]]:
    with open(categories_csv, newline="", encoding="utf-8") as f:
        rows = [(r["cat_id"], r["name_uz"]) for r in csv.DictReader(f)]
    return [(cid, name) for cid, name in rows if name not in EXCLUDED_TOPICS]


def load_isolated_signs(signs_csv: Path) -> list[tuple[str, str]]:
    """Returns [(cat_id, label_uz), ...] for isolated signs."""
    out = []
    with open(signs_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("sign_type") or "isolated") != "isolated":
                continue
            label = (r.get("label_uz") or "").strip()
            if label:
                out.append((r["cat_id"], label))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uzsl-data-dir", type=Path, default=default_uzsl_data(),
                         help="dataset root containing metadata/categories.csv and metadata/signs.csv")
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("-o", "--output", type=Path,
                         default=repo_root / "data" / "topic_vocab_review.csv",
                         help="where to write the review sheet")
    parser.add_argument("--force", action="store_true",
                         help="overwrite an existing review file (default: refuse, to protect curation)")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        print(f"{args.output} already exists (pass --force to regenerate and lose "
              f"any curation done on it).", file=sys.stderr)
        return 1

    categories_csv = args.uzsl_data_dir / "metadata" / "categories.csv"
    signs_csv = args.uzsl_data_dir / "metadata" / "signs.csv"
    for path in (categories_csv, signs_csv):
        if not path.is_file():
            print(f"{path.name} not found at {path}\n"
                  f"Set UZSL_DATA_DIR or pass --uzsl-data-dir.", file=sys.stderr)
            return 1

    categories = load_active_categories(categories_csv)
    topic_by_cat = dict(categories)
    signs = load_isolated_signs(signs_csv)

    rows = [
        {"cat_id": cat_id, "topic_uz": topic_by_cat[cat_id], "label_uz": label, "keep": "y"}
        for cat_id, label in signs
        if cat_id in topic_by_cat
    ]
    rows.sort(key=lambda r: (r["topic_uz"], r["label_uz"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cat_id", "topic_uz", "label_uz", "keep"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (topic, label) rows to {args.output}")
    print("Review it and flip 'keep' to 'n' for anything that isn't a usable noun/noun "
          "phrase (pronouns, modals like kerak/kerak emas, interjections like rahmat/ha, "
          "bare adjectives, etc.), then run generate_template_sentences.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
