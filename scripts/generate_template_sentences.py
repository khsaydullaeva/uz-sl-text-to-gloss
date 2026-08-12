#!/usr/bin/env python3
"""Generate template-based text/gloss sentence pairs from curated per-topic vocabulary.

Reads the human-curated `data/topic_vocab_review.csv` (produced and pruned via
`export_topic_vocab.py` - see that script's docstring for why curation is needed:
`signs.csv` mixes nouns with pronouns/modals/interjections that don't fit these
noun-slot templates). For each topic, cycles its kept words through a small set
of simple/advanced sentence templates.

Usage:
    python scripts/export_topic_vocab.py               # 1. write data/topic_vocab_review.csv
    # ... review it, flip keep to 'n' on non-nouns ...
    python scripts/generate_template_sentences.py       # 2. writes data/generated_sentences.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TEMPLATES = {
    "simple": [
        "Bu {item} juda chiroyli.",
        "Men bugun {item} ko'rdim.",
        "Bizga yangi {item} kerak.",
        "U {item} sotib oldi.",
        "Sizda {item} bormi?",
    ],
    "advanced": [
        "{item} rivojlanishi jamiyat taraqqiyotini belgilaydi.",
        "Zamonaviy {item} tizimlari tubdan isloh qilindi.",
        "Ushbu {item} sohasida ko'plab muammolar mavjud.",
        "Strategik {item} rejasi muvaffaqiyatli yakunlandi.",
    ],
}

SENTENCES_PER_LEVEL = 250


def load_curated_vocab(review_csv: Path) -> dict[str, list[str]]:
    """topic_uz -> [label_uz, ...] for rows marked keep=y, in file order."""
    words: dict[str, list[str]] = {}
    with open(review_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("keep") or "").strip().lower() not in ("y", "yes", "1", "true"):
                continue
            label = (r.get("label_uz") or "").strip()
            topic = (r.get("topic_uz") or "").strip()
            if label and topic:
                words.setdefault(topic, []).append(label)
    return words


def render(template: str, word: str) -> str:
    text = template.format(item=word)
    # Only the templates where {item} is the first token need it capitalized;
    # the others already open with a capitalized fixed word (Bu, Zamonaviy, ...).
    if template.startswith("{item}") and text:
        text = text[0].upper() + text[1:]
    return text


def build(words_by_topic: dict[str, list[str]]) -> dict[str, list[dict]]:
    dataset: dict[str, list[dict]] = {}

    for topic_uz, words in words_by_topic.items():
        rows = []
        for i in range(SENTENCES_PER_LEVEL):
            simp_tpl = TEMPLATES["simple"][i % len(TEMPLATES["simple"])]
            word = words[i % len(words)]
            text = render(simp_tpl, word)
            rows.append({"level": "simple", "variant_id": i, "sentence": text, "gloss": text})

            adv_tpl = TEMPLATES["advanced"][i % len(TEMPLATES["advanced"])]
            word = words[(i + 1) % len(words)]
            text = render(adv_tpl, word)
            rows.append({"level": "advanced", "variant_id": i, "sentence": text, "gloss": text})

        dataset[topic_uz] = rows

    return dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--vocab-review", type=Path,
                         default=repo_root / "data" / "topic_vocab_review.csv",
                         help="curated vocab sheet from export_topic_vocab.py")
    parser.add_argument("-o", "--output", type=Path,
                         default=repo_root / "data" / "generated_sentences.json",
                         help="where to write the generated sentences")
    args = parser.parse_args()

    if not args.vocab_review.is_file():
        print(f"{args.vocab_review} not found.\n"
              f"Run scripts/export_topic_vocab.py first, review the 'keep' column, "
              f"then re-run this script.", file=sys.stderr)
        return 1

    words_by_topic = load_curated_vocab(args.vocab_review)
    print(f"Generating sentences across {len(words_by_topic)} topics "
          f"({SENTENCES_PER_LEVEL} simple + {SENTENCES_PER_LEVEL} advanced each)...")

    dataset = build(words_by_topic)
    total = sum(len(rows) for rows in dataset.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {total} sentences across {len(dataset)} topics to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
