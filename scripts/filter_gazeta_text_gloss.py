#!/usr/bin/env python3
"""Filter scored_v2_gazeta_text_gloss.jsonl per dataset cleanup rules."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXACT_TEXT_REMOVE = frozenset(
    {
        "Reklama huquqlari asosida.",
        "O'zbekiston prezidenti matbuot xizmati.",
    }
)

# Multi-word capitalized spans that are places/orgs, not people.
_NON_PERSON_PHRASE_PARTS = (
    "markaziy osiyo",
    "janubiy koreya",
    "buyuk britaniya",
    "yevropa ittifoqi",
    "o'zbekiston",
    "toshkent",
    "airways",
    "mahkamasi",
    "universitet",
    "respublikasi",
    "viloyati",
    "birlashgan",
    "development",
    "ittifoq",
    "konsalting",
    "holding",
    "bank",
    "federatsiyasi",
    "kompaniyasi",
    "agentligi",
)

PERSON_NAME_HINTS = (
    "mirziyoyev",
    "erdogan",
    "erdog'an",
    "tayyip",
    "putin",
    "biden",
    "trump",
    "macron",
    "zelenskiy",
    "nazarbayev",
    "rais",
)

SURNAME_LIKE_ENDINGS = (
    "ev",
    "ov",
    "ich",
    "yan",
    "zad",
    "gin",
    "g'in",
    "qul",
    "ova",
    "ova",
)

CAP_NAME_RE = re.compile(
    r"\b[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){1,3}\b"
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MONTH_RE = re.compile(
    r"\b(yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sentabr|oktabr|noyabr|dekabr)\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
NUM_RE = re.compile(r"\d+")
WORD_RE = re.compile(r"[\w\u0400-\u04FF]+(?:'[\w\u0400-\u04FF]+)?", re.UNICODE)

# Sentence content — lines with these are not "names only".
_UZ_CONTENT_WORDS = frozenset(
    {
        "va",
        "uchun",
        "bilan",
        "dan",
        "ga",
        "da",
        "bu",
        "har",
        "esa",
        "edi",
        "bo'l",
        "qildi",
        "qilmoq",
        "bo'ldi",
        "tomonidan",
        "asosida",
        "kompaniyasi",
        "muvofiq",
        "orasida",
        "keyin",
        "oldin",
        "ham",
        "yoki",
        "lekin",
        "agar",
        "deb",
        "kerak",
        "mumkin",
        "ko'ra",
        "yo'l",
        "ish",
        "mamlakat",
    }
)

COMPANY_TAIL_RE = re.compile(
    r"(?:,\s*)?(?:Co\.|Ltd\.?|Inc\.?|LLC|GmbH|Corp\.?)\s*\.?\s*$",
    re.IGNORECASE,
)
COMPANY_WORD_RE = re.compile(
    r"\b(Equipment|Amusement|Machinery|Automobile|Technologies|Technology|Company|"
    r"Corporation|Group|Holding|Industries|Enterprises|Solutions|International)\b",
    re.IGNORECASE,
)
TITLE_CASE_RE = re.compile(r"^[A-Z][a-zA-Z']{2,}$")


def _is_name_like_token(token: str) -> bool:
    if TITLE_CASE_RE.match(token):
        return True
    if token.isupper() and len(token) >= 2 and token.isascii():
        return True
    if token.casefold().endswith("ning") and token[0].isupper():
        return True
    return False


def _has_uzbek_sentence_content(tokens: list[str]) -> bool:
    for t in tokens:
        tl = t.casefold()
        if tl in _UZ_CONTENT_WORDS:
            return True
        if re.search(r"(moq|ligi|larida|langan|qilgan|sifatida|tomonida)$", tl):
            return True
        if tl.endswith(("di", "gan", "lar", "moq")) and not tl.endswith("ning"):
            return True
    return False


def is_name_or_company_only(text: str) -> bool:
    """True when text is essentially a person/company name listing, not a sentence."""
    text = text.strip()
    tokens = WORD_RE.findall(text)
    if len(tokens) < 2:
        return False

    if _has_uzbek_sentence_content(tokens):
        return False

    name_like = sum(1 for t in tokens if _is_name_like_token(t))
    ratio = name_like / len(tokens)

    if COMPANY_TAIL_RE.search(text) and ratio >= 0.5:
        return True

    if COMPANY_WORD_RE.search(text) and ratio >= 0.6:
        return True

    # e.g. Shavkat Mirziyoyev or Xitoyning Zhengzhou Beston ...
    if ratio >= 0.85 and name_like >= 2:
        return True

    if ratio >= 0.75 and name_like >= 3 and len(tokens) <= 12:
        return True

    return False


def _is_person_name_phrase(phrase: str) -> bool:
    low = phrase.casefold().strip()
    if any(part in low for part in _NON_PERSON_PHRASE_PARTS):
        return False
    if any(h in low for h in PERSON_NAME_HINTS):
        return True
    words = phrase.split()
    if len(words) < 2 or len(words) > 4:
        return False
    last = words[-1].casefold()
    if any(last.endswith(end) for end in SURNAME_LIKE_ENDINGS):
        return True
    # Two+ capitalized tokens, middle looks like a given name (length >= 3).
    if len(words) == 2 and all(len(w) >= 3 for w in words):
        return True
    if len(words) == 3 and all(len(w) >= 2 for w in words):
        return True
    return False


def has_person_name(text: str) -> bool:
    low = text.casefold()
    if any(h in low for h in PERSON_NAME_HINTS):
        return True
    for m in CAP_NAME_RE.finditer(text):
        if _is_person_name_phrase(m.group()):
            return True
    return False


def has_date_or_year(text: str) -> bool:
    return bool(YEAR_RE.search(text) or MONTH_RE.search(text) or DATE_RE.search(text))


def count_numbers(text: str) -> int:
    return len(NUM_RE.findall(text))


def filter_file(
    inp: Path,
    out: Path,
    *,
    seed: int = 42,
) -> dict[str, int]:
    rng = random.Random(seed)
    records: list[dict] = []
    stats = {
        "input": 0,
        "exact_text": 0,
        "multi_number": 0,
        "name_half": 0,
        "date_half": 0,
        "name_company_only": 0,
        "kept": 0,
    }

    with inp.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["input"] += 1
            records.append(json.loads(line))

    name_idxs = [i for i, r in enumerate(records) if has_person_name(r.get("text", ""))]
    date_idxs = [i for i, r in enumerate(records) if has_date_or_year(r.get("text", ""))]

    remove_name = set(rng.sample(name_idxs, len(name_idxs) // 2)) if name_idxs else set()
    remove_date = set(rng.sample(date_idxs, len(date_idxs) // 2)) if date_idxs else set()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fout:
        for i, rec in enumerate(records):
            text = str(rec.get("text", "")).strip()
            if text in EXACT_TEXT_REMOVE:
                stats["exact_text"] += 1
                continue
            if count_numbers(text) > 1:
                stats["multi_number"] += 1
                continue
            if is_name_or_company_only(text):
                stats["name_company_only"] += 1
                continue
            if i in remove_name:
                stats["name_half"] += 1
                continue
            if i in remove_date:
                stats["date_half"] += 1
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    return stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "scored_v2_gazeta_text_gloss.jsonl",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: overwrite input",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.output or args.input
    stats = filter_file(args.input, out, seed=args.seed)
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
