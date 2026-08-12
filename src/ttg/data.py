"""Dataset loading and split preparation."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, PROJECT_ROOT, SEED


@dataclass
class SplitData:
    texts: list[str]
    glosses: list[str]
    ids: list[str]


def latest_raw_dataset(raw_dir: Path | None = None) -> Path:
    """Return the most recently modified sentences-gloss jsonl under data/raw/."""
    raw_dir = raw_dir or (DATA_DIR / "raw")
    candidates = sorted(raw_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no raw datasets found in {raw_dir}")
    return candidates[-1]


def load_raw(path: Path) -> list[dict]:
    """Load a JSON array (often stored with a .jsonl extension)."""
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{path}: expected JSON array")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_split(path: Path) -> SplitData:
    texts: list[str] = []
    glosses: list[str] = []
    ids: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = str(rec.get("text", "")).strip()
            gloss = str(rec.get("gloss", "")).strip()
            rid = str(rec.get("id", f"line_{line_no}"))
            if not text or not gloss:
                raise ValueError(f"{path}:{line_no}: missing text or gloss")
            texts.append(text)
            glosses.append(gloss)
            ids.append(rid)
    return SplitData(texts=texts, glosses=glosses, ids=ids)


def load_test(data_dir: Path | None = None) -> list[dict[str, str]]:
    data_dir = data_dir or DATA_DIR
    path = data_dir / "mbart" / "test.jsonl"
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append(
                {
                    "id": str(rec["id"]),
                    "text": str(rec["text"]),
                    "reference": str(rec["gloss"]),
                }
            )
    return rows


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


def _dedup_by_text(rows: list[dict]) -> tuple[list[dict], int]:
    """Keep one row per unique (stripped) `text`, preferring the most recently updated."""
    by_text: dict[str, dict] = {}
    for row in rows:
        text = str(row.get("text", "")).strip()
        existing = by_text.get(text)
        if existing is None or _parse_timestamp(str(row.get("updated_at", ""))) > _parse_timestamp(
            str(existing.get("updated_at", ""))
        ):
            by_text[text] = row
    deduped = list(by_text.values())
    return deduped, len(rows) - len(deduped)


def prepare_splits(
    *,
    input_path: Path | None = None,
    out_dir: Path | None = None,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = SEED,
    lowercase_gloss: bool = True,
) -> dict:
    """Build train/dev/test splits and mBART jsonl from raw annotations."""
    input_path = input_path or latest_raw_dataset()
    out_dir = out_dir or DATA_DIR

    raw_rows = [r for r in load_raw(input_path) if r.get("is_correct", True)]
    num_raw = len(raw_rows)
    rows, num_duplicates_collapsed = _dedup_by_text(raw_rows)
    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_test = max(1, int(n * test_ratio))
    n_dev = max(1, int(n * dev_ratio))
    test = rows[:n_test]
    dev = rows[n_test : n_test + n_dev]
    train = rows[n_test + n_dev :]

    splits_dir = out_dir / "splits"
    mbart_dir = out_dir / "mbart"
    splits_dir.mkdir(parents=True, exist_ok=True)
    mbart_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_rows in ("train", train), ("dev", dev), ("test", test):
        split_recs: list[dict] = []
        mbart_recs: list[dict] = []
        for rec in split_rows:
            text = str(rec["text"]).strip()
            gloss = str(rec["gloss"]).strip()
            if lowercase_gloss:
                gloss = gloss.lower()
            rid = str(rec.get("public_id", rec.get("id", "")))
            split_recs.append({"id": rid, "text": text, "gloss": gloss})
            mbart_recs.append({"id": rid, "text": text, "gloss": gloss})
        write_jsonl(splits_dir / f"{split_name}.jsonl", split_recs)
        write_jsonl(mbart_dir / f"{split_name}.jsonl", mbart_recs)

    train_texts = {str(r["text"]).strip() for r in train}
    dev_texts = {str(r["text"]).strip() for r in dev}
    test_texts = {str(r["text"]).strip() for r in test}
    if train_texts & dev_texts or train_texts & test_texts or dev_texts & test_texts:
        raise AssertionError(
            "prepare_splits: train/dev/test text sets are not disjoint after dedup"
        )

    meta = {
        "input": str(
            input_path.relative_to(PROJECT_ROOT)
            if input_path.is_relative_to(PROJECT_ROOT)
            else input_path
        ),
        "num_raw": num_raw,
        "num_after_dedup": n,
        "num_duplicate_text_groups_collapsed": num_duplicates_collapsed,
        "num_total": n,
        "num_train": len(train),
        "num_dev": len(dev),
        "num_test": len(test),
        "dev_ratio": dev_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "lowercase_gloss": lowercase_gloss,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return meta


def swap_direction(split: SplitData) -> SplitData:
    """Flip a SplitData so gloss becomes the model input and text the target,
    for training a gloss -> text model with the same data pipeline."""
    return SplitData(texts=split.glosses, glosses=split.texts, ids=split.ids)


def to_chat_messages(split: SplitData, user_prompt: str) -> list[dict]:
    rows: list[dict] = []
    for text, gloss in zip(split.texts, split.glosses):
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_prompt.format(text=text)},
                    {"role": "assistant", "content": gloss},
                ]
            }
        )
    return rows
