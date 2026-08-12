"""Context-aware word-sense disambiguation for homograph gloss tokens.

A single Uzbek label can map to several signs with different meanings (e.g.
`rasm` = картина vs фото). The pose pipeline otherwise picks the most-recorded
sense regardless of context. This module reads the sentence, and for each
ambiguous label present, picks the sense whose curated meaning best matches the
context by sentence-embedding cosine similarity — returning a `{label: sign_id}`
override map the pose pipeline applies.

Fully local: no external API. The embedding model (default LaBSE, multilingual so
Uzbek context matches Russian/Uzbek sense descriptors) is loaded lazily and its
per-sense embeddings are cached to disk. Heavy deps (sentence-transformers,
numpy) are imported lazily so callers can import this module and degrade
gracefully when they're absent — see get_disambiguator().
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable

from .config import EMBEDDING_MODEL, SENSE_EMBED_CACHE_DIR, SENSE_LEXICON_PATH

logger = logging.getLogger(__name__)

# Keep normalization identical to scripts/build_sense_lexicon.py so that the
# labels we detect in a sentence match the lexicon's keys. The pose side
# re-normalizes override keys with its own normalizer as a final safety net.
_APOSTROPHE_VARIANTS = "'‘’ʻʼˈ`´"
_CANONICAL_APOSTROPHE = "'"

# Small, high-precision set of Uzbek case/possessive/plural suffixes, longest
# first, for detecting an inflected homograph ("rasmga" -> "rasm"). This is a
# light matcher, NOT the pose repo's full stemmer; a heavily inflected homograph
# that slips through simply falls back to the pipeline's default sense.
# Sorted longest-first so the greedy stripper prefers "ni" over "i", etc.
_SUFFIXES = tuple(sorted(
    {
        "laringizga", "laringizni", "larimizga", "larimizni",
        "ningki", "larni", "larga", "lardan", "larda", "larim", "laring",
        "ningiz", "imizga", "imizni",
        "ning", "dan", "lar", "miz", "ngiz",
        "ni", "ga", "da", "im", "ing", "si", "i",
    },
    key=len, reverse=True,
))


def normalize_label(text: str) -> str:
    out = "".join(_CANONICAL_APOSTROPHE if ch in _APOSTROPHE_VARIANTS else ch
                  for ch in text)
    return out.strip().lower()


def _candidate_forms(token: str) -> list[str]:
    """Surface form plus greedily suffix-stripped variants (e.g. tugmasini ->
    tugmasi -> tugma). Strips up to three stacked suffixes, longest-first, and
    keeps every intermediate form so the surface is always preferred."""
    forms = [token]
    cur = token
    for _ in range(3):
        stripped = None
        for suf in _SUFFIXES:
            if cur.endswith(suf) and len(cur) - len(suf) >= 3:
                stripped = cur[: -len(suf)]
                break
        if not stripped:
            break
        forms.append(stripped)
        cur = stripped
    return forms


def _tokenize(sentence: str) -> list[str]:
    """Word tokens, keeping intra-word apostrophes (bo'ldi), dropping punctuation."""
    norm = normalize_label(sentence)
    return re.findall(r"[a-z']+", norm)


def load_sense_lexicon(path: Path = SENSE_LEXICON_PATH) -> dict[str, list[dict]]:
    """Load the curated lexicon: label -> [{sign_id, label_ru, meaning_uz, examples}, ...]."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Normalize keys defensively; keep only genuinely ambiguous labels.
    return {normalize_label(k): v for k, v in data.items() if len(v) > 1}


def _sense_descriptor(sense: dict) -> str:
    """Text embedded to represent a sense: curated Uzbek meaning + examples,
    falling back to the Russian label when nothing has been curated yet."""
    parts = [sense.get("meaning_uz") or ""]
    parts += list(sense.get("examples") or [])
    text = " ".join(p for p in parts if p).strip()
    return text or (sense.get("label_ru") or "").strip()


class SenseDisambiguator:
    """Picks the contextually correct sign_id for ambiguous labels in a sentence."""

    def __init__(
        self,
        lexicon: dict[str, list[dict]],
        model_name: str = EMBEDDING_MODEL,
        cache_dir: Path = SENSE_EMBED_CACHE_DIR,
    ):
        self.lexicon = lexicon
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self._model = None
        # Per label: (list[sign_id], np.ndarray of L2-normalized sense embeddings).
        self._sense_embeddings: dict[str, tuple[list[str], "object"]] = {}

    # -- model / embedding cache ------------------------------------------------
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # lazy
            logger.info("Loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, texts: list[str]):
        import numpy as np  # lazy
        model = self._load_model()
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    def _cache_key(self) -> str:
        blob = json.dumps(self.lexicon, ensure_ascii=False, sort_keys=True).encode("utf-8")
        h = hashlib.sha256(blob).hexdigest()[:16]
        safe_model = re.sub(r"[^A-Za-z0-9]+", "_", self.model_name)
        return f"{safe_model}_{h}"

    def _ensure_sense_embeddings(self) -> None:
        if self._sense_embeddings:
            return
        import numpy as np  # lazy
        cache_path = self.cache_dir / f"{self._cache_key()}.npz"
        labels = sorted(self.lexicon)
        if cache_path.is_file():
            try:
                data = np.load(cache_path, allow_pickle=True)
                for label in labels:
                    self._sense_embeddings[label] = (
                        list(data[f"{label}__ids"]), data[f"{label}__vec"]
                    )
                logger.info("Loaded cached sense embeddings from %s", cache_path)
                return
            except Exception as exc:  # corrupt/stale cache -> recompute
                logger.warning("Ignoring unusable sense-embedding cache %s: %s", cache_path, exc)
                self._sense_embeddings.clear()

        to_save: dict[str, object] = {}
        for label in labels:
            senses = self.lexicon[label]
            ids = [s["sign_id"] for s in senses]
            vecs = self._embed([_sense_descriptor(s) for s in senses])
            self._sense_embeddings[label] = (ids, vecs)
            to_save[f"{label}__ids"] = np.array(ids, dtype=object)
            to_save[f"{label}__vec"] = vecs
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **to_save)
        logger.info("Cached sense embeddings to %s", cache_path)

    # -- public API -------------------------------------------------------------
    def labels_in(self, sentence: str) -> set[str]:
        """Ambiguous lexicon labels detected in the sentence (surface + light stem)."""
        present: set[str] = set()
        for tok in _tokenize(sentence):
            for form in _candidate_forms(tok):
                if form in self.lexicon:
                    present.add(form)
                    break
        return present

    def resolve(
        self, sentence: str, present_labels: Iterable[str] | None = None
    ) -> dict[str, str]:
        """Return {label: chosen_sign_id} for ambiguous labels in the sentence.

        `present_labels` optionally restricts work to labels the caller already
        knows are used (e.g. from the pose report); otherwise labels are
        auto-detected. Any label with no usable context match is omitted (the
        pipeline then keeps its default sense).
        """
        import numpy as np  # lazy
        labels = set(present_labels) if present_labels is not None else self.labels_in(sentence)
        labels &= set(self.lexicon)
        if not labels:
            return {}

        self._ensure_sense_embeddings()
        sent_vec = self._embed([sentence])[0]  # L2-normalized

        chosen: dict[str, str] = {}
        for label in labels:
            ids, vecs = self._sense_embeddings.get(label, (None, None))
            if not ids:
                continue
            sims = vecs @ sent_vec  # cosine (both normalized)
            chosen[label] = ids[int(np.argmax(sims))]
        return chosen


_SINGLETON: SenseDisambiguator | None = None
_SINGLETON_FAILED = False


def get_disambiguator(
    lexicon_path: Path = SENSE_LEXICON_PATH,
) -> SenseDisambiguator | None:
    """Process-wide disambiguator, or None if unavailable (missing lexicon or deps).

    Callers should treat None as "disambiguation disabled" and fall back to the
    pipeline's default sense selection — the feature is always optional.
    """
    global _SINGLETON, _SINGLETON_FAILED
    if _SINGLETON is not None or _SINGLETON_FAILED:
        return _SINGLETON
    try:
        if not Path(lexicon_path).is_file():
            logger.info("No sense lexicon at %s — disambiguation disabled.", lexicon_path)
            _SINGLETON_FAILED = True
            return None
        lexicon = load_sense_lexicon(lexicon_path)
        if not lexicon:
            logger.info("Sense lexicon is empty — disambiguation disabled.")
            _SINGLETON_FAILED = True
            return None
        _SINGLETON = SenseDisambiguator(lexicon)
        return _SINGLETON
    except Exception as exc:
        logger.warning("Could not initialize sense disambiguator: %s", exc)
        _SINGLETON_FAILED = True
        return None
