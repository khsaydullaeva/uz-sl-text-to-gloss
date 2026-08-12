"""
Uzbek text-to-gloss for sign-language-processing pipeline.

Same API as spoken_to_signed.text_to_gloss.* modules:
  text_to_gloss(text, language, ...) -> list[Gloss]

Maps tokens to Uzbek ISLR lexicon glosses (with case/verb normalization).
Unknown tokens are lemmatized/stemmed (e.g. optimallashtiriladi -> optimallashtirmoq).
"""
from __future__ import annotations

from functools import lru_cache

from spoken_to_signed.text_to_gloss.types import Gloss, GlossItem

from scripts.uzsl_lexicon import GlossLexicon, _split_sentence, surface_to_lemma_gloss


@lru_cache(maxsize=1)
def _lexicon() -> GlossLexicon:
    return GlossLexicon()


def _parsed_word_gloss(token: str) -> str:
    """Lemma/stem gloss for tokens with no sign in the lexicon."""
    return surface_to_lemma_gloss(token)


def text_to_gloss(
    text: str,
    language: str,
    *,
    topic_hint: str = "",
    fingerspell_unknown: bool = False,
    **unused_kwargs,
) -> list[Gloss]:
    """Return one gloss sequence (list with a single sentence Gloss)."""
    lex = _lexicon()
    items: list[GlossItem] = []
    prev_topic = ""
    for tok in _split_sentence(text):
        entry, _method = lex.resolve_with_method(
            tok, topic_hint=topic_hint, prev_topic=prev_topic
        )
        if entry is not None:
            if entry.topic == "Alifbo" and fingerspell_unknown:
                for fe in lex.fingerspell(tok):
                    items.append(GlossItem(word=tok.lower(), gloss=fe.gloss.casefold()))
                    prev_topic = fe.topic
                continue
            items.append(GlossItem(word=tok.lower(), gloss=entry.gloss.casefold()))
            prev_topic = entry.topic
            continue
        if fingerspell_unknown:
            try:
                for fe in lex.fingerspell(tok):
                    items.append(GlossItem(word=tok.lower(), gloss=fe.gloss.casefold()))
                    prev_topic = fe.topic
                continue
            except KeyError:
                pass
        items.append(GlossItem(word=tok.lower(), gloss=_parsed_word_gloss(tok)))
    return [items]
