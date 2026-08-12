"""Evaluation metrics for text → gloss."""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"\S+")


def tokenize_gloss(line: str) -> list[str]:
    return TOKEN_RE.findall(line.strip().casefold())


def gloss_token_f1(hyp: str, ref: str) -> float:
    hyp_toks = tokenize_gloss(hyp)
    ref_toks = tokenize_gloss(ref)
    if not hyp_toks and not ref_toks:
        return 1.0
    if not hyp_toks or not ref_toks:
        return 0.0
    ref_counts: dict[str, int] = {}
    hyp_counts: dict[str, int] = {}
    for t in ref_toks:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    for t in hyp_toks:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1
    common = sum(min(c, ref_counts.get(tok, 0)) for tok, c in hyp_counts.items())
    prec = common / len(hyp_toks)
    rec = common / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def corpus_bleu(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)


def score_predictions(name: str, hyps: list[str], refs: list[str]) -> dict:
    f1s = [gloss_token_f1(h, r) for h, r in zip(hyps, refs)]
    exact = sum(1 for h, r in zip(hyps, refs) if tokenize_gloss(h) == tokenize_gloss(r))
    return {
        "model": name,
        "num_examples": len(refs),
        "bleu": corpus_bleu(hyps, refs),
        "chrf": corpus_chrf(hyps, refs),
        "gloss_token_f1": sum(f1s) / max(1, len(f1s)),
        "exact_match": exact / max(1, len(refs)),
    }
