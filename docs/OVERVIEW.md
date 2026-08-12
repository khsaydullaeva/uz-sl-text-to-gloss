# Overview — Uzbek Text → Gloss

## What we build

Input: a written Uzbek sentence.  
Output: a **gloss** — a sequence of sign lemmas the signer would produce
(lowercased, space-separated).

Example:

```
text:  Men uyda o'tiribman.
gloss: men uy o'tirmoq
```

This is the first stage of a larger text→sign pipeline (gloss → pose → video).

## Models

| Model | Type | Notes |
|---|---|---|
| **NLLB-200** (distilled 600M) | seq2seq fine-tune | usually best BLEU / gloss-F1 |
| **mBART-50** | seq2seq fine-tune | strong multilingual baseline |
| **Qwen2.5-1.5B** | LoRA instruct | chat-style prompting |
| **Gemma-2-2B** | LoRA instruct | needs HF license acceptance |
| Sockeye | classical NMT | optional / legacy |

## Data

~1.2k corrected Uzbek↔gloss sentence pairs under `data/raw/`, split 80/10/10
into `data/mbart/` (and mirrored `data/splits/`).

## Pipeline

```
01 explore → 02 prepare splits → 03–06 train models → 07 evaluate
```

Training is notebook-only (same convention as SighnTT). Shared code lives in
`src/ttg/`.
