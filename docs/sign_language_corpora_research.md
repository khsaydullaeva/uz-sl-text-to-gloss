# How other sign-language projects collected sentence↔gloss data

Survey of the main public corpora used for sign language recognition/translation,
focused on **how they got sentence-level gloss annotations** and **how much they got**.
Useful as reference points for scaling/annotating our Uzbek text→gloss data.

## Summary table

| Language | Corpus | Source of video | Who glossed it | Size |
|---|---|---|---|---|
| German (DGS) | RWTH-PHOENIX-Weather 2014T | TV weather forecasts, sign interpreter overlay | Deaf/hard-of-hearing native DGS annotators, manual, sentence-level | 8,257 sentences (7,096 train / 519 dev / 642 test), 9 signers, ~11h |
| Chinese (CSL) | CSL-Daily | Studio recordings, scripted daily-life sentences | Manual gloss annotation by fluent signers | 20,654 (video, gloss, text) triplets, 10 signers, gloss vocab 2,000 |
| Hong Kong (HKSL) | TVB-HKSL-News | TV news broadcast, sign interpreter | Manual, sentence-aligned with Chinese subtitles | 16.07h, 2 signers, gloss vocab 6,515 |
| Russian (RSL) | Academic RSL corpus (Burkova 2015, Prozorova, Kimmelman) | Elicited/native narratives, filmed for linguistic research | Manual, ELAN, gloss tiers per hand + Russian translation | Linguistic corpus, not ML-scale (hours, not tens of thousands of sentences) |
| Russian (RSL) | Slovo | Crowdsourced recordings (194 signers) | N/A — isolated gloss/word labels, not sentences | 20,000 clips / 1,000 isolated gloss classes, ~20h |
| English (ASL) | How2Sign | Studio recordings of instructional ("how-to") monologues | No gloss — English transcript aligned to signing, sentence-level | 80h, 35,000+ sentence-aligned segments |
| English (ASL) | YouTube-ASL | Scraped from YouTube (public ASL-tagged videos + captions) | No gloss — captions filtered/verified by Deaf annotators | 11,093 videos, 984h video, 610,193 English caption segments |
| English (BSL) | BOBSL (BBC-Oxford) | TV broadcasts with in-vision BSL interpreter | Mostly automatic (mouthing spotting + subtitle alignment + dictionary matching), sparse manual gloss | 1,962 episodes, 1,467h, ~1.2M English sentences, sign vocab 2,000+ |

## Details by language

### German — RWTH-PHOENIX-Weather 2014T
Recorded from the German public broadcaster PHOENIX's daily weather forecast,
which is simultaneously interpreted into German Sign Language (DGS), over
three years of broadcasts, 9 different interpreters. Two parallel annotation
streams:
- **Gloss**: manually transcribed at the sentence level by deaf/hard-of-hearing
  native DGS speakers.
- **Spoken German**: semi-automatically transcribed from the broadcast audio
  using the RASR speech recognizer, then a second translation pass produced
  from the glosses to capture natural translation variability.

This is the most-cited "sentence→gloss→sentence" parallel corpus in the field
precisely because the domain (weather) keeps vocabulary bounded while still
being naturally occurring, continuous signing rather than isolated signs.

### Chinese — CSL-Daily
Unlike PHOENIX, CSL-Daily is not broadcast footage — it's studio-recorded with
10 signers performing scripted sentences about daily life (family, healthcare,
school, banking, shopping, social interaction). Each sample is a
(video, gloss, text) triplet with a bounded gloss vocabulary (~2,000) and
Chinese text vocabulary (~2,343), split 18,401 / 1,077 / 1,176 for
train/dev/test. The scripted-studio approach trades broadcast scale for
control over sentence/topic coverage and consistent signer quality.

### Hong Kong — TVB-HKSL-News
Same broadcast-interpreter approach as PHOENIX but for Cantonese/HKSL: TV news
programs over 7 months, 2 signers, 16.07 hours, manually annotated with a
large gloss vocabulary (6,515) since news covers far more topics than weather.
Aligned to Chinese subtitles at the sentence level.

### Russian — two very different corpora
- **Linguistic RSL corpus** (Burkova 2015; earlier work by Prozorova 2009 and
  Kimmelman 2009–2014): built for descriptive/theoretical linguistics, not ML.
  Filmed native/elicited signing, annotated in ELAN with separate gloss tiers
  for each hand plus a Russian translation tier, plus mouthing annotation.
  Small by ML standards (hours of footage, not thousands of sentences) but
  linguistically rich.
- **Slovo** (2023): crowdsourced via 194 signers recording themselves, aimed at
  isolated sign/gesture recognition rather than continuous sentences — 20,000
  clips across 1,000 gloss classes, ~20 hours, 75/25 train/test split. No
  sentence-level gloss sequences; each clip is a single label.

### English — three different strategies, one clear trend away from gloss
- **How2Sign**: studio-recorded Deaf presenters performing "how-to" instructional
  monologues (cooking, crafts, etc.), 80 hours, 35,000+ sentences aligned to
  English transcripts. No gloss annotation at all — translation is done
  directly video→text.
- **YouTube-ASL**: scraped YouTube for videos tagged as ASL-related via
  Knowledge-Graph entity tags, then had **native Deaf annotators filter** for
  video quality and caption/sign alignment (no re-annotation — reuses existing
  YouTube captions). Ends up an order of magnitude larger than prior ASL sets:
  984 hours, 610,193 caption segments. Also gloss-free.
- **BOBSL** (British Sign Language): BBC broadcast footage with a BSL
  interpreter, aligned to the broadcast's own English subtitles — by far the
  largest (1,467h, ~1.2M sentences), but gloss labels are produced **mostly
  automatically**: mouthing keyword-spotting, dictionary matching against a
  sign-spotting model, and subtitle-signing alignment, rather than manual
  gloss transcription. A later "densification" pass increased confident
  automatic annotations from 670K to 5M spottings.

## Takeaways relevant to this project

1. **Two collection archetypes dominate**: (a) broadcast/TV footage with an
   existing sign interpreter (PHOENIX, TVB-HKSL-News, BOBSL) — cheap to source
   at scale but requires post-hoc alignment/annotation work; (b) studio-recorded
   scripted sentences by hired signers (CSL-Daily) — more control over topic/
   vocabulary coverage and consistent gloss quality, but far smaller scale and
   costlier per sentence.
2. **Manual sentence-level gloss annotation only happens at modest scale**
   (PHOENIX: ~8K sentences; CSL-Daily: ~20K; TVB-HKSL-News: 16h) even for
   well-funded research groups — it's the bottleneck, which is why newer large
   English corpora (How2Sign, YouTube-ASL, BOBSL) drop manual gloss entirely
   and go straight to text-aligned or automatically-spotted labels once they
   aim for >100h/>500K-sentence scale.
3. Where linguistically rigorous gloss really matters (RSL academic corpus,
   PHOENIX), annotation is done by **native/Deaf annotators**, often with
   ELAN, and includes hand-specific tiers — worth matching if gloss quality
   for Uzbek Sign Language is the priority over raw scale.

## Sources

- [RWTH-PHOENIX-Weather 2014T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- [Extensions of the Sign Language Recognition and Translation Corpus (LREC 2014)](https://aclanthology.org/L14-1472/)
- [A Simple Multi-Modality Transfer Learning Baseline for Sign Language Translation (CSL-Daily)](https://arxiv.org/pdf/2203.04287)
- [A Hong Kong Sign Language Corpus Collected from Sign-interpreted TV News](https://arxiv.org/abs/2405.00980)
- [A Chinese Continuous Sign Language Dataset Based on Complex Environments (CE-CSL)](https://arxiv.org/abs/2409.11960)
- [Slovo: Russian Sign Language Dataset](https://arxiv.org/pdf/2305.14527)
- [Russian Sign Language: History, Grammar and Research (LT4All 2019)](https://lt4all.elra.info/proceedings/lt4all2019/pdf/2019.lt4all-1.18.pdf)
- [New Insights Into Mouthings: Evidence From a Corpus-Based Study of Russian Sign Language](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2021.779958/full)
- [How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language](https://arxiv.org/pdf/2008.08143)
- [YouTube-ASL: A Large-Scale, Open-Domain American Sign Language-English Parallel Corpus](https://arxiv.org/pdf/2306.15162)
- [BBC-Oxford British Sign Language Dataset (BOBSL)](https://arxiv.org/pdf/2111.03635)
- [Automatic dense annotation of large-vocabulary sign language videos](https://arxiv.org/abs/2208.02802)
