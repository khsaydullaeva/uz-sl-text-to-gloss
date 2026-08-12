"""Central paths and constants for Uzbek text → gloss."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS / "ckpts"
OUTPUTS_DIR = ARTIFACTS
MODELS_DIR = PROJECT_ROOT / "models"
CORPUS_DIR = ARTIFACTS / "corpus"

# Word-sense disambiguation (gloss token → correct sign for homograph labels).
# The curated sense lexicon maps an Uzbek label to its candidate senses (each a
# sign_id + meaning); the embedding model scores sentence context against each
# sense. Sense embeddings are cached under ARTIFACTS keyed by (lexicon, model).
SENSE_LEXICON_PATH = DATA_DIR / "sense_lexicon.json"
EMBEDDING_MODEL = os.environ.get("TTG_EMBEDDING_MODEL", "sentence-transformers/LaBSE")
SENSE_EMBED_CACHE_DIR = ARTIFACTS / "sense_embeddings"

# Optional external tree override (legacy layout).
TEXT_GLOSS_NMT_DIR = Path(
    os.environ.get("TEXT_GLOSS_NMT_DIR", str(ARTIFACTS / "ckpts"))
)

SEED = 42

# Default Hugging Face bases / language codes
DEFAULT_MBART_BASE = "facebook/mbart-large-50-many-to-many-mmt"
# mBART-50 has no Uzbek language code. "uz_AR" is not a real code in this
# tokenizer's vocab and silently resolves to <unk>, giving every example the
# same non-informative language-prefix token. "tr_TR" (Turkish) is a real,
# valid mBART-50 code and at least a Latin-script Turkic-family proxy.
DEFAULT_MBART_SRC_LANG = "tr_TR"
DEFAULT_MBART_TGT_LANG = "en_XX"
DEFAULT_NLLB_BASE = "facebook/nllb-200-distilled-600M"
DEFAULT_NLLB_LANG = "uzn_Latn"
DEFAULT_QWEN_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_GEMMA_BASE = "google/gemma-2-2b-it"

USER_PROMPT = (
    "Uzbek jumlasini imo-ishora glosslariga aylantiring. "
    "Faqat gloss so'zlarini yozing (kichik harf, bo'shliq bilan ajratilgan). "
    "Inglizcha ishlatmang.\n\nJumla: {text}"
)

# Reverse direction: gloss -> text. Uses the same {text} placeholder as
# USER_PROMPT for interface compatibility with to_chat_messages() — after
# ttg.data.swap_direction(), split.texts holds gloss strings, not sentences.
USER_PROMPT_GLOSS_TO_TEXT = (
    "Quyidagi imo-ishora gloss ketma-ketligini ravon o'zbek tiliga aylantiring. "
    "Faqat tabiiy o'zbekcha jumlani yozing.\n\nGloss: {text}"
)

# Full NLLB-600M fp32 ~2.4 GB; fp16 ~1.2 GB. Reject obvious truncations (<100 MB).
_NLLB_MIN_BYTES = 100_000_000


def _has_hf_weights(model_dir: Path) -> bool:
    if (model_dir / "model.safetensors").is_file():
        return True
    if (model_dir / "pytorch_model.bin").is_file():
        return True
    return bool(list(model_dir.glob("model-*.safetensors")))


def _is_nllb_checkpoint(model_dir: Path) -> bool:
    if not (model_dir / "config.json").is_file():
        return False
    weights = model_dir / "model.safetensors"
    if weights.is_file():
        return weights.stat().st_size >= _NLLB_MIN_BYTES
    if (model_dir / "pytorch_model.bin").is_file():
        return True
    shards = list(model_dir.glob("model-*.safetensors"))
    return bool(shards) and sum(p.stat().st_size for p in shards) >= _NLLB_MIN_BYTES


def _is_peft_adapter(model_dir: Path) -> bool:
    return (model_dir / "adapter_config.json").is_file()


def _model_candidates(name: str) -> list[Path]:
    nmt = TEXT_GLOSS_NMT_DIR
    if name == "sockeye":
        return [
            CHECKPOINTS_DIR / "sockeye" / "models" / "text_to_gloss",
            nmt / "sockeye" / "models" / "text_to_gloss",
            MODELS_DIR / "sockeye" / "models" / "text_to_gloss",
        ]
    return [
        CHECKPOINTS_DIR / name / "best",
        nmt / name / "best",
        nmt / name / "checkpoints" / "best",
        MODELS_DIR / name,
        CHECKPOINTS_DIR / name,
    ]


def resolve_spm_model() -> Path:
    for path in (
        CHECKPOINTS_DIR / "sockeye" / "spm.model",
        MODELS_DIR / "sockeye" / "spm.model",
    ):
        if path.is_file():
            return path
    return CHECKPOINTS_DIR / "sockeye" / "spm.model"


def resolve_model_dir(name: str) -> Path:
    """Return the first usable checkpoint directory for nllb, mbart, qwen, gemma, or sockeye."""
    for path in _model_candidates(name):
        if not path.is_dir():
            continue
        if name in ("qwen", "gemma") and _is_peft_adapter(path):
            return path
        if name in ("nllb", "mbart") and _is_nllb_checkpoint(path):
            return path
        if name == "sockeye" and (
            (path / "params.best").exists() or (path / "config").is_file()
        ):
            return path
        if name not in ("nllb", "mbart", "qwen", "gemma", "sockeye") and _has_hf_weights(path):
            return path
    for path in _model_candidates(name):
        if path.is_dir() and (_has_hf_weights(path) or _is_peft_adapter(path)):
            return path
    return _model_candidates(name)[0]


def require_nllb_checkpoint(model_dir: Path) -> Path:
    """Fail fast when model.safetensors is missing or truncated."""
    weights = model_dir / "model.safetensors"
    if weights.is_file():
        size = weights.stat().st_size
        if size < _NLLB_MIN_BYTES:
            raise ValueError(
                f"NLLB checkpoint is incomplete ({size / 1e6:.1f} MB at {weights}; "
                f"expected ≥100 MB, full fine-tune ~1200–2400 MB).\n"
                f"Expected layout:\n"
                f"  {CHECKPOINTS_DIR / 'nllb' / 'best'}/model.safetensors"
            )
        return weights
    if (model_dir / "pytorch_model.bin").is_file() or list(model_dir.glob("model-*.safetensors")):
        return model_dir / "pytorch_model.bin"
    raise FileNotFoundError(
        f"NLLB checkpoint not found in {model_dir}\n"
        f"Expected Hugging Face files (model.safetensors + config.json) under:\n"
        f"  {CHECKPOINTS_DIR / 'nllb' / 'best'}"
    )
