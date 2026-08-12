"""Uzbek text → sign-language gloss (NLLB, mBART, Qwen, Gemma, Sockeye)."""

from .config import (
    ARTIFACTS,
    CHECKPOINTS_DIR,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    resolve_model_dir,
)
from .config import (
    DEFAULT_GEMMA_BASE,
    DEFAULT_MBART_BASE,
    DEFAULT_MBART_SRC_LANG,
    DEFAULT_MBART_TGT_LANG,
    DEFAULT_NLLB_BASE,
    DEFAULT_NLLB_LANG,
    DEFAULT_QWEN_BASE,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "ARTIFACTS",
    "CHECKPOINTS_DIR",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "resolve_model_dir",
    "predict_gloss",
]


def predict_gloss(
    text: str,
    model: str = "nllb",
    *,
    max_length: int = 64,
    num_beams: int = 5,
    max_new_tokens: int = 64,
) -> str:
    """Return gloss string for one Uzbek sentence using the default checkpoint."""
    from .predict import predict_gemma, predict_mbart, predict_nllb, predict_qwen

    if model == "nllb":
        return predict_nllb(
            text,
            resolve_model_dir("nllb"),
            base_model=DEFAULT_NLLB_BASE,
            lang=DEFAULT_NLLB_LANG,
            max_length=max_length,
            num_beams=num_beams,
        )
    if model == "qwen":
        return predict_qwen(
            text,
            resolve_model_dir("qwen"),
            base_model=DEFAULT_QWEN_BASE,
            max_new_tokens=max_new_tokens,
        )
    if model == "gemma":
        return predict_gemma(
            text,
            resolve_model_dir("gemma"),
            base_model=DEFAULT_GEMMA_BASE,
            max_new_tokens=max_new_tokens,
        )
    if model == "mbart":
        return predict_mbart(
            text,
            resolve_model_dir("mbart"),
            base_model=DEFAULT_MBART_BASE,
            src_lang=DEFAULT_MBART_SRC_LANG,
            tgt_lang=DEFAULT_MBART_TGT_LANG,
            max_length=max_length,
            num_beams=num_beams,
        )
    raise ValueError(f"unsupported model: {model}")
