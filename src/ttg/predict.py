"""Single- and batch-sentence prediction for text → gloss models."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .config import (
    DEFAULT_GEMMA_BASE,
    DEFAULT_MBART_BASE,
    DEFAULT_MBART_SRC_LANG,
    DEFAULT_MBART_TGT_LANG,
    DEFAULT_NLLB_BASE,
    DEFAULT_NLLB_LANG,
    DEFAULT_QWEN_BASE,
    USER_PROMPT,
    require_nllb_checkpoint,
    resolve_model_dir,
    resolve_spm_model,
)

QWEN_USER_PROMPT = USER_PROMPT
GEMMA_USER_PROMPT = USER_PROMPT


def predict_sockeye(
    text: str,
    model_dir: Path | None = None,
    spm_model: Path | None = None,
    *,
    python: str = "python",
    beam_size: int = 5,
) -> str:
    import sentencepiece as spm

    model_dir = model_dir or resolve_model_dir("sockeye")
    spm_model = spm_model or resolve_spm_model()
    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "in.src"
        out = tmp_path / "out.trg"
        pieces = processor.encode(text, out_type=str)
        inp.write_text(" ".join(pieces) + "\n", encoding="utf-8")
        subprocess.run(
            [
                python,
                "-m",
                "sockeye.translate",
                "--models",
                str(model_dir),
                "--input",
                str(inp),
                "--output",
                str(out),
                "--beam-size",
                str(beam_size),
                "--use-cpu",
            ],
            check=True,
        )
        line = out.read_text(encoding="utf-8").strip()
        toks = line.split()
        return processor.decode(toks) if toks else ""


def predict_sockeye_batch(
    model_dir: Path,
    spm_model: Path,
    texts: list[str],
    *,
    beam_size: int,
    python: str,
) -> list[str]:
    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(spm_model))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        inp = tmp_path / "input.pieces.src"
        out_pieces = tmp_path / "output.pieces.trg"
        with inp.open("w", encoding="utf-8") as f:
            for text in texts:
                pieces = processor.encode(text, out_type=str)
                f.write(" ".join(pieces) + "\n")
        subprocess.run(
            [
                python,
                "-m",
                "sockeye.translate",
                "--models",
                str(model_dir),
                "--input",
                str(inp),
                "--output",
                str(out_pieces),
                "--beam-size",
                str(beam_size),
                "--batch-size",
                "16",
                "--use-cpu",
            ],
            check=True,
        )
        decoded: list[str] = []
        for line in out_pieces.read_text(encoding="utf-8").splitlines():
            toks = line.strip().split()
            decoded.append(processor.decode(toks) if toks else "")
        return decoded


def predict_mbart(
    text: str,
    model_dir: Path | None = None,
    *,
    base_model: str = DEFAULT_MBART_BASE,
    src_lang: str = DEFAULT_MBART_SRC_LANG,
    tgt_lang: str = DEFAULT_MBART_TGT_LANG,
    max_length: int = 64,
    num_beams: int = 5,
) -> str:
    return predict_mbart_batch(
        model_dir or resolve_model_dir("mbart"),
        [text],
        base_model=base_model,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        max_length=max_length,
        num_beams=num_beams,
    )[0]


def predict_mbart_batch(
    model_dir: Path,
    texts: list[str],
    *,
    base_model: str,
    src_lang: str,
    tgt_lang: str,
    max_length: int,
    num_beams: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Load from the fine-tuned checkpoint, not the base model: the checkpoint's
    # tokenizer may have added tokens (e.g. "[dct]") that the base model's
    # tokenizer doesn't know, and loading from base would silently re-fragment them.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer.src_lang = src_lang
    forced_bos = tokenizer.lang_code_to_id[tgt_lang]
    hyps: list[str] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=num_beams,
            )
        hyps.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return hyps


def predict_nllb(
    text: str,
    model_dir: Path | None = None,
    *,
    base_model: str = DEFAULT_NLLB_BASE,
    lang: str = DEFAULT_NLLB_LANG,
    max_length: int = 64,
    num_beams: int = 5,
) -> str:
    model_dir = model_dir or resolve_model_dir("nllb")
    require_nllb_checkpoint(model_dir)
    return predict_nllb_batch(
        model_dir,
        [text],
        base_model=base_model,
        lang=lang,
        max_length=max_length,
        num_beams=num_beams,
    )[0]


def predict_nllb_batch(
    model_dir: Path,
    texts: list[str],
    *,
    base_model: str,
    lang: str,
    max_length: int,
    num_beams: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # Load from the fine-tuned checkpoint, not the base model: the checkpoint's
    # tokenizer may have added tokens (e.g. "[dct]") that the base model's
    # tokenizer doesn't know, and loading from base would silently re-fragment them.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer.src_lang = lang
    forced_bos = tokenizer.convert_tokens_to_ids(lang)
    hyps: list[str] = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=max_length,
                num_beams=num_beams,
            )
        hyps.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return hyps


def predict_qwen(
    text: str,
    adapter_dir: Path | None = None,
    *,
    base_model: str = DEFAULT_QWEN_BASE,
    max_new_tokens: int = 64,
    prompt_template: str = QWEN_USER_PROMPT,
) -> str:
    return predict_qwen_batch(
        adapter_dir or resolve_model_dir("qwen"),
        [text],
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        prompt_template=prompt_template,
    )[0]


def _resize_to_adapter_checkpoint(model, adapter_dir: Path) -> None:
    """Resize `model`'s token embeddings to match the shape actually saved in
    the adapter checkpoint (LoRA training may have grown *or* shrunk the base
    model's embedding matrix relative to the freshly downloaded base model —
    e.g. base checkpoints often pad their embedding table beyond the
    tokenizer's real vocab size, so re-deriving the target size from the
    tokenizer instead of the checkpoint itself can pick the wrong number).
    """
    from safetensors import safe_open

    path = Path(adapter_dir) / "adapter_model.safetensors"
    if not path.is_file():
        return
    with safe_open(str(path), framework="pt") as f:
        for key in f.keys():
            if "embed_tokens" in key and key.endswith("weight"):
                target_rows = f.get_slice(key).get_shape()[0]
                if target_rows != model.get_input_embeddings().weight.shape[0]:
                    model.resize_token_embeddings(target_rows)
                return


def predict_qwen_batch(
    adapter_dir: Path,
    texts: list[str],
    *,
    base_model: str,
    max_new_tokens: int,
    prompt_template: str = QWEN_USER_PROMPT,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load from the fine-tuned adapter dir, not the base model: the checkpoint's
    # tokenizer may have added tokens (e.g. "[dct]") that the base model's
    # tokenizer doesn't know, and loading from base would silently re-fragment them.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    _resize_to_adapter_checkpoint(model, adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    hyps: list[str] = []
    for text in texts:
        messages = [{"role": "user", "content": prompt_template.format(text=text)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        hyps.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return hyps


def predict_gemma(
    text: str,
    adapter_dir: Path | None = None,
    *,
    base_model: str = DEFAULT_GEMMA_BASE,
    max_new_tokens: int = 64,
    prompt_template: str = GEMMA_USER_PROMPT,
) -> str:
    return predict_gemma_batch(
        adapter_dir or resolve_model_dir("gemma"),
        [text],
        base_model=base_model,
        max_new_tokens=max_new_tokens,
        prompt_template=prompt_template,
    )[0]


def predict_gemma_batch(
    adapter_dir: Path,
    texts: list[str],
    *,
    base_model: str,
    max_new_tokens: int,
    prompt_template: str = GEMMA_USER_PROMPT,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load from the fine-tuned adapter dir, not the base model: the checkpoint's
    # tokenizer may have added tokens (e.g. "[dct]") that the base model's
    # tokenizer doesn't know, and loading from base would silently re-fragment them.
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    _resize_to_adapter_checkpoint(model, adapter_dir)
    model = PeftModel.from_pretrained(model, adapter_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    hyps: list[str] = []
    for text in texts:
        messages = [{"role": "user", "content": prompt_template.format(text=text)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        hyps.append(tokenizer.decode(generated, skip_special_tokens=True).strip())
    return hyps
