#!/usr/bin/env python3
"""Serve a local dashboard for text-to-gloss model comparison results."""
from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader: KEY=value / KEY = value per line, no deps.

    Existing os.environ values win (real env vars override the file), matching
    how tools like python-dotenv behave by default.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(_ROOT / ".env")
_load_dotenv(Path(".env"))

from ttg.config import (
    OUTPUTS_DIR,
    PROJECT_ROOT as REPO_ROOT,
    PROJECT_ROOT as PKG_ROOT,
    resolve_model_dir,
    resolve_spm_model,
    DEFAULT_GEMMA_BASE,
    DEFAULT_MBART_BASE,
    DEFAULT_MBART_SRC_LANG,
    DEFAULT_MBART_TGT_LANG,
    DEFAULT_NLLB_BASE,
    DEFAULT_NLLB_LANG,
    DEFAULT_QWEN_BASE,
    USER_PROMPT_GLOSS_TO_TEXT,
)
from ttg.predict import predict_gemma, predict_mbart, predict_nllb, predict_qwen, predict_sockeye


MODEL_LABELS = {
    "nllb": "NLLB-200 (fine-tuned)",
    "mbart": "mBART-50 (fine-tuned)",
    "gemma": "Gemma 2 (LoRA)",
    "qwen": "Qwen2.5 (LoRA)",
    "sockeye": "Sockeye",
}

# gloss -> text is not trained for sockeye (no checkpoint exists at all,
# either direction); the other four all have a "*_g2t" checkpoint (see the
# DIRECTION toggle in notebooks 03/04/05/06).
GLOSS_TO_TEXT_MODEL_LABELS = {
    "nllb": MODEL_LABELS["nllb"],
    "mbart": MODEL_LABELS["mbart"],
    "gemma": MODEL_LABELS["gemma"],
    "qwen": MODEL_LABELS["qwen"],
}

POSE_OUTPUT_DIR = OUTPUTS_DIR / "pose_ui"
POSE_CONDA_ENV = "uzsl-pose"


def _default_pose_repo_root() -> Path:
    """Where record/tools/text_to_pose.py lives.

    The pose/record pipeline is not part of this repo — it lives in the sibling
    ``uzbek-sign-language`` checkout. Override with $POSE_REPO_ROOT if that repo
    is cloned somewhere else.
    """
    env_override = os.environ.get("POSE_REPO_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()
    sibling = REPO_ROOT.parent / "uzbek-sign-language"
    if (sibling / "record" / "tools" / "text_to_pose.py").is_file():
        return sibling
    return REPO_ROOT


def _default_pose_python() -> Path:
    """Python for the pose pipeline (mediapipe / pose_format / vidgear / opencv).

    Prefers the ``uzsl-pose`` conda env (see README) over a venv, since a conda
    env's ``bin/python`` is a self-contained interpreter that works the same way
    a venv python does when invoked by absolute path.
    """
    env_override = os.environ.get("POSE_PYTHON")
    if env_override:
        return Path(env_override).expanduser().resolve()
    for conda_root in ("~/miniforge3", "~/miniconda3", "~/anaconda3"):
        candidate = Path(conda_root).expanduser() / "envs" / POSE_CONDA_ENV / "bin" / "python"
        if candidate.is_file():
            return candidate
    return DEFAULT_POSE_REPO_ROOT / "pose" / "venv" / "bin" / "python"


DEFAULT_POSE_REPO_ROOT = _default_pose_repo_root()
DEFAULT_POSE_PYTHON = _default_pose_python()
DEFAULT_UZSL_DATA = Path(os.environ.get("UZSL_DATA_DIR", str(REPO_ROOT / "uzsl_data")))

# Context-based word-sense disambiguation of homograph gloss tokens. On by
# default; silently a no-op per request when the lexicon or embedding model is
# unavailable (compute_sense_overrides returns {} → pipeline keeps default sense).
DISAMBIGUATE_DEFAULT = os.environ.get("TTG_DISAMBIGUATE", "1").lower() not in (
    "0", "false", "no", "off",
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        return {"test_size": 0, "models": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def load_predictions(predictions_dir: Path) -> dict[str, list[dict]]:
    predictions: dict[str, list[dict]] = {}
    if not predictions_dir.is_dir():
        return predictions
    for path in sorted(predictions_dir.glob("*_test.jsonl")):
        model = path.name.removesuffix("_test.jsonl")
        rows: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        predictions[model] = rows
    return predictions


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_stem(text: str) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return stem[:80] or "gloss"


def python_launcher(path: Path) -> Path:
    """Return an absolute venv python path without resolving its symlink.

    Resolving ``pose/venv/bin/python`` can turn into ``/usr/bin/python3.11`` and
    drop the virtualenv, which then imports broken system site-packages.
    """
    launcher = path.expanduser()
    if not launcher.is_absolute():
        launcher = (Path.cwd() / launcher).absolute()
    return launcher


def resolve_uzsl_data(data_dir: Path | None = None) -> Path:
    root = data_dir if data_dir is not None else DEFAULT_UZSL_DATA
    return root.expanduser().resolve()


def require_sign_dictionary(uzsl_data: Path) -> None:
    signs_csv = uzsl_data / "metadata" / "signs.csv"
    samples_csv = uzsl_data / "metadata" / "samples.csv"
    if signs_csv.is_file() and samples_csv.is_file():
        return
    missing = [str(p) for p in (signs_csv, samples_csv) if not p.is_file()]
    raise RuntimeError(
        "Sign dictionary dataset is missing.\n"
        f"Expected files under {uzsl_data}:\n"
        + "\n".join(f"  - {path}" for path in missing)
        + "\n\nThe full UzSL dataset is not stored in this git repo.\n"
        "Point the UI at your local dataset root, then restart:\n"
        "  export UZSL_DATA_DIR=/path/to/uzsl_data\n"
        "  conda activate uzsl-pose && python ui.py --open "
        "--uzsl-data-dir /path/to/uzsl_data"
    )


def compute_sense_overrides(text: str) -> dict[str, str]:
    """Context-based word-sense choices {label: sign_id} for a sentence.

    Runs in this (uzsl-ttg) process, which has the embedding model. Returns {}
    when disambiguation is unavailable (no lexicon / missing deps) or nothing is
    ambiguous — the pose pipeline then keeps its default (most-attested) sense.
    """
    try:
        from ttg.disambiguate import get_disambiguator
    except Exception as exc:  # import guard — feature is always optional
        print(f"[disambiguate] unavailable: {exc}", file=sys.stderr)
        return {}
    disambiguator = get_disambiguator()
    if disambiguator is None:
        return {}
    try:
        return disambiguator.resolve(text)
    except Exception as exc:
        print(f"[disambiguate] failed, using default senses: {exc}", file=sys.stderr)
        return {}


_SENSE_RU_MAP: dict[str, str] | None = None


def _sense_ru_map() -> dict[str, str]:
    """sign_id → Russian label, from the sense lexicon (for display only)."""
    global _SENSE_RU_MAP
    if _SENSE_RU_MAP is None:
        _SENSE_RU_MAP = {}
        try:
            from ttg.config import SENSE_LEXICON_PATH
            data = json.loads(Path(SENSE_LEXICON_PATH).read_text(encoding="utf-8"))
            for senses in data.values():
                for s in senses:
                    if s.get("sign_id"):
                        _SENSE_RU_MAP[s["sign_id"]] = (s.get("label_ru") or "").strip()
        except Exception:
            _SENSE_RU_MAP = {}
    return _SENSE_RU_MAP


def format_resolutions(resolutions: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in resolutions:
        word = str(item.get("word", "")).strip().upper()
        method = str(item.get("method", "oov"))
        form = item.get("form")
        pose_form = str(form).strip().upper() if form else word
        if method == "fingerspell":
            display = f"[DCT]{word}[DCT]"
        elif method == "sign":
            display = pose_form if pose_form != word else word
        elif method == "closest":
            display = f"~{pose_form}~" if pose_form != word else word
        else:
            display = word
        # Homograph resolved by context: annotate with the chosen sense so the
        # UI shows e.g. RASM→фото rather than silently picking a sense.
        disambiguated = bool(item.get("disambiguated"))
        sense_sign_id = item.get("sense_sign_id")
        sense_ru = _sense_ru_map().get(sense_sign_id, "") if sense_sign_id else ""
        if disambiguated and sense_ru:
            display = f"{display}→{sense_ru}"
        rows.append(
            {
                "word": word,
                "method": method,
                "display": display,
                "pose_form": pose_form,
                "disambiguated": disambiguated,
                "sense_sign_id": sense_sign_id,
                "sense_ru": sense_ru,
            }
        )
    return rows


def build_annotated_gloss_line(resolutions: list[dict]) -> str:
    return " ".join(row["display"] for row in format_resolutions(resolutions))


def predict_sentence(text: str, model: str, direction: str = "text2gloss") -> dict:
    text = text.strip()
    if not text:
        raise ValueError("Enter some input text first.")

    g2t = direction == "gloss2text"
    started = time.perf_counter()
    if g2t:
        if model == "nllb":
            output = predict_nllb(
                text,
                resolve_model_dir("nllb_g2t"),
                base_model=DEFAULT_NLLB_BASE,
                lang=DEFAULT_NLLB_LANG,
                max_length=64,
                num_beams=5,
            )
        elif model == "mbart":
            output = predict_mbart(
                text,
                resolve_model_dir("mbart_g2t"),
                base_model=DEFAULT_MBART_BASE,
                # gloss2text swaps which content flows through each tag, so
                # the tags swap too (see notebooks/03_train_mbart.ipynb).
                src_lang=DEFAULT_MBART_TGT_LANG,
                tgt_lang=DEFAULT_MBART_SRC_LANG,
                max_length=64,
                num_beams=5,
            )
        elif model == "qwen":
            output = predict_qwen(
                text,
                resolve_model_dir("qwen_g2t"),
                base_model=DEFAULT_QWEN_BASE,
                max_new_tokens=64,
                prompt_template=USER_PROMPT_GLOSS_TO_TEXT,
            )
        elif model == "gemma":
            output = predict_gemma(
                text,
                resolve_model_dir("gemma_g2t"),
                base_model=DEFAULT_GEMMA_BASE,
                max_new_tokens=64,
                prompt_template=USER_PROMPT_GLOSS_TO_TEXT,
            )
        else:
            raise ValueError(f"gloss→text is not available for {model}")
    elif model == "nllb":
        output = predict_nllb(
            text,
            resolve_model_dir("nllb"),
            base_model=DEFAULT_NLLB_BASE,
            lang=DEFAULT_NLLB_LANG,
            max_length=64,
            num_beams=5,
        )
    elif model == "mbart":
        output = predict_mbart(
            text,
            resolve_model_dir("mbart"),
            base_model=DEFAULT_MBART_BASE,
            src_lang=DEFAULT_MBART_SRC_LANG,
            tgt_lang=DEFAULT_MBART_TGT_LANG,
            max_length=64,
            num_beams=5,
        )
    elif model == "gemma":
        output = predict_gemma(
            text,
            resolve_model_dir("gemma"),
            base_model=DEFAULT_GEMMA_BASE,
            max_new_tokens=64,
        )
    elif model == "qwen":
        output = predict_qwen(
            text,
            resolve_model_dir("qwen"),
            base_model=DEFAULT_QWEN_BASE,
            max_new_tokens=64,
        )
    elif model == "sockeye":
        output = predict_sockeye(
            text,
            resolve_model_dir("sockeye"),
            resolve_spm_model(),
            python=str(PKG_ROOT / ".venv-sockeye" / "bin" / "python"),
            beam_size=5,
        )
    else:
        raise ValueError(f"Unsupported model: {model}")

    # Gloss is displayed upper-case by convention; fluent generated text
    # (gloss2text) keeps its natural casing.
    display = output if g2t else output.upper()
    return {
        "text": text,
        "model": model,
        "direction": direction,
        "model_label": MODEL_LABELS.get(model, model),
        "gloss": display,
        "tokens": display.split(),
        "latency_seconds": round(time.perf_counter() - started, 2),
    }


def generate_pose_video(
    text: str,
    gloss: str,
    pose_python: Path,
    output_dir: Path,
    uzsl_data: Path,
    pose_repo_root: Path = DEFAULT_POSE_REPO_ROOT,
    disambiguate: bool = DISAMBIGUATE_DEFAULT,
) -> dict:
    text = text.strip()
    gloss = gloss.strip()
    if not gloss:
        raise ValueError("Generate a gloss first.")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{int(time.time())}_{safe_stem(gloss)}"
    pose_path = output_dir / f"{stem}.pose"
    video_path = output_dir / f"{stem}_skeleton.mp4"
    report_path = output_dir / f"{stem}_report.json"
    if not pose_python.is_file():
        raise RuntimeError(
            f"Pose Python environment ({pose_python}) is not set up. Run:\n"
            f"  conda create -n {POSE_CONDA_ENV} python=3.11 -y\n"
            f"  conda run -n {POSE_CONDA_ENV} pip install -e "
            f"\"{pose_repo_root}/pose/src/python[mediapipe]\"\n"
            f"  conda run -n {POSE_CONDA_ENV} pip install vidgear opencv-python\n\n"
            "Then restart this UI (it auto-detects the conda env), or point at it explicitly:\n"
            f"  python ui.py --open --pose-python ~/miniforge3/envs/{POSE_CONDA_ENV}/bin/python"
        )

    text_to_pose_script = pose_repo_root / "record" / "tools" / "text_to_pose.py"
    if not text_to_pose_script.is_file():
        raise RuntimeError(
            f"text_to_pose.py not found at {text_to_pose_script}.\n"
            "The pose/record pipeline lives in the sibling uzbek-sign-language repo — "
            "point at it with:\n"
            "  export POSE_REPO_ROOT=/path/to/uzbek-sign-language\n"
            "  python ui.py --open --pose-repo-root /path/to/uzbek-sign-language"
        )

    require_sign_dictionary(uzsl_data)

    # Word-sense disambiguation runs here (this env has the embedding model); the
    # resulting {label: sign_id} map is handed to the pose subprocess as JSON.
    overrides_path: Path | None = None
    sense_overrides: dict[str, str] = {}
    if disambiguate:
        sense_overrides = compute_sense_overrides(text or gloss)
        if sense_overrides:
            overrides_path = output_dir / f"{stem}_overrides.json"
            overrides_path.write_text(
                json.dumps(sense_overrides, ensure_ascii=False), encoding="utf-8"
            )

    cmd = [
        str(pose_python),
        str(Path(__file__).resolve().with_name("_pose_driver.py")),
        str(pose_repo_root),
        text or gloss,
        str(pose_path),
        str(video_path),
        str(report_path),
    ]
    if overrides_path is not None:
        cmd.append(str(overrides_path))
    env = os.environ.copy()
    env["UZSL_DATA_DIR"] = str(uzsl_data)
    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "Pose generation failed").strip()
        if "signs.csv" in details or "No such file or directory" in details:
            details += (
                f"\n\nSet your dataset root (must contain metadata/signs.csv and poses/):\n"
                f"  export UZSL_DATA_DIR=/path/to/uzsl_data"
            )
        if "No module named 'pose_format'" in details:
            details += (
                "\n\nInstall pose dependencies:\n"
                f"  conda create -n {POSE_CONDA_ENV} python=3.11 -y\n"
                f"  conda run -n {POSE_CONDA_ENV} pip install -e "
                f"\"{pose_repo_root}/pose/src/python[mediapipe]\"\n"
                f"  conda run -n {POSE_CONDA_ENV} pip install vidgear opencv-python"
            )
        raise RuntimeError(details)
    if not video_path.is_file():
        raise RuntimeError(f"Pose video was not created: {video_path}")

    report: dict = {}
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    resolutions = format_resolutions(report.get("resolutions", []))

    return {
        "pose_path": str(pose_path),
        "video_path": str(video_path),
        "video_url": f"/media/{video_path.name}",
        "log": result.stdout.strip(),
        "resolutions": resolutions,
        "annotated_gloss": build_annotated_gloss_line(report.get("resolutions", [])),
        "oov": [str(w).upper() for w in report.get("oov", [])],
        "sense_overrides": sense_overrides,
    }


def render_dashboard(comparison_path: Path, predictions_dir: Path) -> str:
    comparison = load_json(comparison_path)
    predictions = load_predictions(predictions_dir)
    models = comparison.get("models", {})
    rows = sorted(
        models.values(),
        key=lambda row: row.get("gloss_token_f1", 0.0),
        reverse=True,
    )
    best_f1 = max((row.get("gloss_token_f1", 0.0) for row in rows), default=0.0)

    metric_rows = []
    for row in rows:
        model = html.escape(str(row.get("model", "")))
        f1 = float(row.get("gloss_token_f1", 0.0))
        exact = float(row.get("exact_match", 0.0))
        bleu = float(row.get("bleu", 0.0))
        chrf = float(row.get("chrf", 0.0))
        bar_width = max(1, round(f1 * 100))
        badge = " best" if f1 == best_f1 and rows else ""
        metric_rows.append(
            f"""
            <tr>
              <td><strong>{model}</strong><span class="badge{badge}">{'best' if badge else ''}</span></td>
              <td>{pct(f1)}<div class="bar"><span style="width:{bar_width}%"></span></div></td>
              <td>{pct(exact)}</td>
              <td>{bleu:.2f}</td>
              <td>{chrf:.2f}</td>
              <td>{int(row.get("num_examples", 0))}</td>
            </tr>
            """
        )

    prediction_cards = []
    for model, model_rows in sorted(predictions.items()):
        examples = []
        for row in model_rows[:12]:
            examples.append(
                f"""
                <article class="example">
                  <p class="text">{html.escape(str(row.get("text", "")))}</p>
                  <p><span>Reference</span>{html.escape(str(row.get("reference", "")))}</p>
                  <p><span>Prediction</span>{html.escape(str(row.get("prediction", "")))}</p>
                </article>
                """
            )
        prediction_cards.append(
            f"""
            <section class="panel">
              <h2>{html.escape(model)}</h2>
              {''.join(examples) if examples else '<p>No prediction rows found.</p>'}
            </section>
            """
        )

    empty = "<p>No metrics yet. Run <code>python eval.py --only all</code> after training.</p>"
    model_options = "\n".join(
        f'<option value="{key}">{html.escape(label)}</option>'
        for key, label in MODEL_LABELS.items()
    )
    g2t_model_options = "\n".join(
        f'<option value="{key}">{html.escape(label)}</option>'
        for key, label in GLOSS_TO_TEXT_MODEL_LABELS.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UzSL Text to Gloss</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #101010;
      --card: #151515;
      --field: #3a3a3a;
      --text: #f5f5f5;
      --muted: #9f9f9f;
      --line: #2a2a2a;
      --accent: #2563eb;
      --good: #22c55e;
      --chip: #063f1f;
    }}
    body {{
      margin: 0;
      font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px; }}
    h1 {{ margin: 0 0 4px; font-size: 34px; letter-spacing: -.03em; }}
    h2 {{ margin: 0 0 16px; }}
    .sub {{ color: var(--muted); margin: 0 0 24px; }}
    .tabs {{ display: flex; gap: 10px; margin: 24px 0; border-bottom: 1px solid var(--line); }}
    .tab {{
      border: 0;
      border-radius: 12px 12px 0 0;
      background: transparent;
      color: var(--muted);
      padding: 12px 18px;
    }}
    .tab.active {{ background: var(--card); color: var(--text); }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    label {{ display: block; color: var(--muted); font-weight: 650; margin: 18px 0 8px; }}
    textarea, select, button {{
      font: inherit;
      border-radius: 10px;
      border: 1px solid #d1d5db;
    }}
    textarea {{
      width: 100%;
      box-sizing: border-box;
      min-height: 120px;
      padding: 14px 16px;
      resize: vertical;
      background: var(--field);
      color: var(--text);
      font-size: 23px;
    }}
    select {{
      min-width: 290px;
      padding: 12px 14px;
      background: #111;
      color: var(--text);
    }}
    button {{
      padding: 12px 22px;
      background: var(--accent);
      color: white;
      border-color: transparent;
      font-weight: 750;
      cursor: pointer;
    }}
    button:disabled {{ opacity: .65; cursor: wait; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 18px; }}
    .hint {{ color: var(--muted); margin-top: 8px; }}
    .panel {{
      background: color-mix(in srgb, var(--card), white 3%);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 20px;
      margin: 18px 0;
      box-shadow: 0 18px 45px rgb(0 0 0 / 18%);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 10px; text-align: left; border-bottom: 1px solid var(--line); }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .bar {{ margin-top: 6px; height: 8px; border-radius: 999px; background: #1f2937; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: linear-gradient(90deg, var(--accent), var(--good)); }}
    .badge {{ margin-left: 8px; color: var(--good); font-size: 12px; }}
    .example {{ border-top: 1px solid var(--line); padding: 14px 0; }}
    .example:first-of-type {{ border-top: 0; padding-top: 0; }}
    .example p {{ margin: 7px 0; }}
    .example span {{ display: block; color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .text {{ font-weight: 650; }}
    #result {{ display: none; }}
    #result.show {{ display: block; }}
    #gloss {{
      margin: 8px 0 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 28px;
      font-weight: 800;
      line-height: 1.45;
      white-space: pre-wrap;
      text-transform: uppercase;
      color: #e2e8f0;
    }}
    #g2t-result {{ display: none; }}
    #g2t-result.show {{ display: block; }}
    #g2t-output {{
      margin: 8px 0 16px;
      font-size: 22px;
      font-weight: 650;
      line-height: 1.5;
      white-space: pre-wrap;
      color: #e2e8f0;
    }}
    .kicker {{ color: var(--muted); text-transform: uppercase; letter-spacing: .08em; font-weight: 750; }}
    .meta {{ color: var(--muted); }}
    .copy {{ background: #64748b; padding: 8px 14px; font-size: 14px; }}
    .secondary {{ background: #334155; }}
    .pose-player {{ margin-top: 18px; }}
    video {{ width: 100%; max-height: 620px; background: #000; border-radius: 14px; border: 1px solid var(--line); }}
    .log {{ white-space: pre-wrap; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; }}
    .error {{ color: #fca5a5; }}
    code {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <h1>UzSL Text -> Gloss</h1>
    <p class="sub">Type Uzbek text, generate gloss, then render and play the pose video.</p>
    <nav class="tabs">
      <button class="tab active" data-tab="try">Try Sentence</button>
      <button class="tab" data-tab="g2t">Gloss -&gt; Text</button>
      <button class="tab" data-tab="comparison">Comparison & Examples</button>
    </nav>
    <section id="tab-try" class="tab-panel active">
      <section>
        <label for="text">Uzbek text</label>
        <textarea id="text">Bizning uyimiz shahardan uzoqroqda joylashgani uchun borib-kelishga sal noqulay.</textarea>
        <div class="controls">
          <select id="model">{model_options}</select>
          <button id="generate">Generate gloss</button>
        </div>
        <p class="hint">Press Ctrl/⌘ + Enter to generate. Then use “Play pose” to render the sign skeleton from the gloss.</p>
      </section>
      <section id="result" class="panel">
        <div class="kicker">Gloss</div>
        <div id="gloss"></div>
        <p id="meta" class="meta"></p>
        <div class="controls">
          <button id="copy" class="copy">Copy gloss</button>
          <button id="play-pose" class="secondary">Play pose</button>
        </div>
        <div id="pose-status" class="meta"></div>
        <div id="pose-player" class="pose-player"></div>
      </section>
    </section>
    <section id="tab-g2t" class="tab-panel">
      <section>
        <label for="g2t-text">Gloss (space-separated)</label>
        <textarea id="g2t-text">men uy o'tirmoq</textarea>
        <div class="controls">
          <select id="g2t-model">{g2t_model_options}</select>
          <button id="g2t-generate">Generate text</button>
        </div>
        <p class="hint">Press Ctrl/⌘ + Enter to generate. Only backbones with a trained gloss-&gt;text checkpoint are listed.</p>
      </section>
      <section id="g2t-result" class="panel">
        <div class="kicker">Text</div>
        <div id="g2t-output"></div>
        <p id="g2t-meta" class="meta"></p>
        <div class="controls">
          <button id="g2t-copy" class="copy">Copy text</button>
        </div>
      </section>
    </section>
    <section id="tab-comparison" class="tab-panel">
      <section class="panel">
        <h2>Model Comparison</h2>
        <p class="sub">Test size: {int(comparison.get("test_size", 0))} examples. Main accuracy rate is gloss-token F1.</p>
        {empty if not rows else f'''
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Gloss Token F1</th>
              <th>Exact Match</th>
              <th>BLEU</th>
              <th>chrF</th>
              <th>Examples</th>
            </tr>
          </thead>
          <tbody>{''.join(metric_rows)}</tbody>
        </table>
        '''}
      </section>
      {''.join(prediction_cards)}
    </section>
  </main>
  <script>
    const textEl = document.getElementById("text");
    const modelEl = document.getElementById("model");
    const buttonEl = document.getElementById("generate");
    const resultEl = document.getElementById("result");
    const glossEl = document.getElementById("gloss");
    const metaEl = document.getElementById("meta");
    const copyEl = document.getElementById("copy");
    const playPoseEl = document.getElementById("play-pose");
    const poseStatusEl = document.getElementById("pose-status");
    const posePlayerEl = document.getElementById("pose-player");
    let lastPrediction = null;

    document.querySelectorAll(".tab").forEach((tab) => {{
      tab.addEventListener("click", () => {{
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`tab-${{tab.dataset.tab}}`).classList.add("active");
      }});
    }});

    async function generateGloss() {{
      const text = textEl.value.trim();
      const model = modelEl.value;
      if (!text) {{
        resultEl.classList.add("show");
        glossEl.innerHTML = '<span class="error">Enter an Uzbek sentence first.</span>';
        metaEl.textContent = "";
        return;
      }}
      buttonEl.disabled = true;
      buttonEl.textContent = "Generating...";
      resultEl.classList.add("show");
      glossEl.textContent = "Loading model and generating...";
      metaEl.textContent = "";
      try {{
        const response = await fetch("/api/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ text, model }}),
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed");
        lastPrediction = data;
        glossEl.textContent = data.gloss || "";
        poseStatusEl.textContent = "";
        posePlayerEl.innerHTML = "";
        metaEl.textContent = `${{data.model_label}} · ${{(data.tokens || []).length}} tokens · ${{data.latency_seconds}}s`;
      }} catch (error) {{
        lastPrediction = null;
        glossEl.innerHTML = `<span class="error">${{error.message}}</span>`;
      }} finally {{
        buttonEl.disabled = false;
        buttonEl.textContent = "Generate gloss";
      }}
    }}

    buttonEl.addEventListener("click", generateGloss);
    textEl.addEventListener("keydown", (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") generateGloss();
    }});
    copyEl.addEventListener("click", async () => {{
      await navigator.clipboard.writeText(glossEl.textContent || "");
      copyEl.textContent = "Copied";
      setTimeout(() => copyEl.textContent = "Copy gloss", 1000);
    }});

    const g2tTextEl = document.getElementById("g2t-text");
    const g2tModelEl = document.getElementById("g2t-model");
    const g2tButtonEl = document.getElementById("g2t-generate");
    const g2tResultEl = document.getElementById("g2t-result");
    const g2tOutputEl = document.getElementById("g2t-output");
    const g2tMetaEl = document.getElementById("g2t-meta");
    const g2tCopyEl = document.getElementById("g2t-copy");

    async function generateText() {{
      const text = g2tTextEl.value.trim();
      const model = g2tModelEl.value;
      if (!text) {{
        g2tResultEl.classList.add("show");
        g2tOutputEl.innerHTML = '<span class="error">Enter a gloss sequence first.</span>';
        g2tMetaEl.textContent = "";
        return;
      }}
      g2tButtonEl.disabled = true;
      g2tButtonEl.textContent = "Generating...";
      g2tResultEl.classList.add("show");
      g2tOutputEl.textContent = "Loading model and generating...";
      g2tMetaEl.textContent = "";
      try {{
        const response = await fetch("/api/predict", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ text, model, direction: "gloss2text" }}),
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction failed");
        g2tOutputEl.textContent = data.gloss || "";
        g2tMetaEl.textContent = `${{data.model_label}} · ${{data.latency_seconds}}s`;
      }} catch (error) {{
        g2tOutputEl.innerHTML = `<span class="error">${{error.message}}</span>`;
      }} finally {{
        g2tButtonEl.disabled = false;
        g2tButtonEl.textContent = "Generate text";
      }}
    }}

    g2tButtonEl.addEventListener("click", generateText);
    g2tTextEl.addEventListener("keydown", (event) => {{
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") generateText();
    }});
    g2tCopyEl.addEventListener("click", async () => {{
      await navigator.clipboard.writeText(g2tOutputEl.textContent || "");
      g2tCopyEl.textContent = "Copied";
      setTimeout(() => g2tCopyEl.textContent = "Copy text", 1000);
    }});
    playPoseEl.addEventListener("click", async () => {{
      const gloss = (lastPrediction && lastPrediction.gloss) || glossEl.textContent || "";
      const text = (lastPrediction && lastPrediction.text) || textEl.value.trim();
      if (!gloss.trim()) {{
        poseStatusEl.innerHTML = '<span class="error">Generate a gloss first.</span>';
        return;
      }}
      playPoseEl.disabled = true;
      playPoseEl.textContent = "Rendering pose...";
      poseStatusEl.textContent = "Generating .pose and skeleton video from gloss...";
      posePlayerEl.innerHTML = "";
      try {{
        const response = await fetch("/api/pose", {{
          method: "POST",
          headers: {{ "Content-Type": "application/json" }},
          body: JSON.stringify({{ text, gloss }}),
        }});
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Pose generation failed");
        if (data.annotated_gloss) glossEl.textContent = data.annotated_gloss;
        poseStatusEl.textContent = "Pose video ready.";
        posePlayerEl.innerHTML = `
          <video src="${{data.video_url}}" controls autoplay loop playsinline></video>
          <p class="meta">Pose: ${{data.pose_path}}</p>
          <details><summary>Generation log</summary><pre class="log">${{data.log || ""}}</pre></details>
        `;
      }} catch (error) {{
        poseStatusEl.innerHTML = `<span class="error">${{error.message}}</span>`;
      }} finally {{
        playPoseEl.disabled = false;
        playPoseEl.textContent = "Play pose";
      }}
    }});
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    comparison_path: Path
    predictions_dir: Path
    pose_output_dir: Path
    pose_python: Path
    pose_repo_root: Path
    uzsl_data_dir: Path
    disambiguate: bool = DISAMBIGUATE_DEFAULT

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/media/"):
            self.serve_media(parsed.path)
            return
        if parsed.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        body = render_dashboard(self.comparison_path, self.predictions_dir).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in ("/api/predict", "/api/pose"):
            self.send_error(404)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            if self.path == "/api/predict":
                result = predict_sentence(
                    str(payload.get("text", "")),
                    str(payload.get("model", "nllb")),
                    str(payload.get("direction", "text2gloss")),
                )
            else:
                result = generate_pose_video(
                    str(payload.get("text", "")),
                    str(payload.get("gloss", "")),
                    self.pose_python,
                    self.pose_output_dir,
                    self.uzsl_data_dir,
                    pose_repo_root=self.pose_repo_root,
                    disambiguate=self.disambiguate,
                )
            body = json.dumps(result, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_media(self, path: str) -> None:
        name = Path(unquote(path.removeprefix("/media/"))).name
        media_path = (self.pose_output_dir / name).resolve()
        root = self.pose_output_dir.resolve()
        if root not in media_path.parents or not media_path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(media_path.name)[0] or "application/octet-stream"
        body = media_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--comparison", type=Path, default=OUTPUTS_DIR / "comparison.json")
    p.add_argument("--predictions-dir", type=Path, default=OUTPUTS_DIR / "predictions")
    p.add_argument("--pose-output-dir", type=Path, default=POSE_OUTPUT_DIR)
    p.add_argument(
        "--pose-python",
        type=Path,
        default=DEFAULT_POSE_PYTHON,
        help=f"Python with mediapipe/pose_format (conda env '{POSE_CONDA_ENV}' auto-detected)",
    )
    p.add_argument(
        "--pose-repo-root",
        type=Path,
        default=DEFAULT_POSE_REPO_ROOT,
        help="Root of the uzbek-sign-language checkout (record/tools/text_to_pose.py)",
    )
    p.add_argument(
        "--uzsl-data-dir",
        type=Path,
        default=DEFAULT_UZSL_DATA,
        help="Root of uzsl_data (metadata/, poses/, videos/)",
    )
    p.add_argument(
        "--disambiguate",
        dest="disambiguate",
        action="store_true",
        default=DISAMBIGUATE_DEFAULT,
        help="Pick homograph senses from context before rendering (default: on)",
    )
    p.add_argument(
        "--no-disambiguate",
        dest="disambiguate",
        action="store_false",
        help="Disable context-based sense selection (use most-attested sense)",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true", help="Open the dashboard in a browser")
    args = p.parse_args()

    DashboardHandler.comparison_path = args.comparison
    DashboardHandler.predictions_dir = args.predictions_dir
    DashboardHandler.pose_output_dir = args.pose_output_dir.expanduser().resolve()
    DashboardHandler.pose_python = python_launcher(args.pose_python)
    DashboardHandler.pose_repo_root = args.pose_repo_root.expanduser().resolve()
    DashboardHandler.uzsl_data_dir = resolve_uzsl_data(args.uzsl_data_dir)
    DashboardHandler.disambiguate = args.disambiguate
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving text-to-gloss dashboard at {url}")
    print(f"Using pose python={DashboardHandler.pose_python}")
    print(f"Using pose repo root={DashboardHandler.pose_repo_root}")
    print(f"Using UZSL_DATA_DIR={DashboardHandler.uzsl_data_dir}")
    print(f"Sense disambiguation={'on' if DashboardHandler.disambiguate else 'off'}")
    if args.open:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
