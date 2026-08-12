#!/usr/bin/env bash
# Download fine-tuned checkpoints from a GitHub release and lay them out
# under models/<name>/ (see src/ttg/config.py:resolve_model_dir).
#
# Usage: scripts/fetch_weights.sh [tag] [repo]
#   tag   release tag to pull from (default: models-v1)
#   repo  owner/repo (default: zehnmind/uzsl-text-to-gloss)
set -euo pipefail

TAG="${1:-models-v1}"
REPO="${2:-zehnmind/uzsl-text-to-gloss}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

command -v gh >/dev/null || { echo "gh (GitHub CLI) is required" >&2; exit 1; }

echo "Downloading release '$TAG' from $REPO into $TMP_DIR ..."
gh release download "$TAG" --repo "$REPO" --dir "$TMP_DIR"

echo "Reassembling checkpoints into $MODELS_DIR ..."
declare -A SEEN_WHOLE=()
for asset in "$TMP_DIR"/*; do
  base="$(basename "$asset")"
  case "$base" in
    CHECKSUMS.sha256|MANIFEST.txt) continue ;;
  esac

  model="${base%%-*}"
  rest="${base#*-}"
  out_dir="$MODELS_DIR/$model"
  mkdir -p "$out_dir"

  if [[ "$rest" =~ ^(.+)\.part[0-9]+$ ]]; then
    fname="${BASH_REMATCH[1]}"
    key="$model/$fname"
    if [[ -z "${SEEN_WHOLE[$key]:-}" ]]; then
      echo "  $key <- concatenating parts"
      cat "$TMP_DIR/${model}-${fname}.part"* > "$out_dir/$fname"
      SEEN_WHOLE[$key]=1
    fi
  else
    fname="$rest"
    echo "  $model/$fname <- copy"
    cp "$asset" "$out_dir/$fname"
  fi
done

if [[ -f "$TMP_DIR/CHECKSUMS.sha256" ]]; then
  echo "Verifying checksums ..."
  (cd "$MODELS_DIR" && sha256sum -c "$TMP_DIR/CHECKSUMS.sha256" --ignore-missing)
fi

echo "Done. Weights are under $MODELS_DIR/<model>/."
