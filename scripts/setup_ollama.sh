#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Install, start and prepare Ollama INSIDE the Google Colab VM.
#
# Ollama models are not a hosted API: they are downloaded into this runtime and
# served locally on http://localhost:11434 for the translator to call.
#
# Usage:   bash scripts/setup_ollama.sh qwen3:32b [gemma3:27b ...]
#          bash scripts/setup_ollama.sh --all      # the four registry models
# Env:     OLLAMA_NUM_PARALLEL (default 2), OLLAMA_KEEP_ALIVE (default 30m),
#          OLLAMA_MAX_LOADED_MODELS (default 1),
#          OLLAMA_MODELS=<a folder on Drive, e.g. next to this repo>  (optional: keep
#          downloads on Drive so they survive runtime resets; ~75 GB for all
#          four models and slower to load from Drive).
# -----------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_LOG="${OLLAMA_LOG:-$REPO_ROOT/logs/ollama_serve.log}"
mkdir -p "$(dirname "$OLLAMA_LOG")"

ALL_MODELS=("qwen3:32b" "qwen3:30b" "gemma3:27b" "aya-expanse:32b")
if [[ $# -eq 0 ]]; then
  echo "usage: $0 <model> [<model> ...] | --all" >&2
  exit 2
fi
if [[ "$1" == "--all" ]]; then MODELS=("${ALL_MODELS[@]}"); else MODELS=("$@"); fi

export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-2}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
if [[ -n "${OLLAMA_MODELS:-}" ]]; then
  mkdir -p "$OLLAMA_MODELS"
  echo "[ollama] Model store: $OLLAMA_MODELS (persistent; ~75 GB for all four models)"
fi

SUDO=""
if [[ $(id -u) -ne 0 ]] && command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# 1) prerequisites -------------------------------------------------------------
missing=()
command -v lspci >/dev/null 2>&1 || missing+=(pciutils)
command -v lshw  >/dev/null 2>&1 || missing+=(lshw)
command -v zstd  >/dev/null 2>&1 || missing+=(zstd)
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[ollama] Installing prerequisites: ${missing[*]}"
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq "${missing[@]}" >/dev/null
fi

# 2) install Ollama via the official script ------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo "[ollama] Installing Ollama (official install script)…"
  curl -fsSL https://ollama.com/install.sh | sh
fi

# 3) start the server in the background -----------------------------------------
health() { curl -fsS "http://${OLLAMA_HOST}/api/version" >/dev/null 2>&1; }
if ! health; then
  echo "[ollama] Starting 'ollama serve' (log: $OLLAMA_LOG)"
  nohup ollama serve >$OLLAMA_LOG 2>&1 &
  for i in $(seq 1 60); do
    if health; then break; fi
    sleep 1
    if [[ $i -eq 60 ]]; then echo "[ollama] Server did not start; see $OLLAMA_LOG" >&2; exit 1; fi
  done
fi
echo "[ollama] Server is up: $(curl -fsS http://${OLLAMA_HOST}/api/version)"

# 4) pull only the selected model(s) ---------------------------------------------
for m in "${MODELS[@]}"; do
  if ollama list | awk 'NR>1 {print $1}' | grep -qx "$m"; then
    echo "[ollama] $m already present"
  else
    echo "[ollama] Pulling $m … (tens of GB; this can take a while)"
    ollama pull "$m"
  fi
done

echo "[ollama] Ready. OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL OLLAMA_MAX_LOADED_MODELS=$OLLAMA_MAX_LOADED_MODELS"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
  echo "[ollama] WARNING: no GPU visible — models will run on CPU (extremely slow)."
fi
