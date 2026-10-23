#!/usr/bin/env bash
# Gera legendas com Qwen3-VL-4B (~9 GB VRAM em bf16).
# Depois LÊ as legendas e corrige à mão; é o passo que mais pesa na qualidade.
# Argumentos extra passam para o caption.py (ex.: --overwrite).
source "$(dirname "$0")/common.sh"
activate_venv
source "$MODELS_ENV" 2>/dev/null || true
[[ -n "${CAPTION_MODEL_DIR:-}" ]] || die "Modelo de captions em falta. Corre scripts/02_download_models.sh (sem --no-caption)."

python scripts/caption.py \
  --dir "$DATA_DIR" \
  --model "$CAPTION_MODEL_DIR" \
  --trigger "$TRIGGER" \
  --class-word "$CLASS_WORD" \
  "$@"

log "Revê os .txt em $DATA_DIR. Próximo: scripts/05_cache.sh"
