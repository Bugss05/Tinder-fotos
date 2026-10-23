#!/usr/bin/env bash
# Gera legendas com a VLM (Qwen3.5-9B em FP8, ~12 GB de VRAM).
# Depois LÊ as legendas e corrige à mão; é o passo que mais pesa na qualidade.
# Argumentos extra passam para o caption.py (ex.: --overwrite).
source "$(dirname "$0")/common.sh"
activate_venv
load_vlm_dir

python scripts/caption.py \
  --dir "$DATA_DIR" \
  --model "$VLM_DIR" \
  --precision "$VLM_PRECISION" ${VLM_PROCESSOR_DIR:+--processor "$VLM_PROCESSOR_DIR"} \
  --trigger "$TRIGGER" \
  --class-word "$CLASS_WORD" \
  "$@"

log "Revê os .txt em $DATA_DIR. Próximo: scripts/05_cache.sh"
