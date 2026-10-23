#!/usr/bin/env bash
# Copia as fotos de DATASET_SRC para data/train, já limpas e com legendas base.
# Argumentos extra passam para o prepare_dataset.py (ex.: --force).
source "$(dirname "$0")/common.sh"
activate_venv

python scripts/prepare_dataset.py \
  --src "$DATASET_SRC" \
  --dst "$DATA_DIR" \
  --trigger "$TRIGGER" \
  --class-word "$CLASS_WORD" \
  --max-side "$PREP_MAX_SIDE" \
  "$@"

log "Revê os avisos acima e apaga à mão as fotos más em $DATA_DIR (apaga o .txt também)."
log "Próximo: scripts/04_caption.sh"
