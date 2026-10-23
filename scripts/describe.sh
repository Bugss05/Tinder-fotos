#!/usr/bin/env bash
# Descreve fotos ou responde a perguntas sobre elas (VLM em FP8).
#   scripts/describe.sh foto.jpg
#   scripts/describe.sh foto.jpg -q "A cara está nítida e bem iluminada?"
#   scripts/describe.sh ~/fotos-lora -q "Classifica de 1 a 10 para um dataset de LoRA e diz porquê" -o avaliacao.txt
#   scripts/describe.sh -i        (modo interativo)
source "$(dirname "$0")/common.sh"
activate_venv
load_vlm_dir
python scripts/describe.py --model "$VLM_DIR" --precision "$VLM_PRECISION" ${VLM_PROCESSOR_DIR:+--processor "$VLM_PROCESSOR_DIR"} "$@"
