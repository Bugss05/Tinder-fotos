#!/usr/bin/env bash
# Descarrega o Qwen-Image 2.1 (DiT + VAE + text encoder Qwen3-VL-8B, ~47 GB)
# e a VLM Qwen3.5-9B (~19 GB) para o cache do Hugging Face.
# Grava os caminhos em models.env.
#
# Uso: scripts/02_download_models.sh            (tudo)
#      scripts/02_download_models.sh --no-vlm  (salta a VLM)
source "$(dirname "$0")/common.sh"
activate_venv

with_vlm=1
[[ "${1:-}" == "--no-vlm" ]] && with_vlm=0

log "A descarregar $QWEN_IMAGE_REPO (pode demorar, retoma se interromperes)"
# hf_fetch.py em vez do CLI `hf`: o output do CLI muda entre versões
# (ex.: acaba em "path: ..."), a API devolve só o caminho.
qwen_dir=$(python scripts/hf_fetch.py "$QWEN_IMAGE_REPO" "assets/*")
[[ -d "$qwen_dir/transformer" ]] || die "Download incompleto: $qwen_dir"

vlm_dir=""
if (( with_vlm )); then
  log "A descarregar $VLM_REPO"
  vlm_dir=$(python scripts/hf_fetch.py "$VLM_REPO")
  [[ -f "$vlm_dir/config.json" ]] || die "Download incompleto: $vlm_dir"
fi

cat > "$MODELS_ENV" <<EOF
# Gerado por 02_download_models.sh
QWEN_DIR="$qwen_dir"
VLM_DIR="$vlm_dir"
EOF

log "Modelos em:"
cat "$MODELS_ENV"
log "Próximo: scripts/03_prepare_dataset.sh"
