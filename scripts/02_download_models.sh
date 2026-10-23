#!/usr/bin/env bash
# Descarrega o Qwen-Image 2.1 (DiT + VAE + text encoder Qwen3-VL-8B, ~47 GB)
# e o modelo de captioning (Qwen3-VL-4B, ~9 GB) para o cache do Hugging Face.
# Grava os caminhos em models.env.
#
# Uso: scripts/02_download_models.sh            (tudo)
#      scripts/02_download_models.sh --no-caption (salta o modelo de captions)
source "$(dirname "$0")/common.sh"
activate_venv

with_caption=1
[[ "${1:-}" == "--no-caption" ]] && with_caption=0

log "A descarregar $QWEN_IMAGE_REPO (pode demorar, retoma se interromperes)"
qwen_dir=$(hf download "$QWEN_IMAGE_REPO" --exclude "assets/*" | tail -n1)
[[ -d "$qwen_dir/transformer" ]] || die "Download incompleto: $qwen_dir"

caption_dir=""
if (( with_caption )); then
  log "A descarregar $CAPTION_MODEL_REPO"
  caption_dir=$(hf download "$CAPTION_MODEL_REPO" | tail -n1)
fi

cat > "$MODELS_ENV" <<EOF
# Gerado por 02_download_models.sh
QWEN_DIR="$qwen_dir"
CAPTION_MODEL_DIR="$caption_dir"
EOF

log "Modelos em:"
cat "$MODELS_ENV"
log "Próximo: python scripts/03_prepare_dataset.py"
