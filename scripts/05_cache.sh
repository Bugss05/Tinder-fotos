#!/usr/bin/env bash
# Gera configs/dataset.toml e pré-calcula latents (VAE) e embeddings do
# text encoder. Tem de ser refeito sempre que mudares fotos, legendas ou
# RESOLUTION. O treino em si não volta a carregar VAE nem text encoder.
source "$(dirname "$0")/common.sh"
activate_venv
load_models

n_imgs=$(find "$DATA_DIR" -maxdepth 1 -name '*.jpg' | wc -l)
(( n_imgs > 0 )) || die "Sem imagens em $DATA_DIR. Corre scripts/03_prepare_dataset.sh"

mkdir -p configs "$CACHE_DIR"
cat > configs/dataset.toml <<EOF
# Gerado por scripts/05_cache.sh a partir de config.env
[general]
resolution = [$RESOLUTION, $RESOLUTION]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "$REPO_ROOT/$DATA_DIR"
cache_directory = "$REPO_ROOT/$CACHE_DIR"
num_repeats = $NUM_REPEATS
EOF
log "configs/dataset.toml escrito ($n_imgs imagens, ${RESOLUTION}px)"

# Mudar resolução ou fotos invalida a cache; recomeçar do zero é o mais seguro.
rm -rf "${CACHE_DIR:?}"/*

log "Cache de latents (VAE)"
python "$MUSUBI_DIR/src/musubi_tuner/qwen_image_21_cache_latents.py" \
  --dataset_config configs/dataset.toml \
  --vae "$VAE_PATH" \
  --vae_tiling

log "Cache do text encoder (Qwen3-VL-8B em fp8)"
warm_cache "$TE_PATH"
python "$MUSUBI_DIR/src/musubi_tuner/qwen_image_21_cache_text_encoder_outputs.py" \
  --dataset_config configs/dataset.toml \
  --text_encoder "$TE_PATH" \
  --fp8_vl \
  --batch_size 1

log "Cache pronta. Próximo: scripts/06_train_lora.sh"
