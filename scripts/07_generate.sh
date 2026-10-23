#!/usr/bin/env bash
# Gera imagens de teste com um checkpoint do LoRA, usando os prompts de
# configs/test_prompts.template.txt (seed fixa, para comparar checkpoints).
#
# Uso: scripts/07_generate.sh <lora.safetensors> [multiplier]
# Exemplo, comparar vários passos:
#   for f in output/lora_afnsx-step*.safetensors; do scripts/07_generate.sh "$f"; done
source "$(dirname "$0")/common.sh"
activate_venv
load_models

lora="${1:-}"; mult="${2:-1.0}"
[[ -f "$lora" ]] || die "Uso: $0 <lora.safetensors> [multiplier]"

out="samples/$(basename "${lora%.safetensors}")_x${mult}"
mkdir -p "$out" configs
# Só o que é carregado primeiro: os três juntos (~33 GB) não cabem na cache da RAM.
warm_cache "$TE_PATH" "$VAE_PATH"
sed "s/{TRIGGER}/$TRIGGER/g; s/{CLASS}/$CLASS_WORD/g" configs/test_prompts.template.txt > configs/test_prompts.txt

log "A gerar para $(basename "$lora") (x$mult) em $out"
python "$MUSUBI_DIR/src/musubi_tuner/qwen_image_21_generate_image.py" \
  --dit "$DIT_PATH" \
  --vae "$VAE_PATH" \
  --text_encoder "$TE_PATH" \
  --fp8_scaled --fp8_vl --vae_tiling \
  --blocks_to_swap "$BLOCKS_TO_SWAP" \
  --attn_mode sdpa \
  --lora_weight "$lora" --lora_multiplier "$mult" \
  --from_file configs/test_prompts.txt \
  --image_size 1024 1024 --infer_steps 40 --seed "$SEED" \
  --save_path "$out"

log "Imagens em $out"
