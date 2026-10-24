#!/usr/bin/env bash
# Gera imagens de teste com um checkpoint do LoRA, usando os prompts de
# configs/test_prompts.template.txt (seed fixa, para comparar checkpoints).
#
# Uso: scripts/07_generate.sh <lora.safetensors | base> [multiplier]
#   base  -> sem LoRA (modelo original), para comparar e diagnosticar artefactos
#
# Diagnóstico de artefactos (variáveis de ambiente, combináveis):
#   DIT_FP8=0      DiT em bf16 em vez de fp8 (mais lento, usa mais block swap)
#   VAE_TILING=0   descodifica a imagem inteira de uma vez (mais VRAM no fim)
#   SDPA_SAFE=1    desliga os kernels de atenção experimentais do ROCm
# Ex.: DIT_FP8=0 scripts/07_generate.sh base
source "$(dirname "$0")/common.sh"
activate_venv
load_models

lora="${1:-}"; mult="${2:-1.0}"
[[ "$lora" == "base" || -f "$lora" ]] || die "Uso: $0 <lora.safetensors | base> [multiplier]"

DIT_FP8="${DIT_FP8:-1}"; VAE_TILING="${VAE_TILING:-1}"; SDPA_SAFE="${SDPA_SAFE:-0}"

opts=(--fp8_vl)
tag=""
swap="$BLOCKS_TO_SWAP"
if [[ "$DIT_FP8" == 1 ]]; then opts+=(--fp8_scaled); else tag+="_bf16"; swap=$(( swap > 18 ? swap : 18 )); fi
if [[ "$VAE_TILING" == 1 ]]; then opts+=(--vae_tiling); else tag+="_notile"; fi
if [[ "$SDPA_SAFE" == 1 ]]; then export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0; tag+="_sdpasafe"; fi

if [[ "$lora" == "base" ]]; then
  name="base"
  lora_opts=()
else
  name="$(basename "${lora%.safetensors}")_x${mult}"
  lora_opts=(--lora_weight "$lora" --lora_multiplier "$mult")
fi
out="samples/${name}${tag}"
mkdir -p "$out" configs
# Só o que é carregado primeiro: os três juntos (~33 GB) não cabem na cache da RAM.
warm_cache "$TE_PATH" "$VAE_PATH"
sed "s/{TRIGGER}/$TRIGGER/g; s/{CLASS}/$CLASS_WORD/g" configs/test_prompts.template.txt > configs/test_prompts.txt

log "A gerar: $name${tag:+ ($tag)} -> $out"
python "$MUSUBI_DIR/src/musubi_tuner/qwen_image_21_generate_image.py" \
  --dit "$DIT_PATH" \
  --vae "$VAE_PATH" \
  --text_encoder "$TE_PATH" \
  "${opts[@]}" \
  --blocks_to_swap "$swap" \
  --attn_mode sdpa \
  "${lora_opts[@]}" \
  --from_file configs/test_prompts.txt \
  --image_size 1024 1024 --infer_steps 40 --seed "$SEED" \
  --save_path "$out"

log "Imagens em $out"
