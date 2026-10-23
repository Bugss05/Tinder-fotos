#!/usr/bin/env bash
# Converte um LoRA do musubi para o formato que o ComfyUI lê por inteiro.
#
# Porquê: o ComfyUI carrega o Qwen-Image 2.1 com as camadas MLP fundidas
# (gate_layer + proj = gate_up). No formato nativo do musubi (lora_unet_...)
# o ComfyUI não reconhece essas duas metades e ignora-as SEM dar erro: o
# LoRA fica mais fraco. Convertido para chaves "transformer.<módulo>.lora_A",
# o ComfyUI mapeia cada metade para a camada fundida.
#
# Uso: scripts/08_export_comfyui.sh <lora.safetensors> [pasta ComfyUI]
#   ex: scripts/08_export_comfyui.sh output/lora_afnsx-step00001250.safetensors ~/ComfyUI
source "$(dirname "$0")/common.sh"
activate_venv

lora="${1:-}"; comfy="${2:-}"
[[ -f "$lora" ]] || die "Uso: $0 <lora.safetensors> [pasta do ComfyUI]"

out="${lora%.safetensors}_comfyui.safetensors"
python "$MUSUBI_DIR/src/musubi_tuner/convert_lora.py" \
  --input "$lora" --output "$out" \
  --target other --diffusers_prefix transformer
log "Convertido: $out"

if [[ -n "$comfy" ]]; then
  [[ -d "$comfy/models/loras" ]] || die "Não encontrei $comfy/models/loras"
  cp "$out" "$comfy/models/loras/"
  log "Copiado para $comfy/models/loras/ (no ComfyUI: nó 'LoraLoaderModelOnly')"
fi
