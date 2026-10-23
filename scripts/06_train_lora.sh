#!/usr/bin/env bash
# Treina o LoRA. Configuração pensada para RX 9070 XT (16 GB) + 32 GB RAM:
#   DiT em fp8 escalado (~7 GB), gradient checkpointing, block swap H2D-only.
#
# Uso: scripts/06_train_lora.sh                 (treino novo)
#      scripts/06_train_lora.sh --resume <pasta-state>
# Argumentos extra passam diretamente ao musubi-tuner.
source "$(dirname "$0")/common.sh"
activate_venv
load_models

[[ -f configs/dataset.toml ]] || die "Falta configs/dataset.toml. Corre scripts/05_cache.sh"
compgen -G "$CACHE_DIR/*.safetensors" >/dev/null || die "Cache vazia. Corre scripts/05_cache.sh"

mkdir -p "$OUTPUT_DIR" logs
warm_cache "$DIT_PATH"

extra=()
if (( BLOCKS_TO_SWAP > 0 )); then
  extra+=(--blocks_to_swap "$BLOCKS_TO_SWAP" --block_swap_h2d_only)
fi

if (( SAMPLE_EVERY_N_STEPS > 0 )); then
  sed "s/{TRIGGER}/$TRIGGER/g; s/{CLASS}/$CLASS_WORD/g" configs/sample_prompts.template.txt > configs/sample_prompts.txt
  extra+=(--sample_prompts configs/sample_prompts.txt --sample_every_n_steps "$SAMPLE_EVERY_N_STEPS"
          --vae "$VAE_PATH" --text_encoder "$TE_PATH" --fp8_vl --vae_tiling)
fi

log "A treinar $OUTPUT_NAME: dim=$NETWORK_DIM lr=$LEARNING_RATE steps=$MAX_TRAIN_STEPS swap=$BLOCKS_TO_SWAP"

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  "$MUSUBI_DIR/src/musubi_tuner/qwen_image_21_train_network.py" \
  --dit "$DIT_PATH" \
  --dataset_config configs/dataset.toml \
  --sdpa --mixed_precision bf16 \
  --fp8_base --fp8_scaled \
  --gradient_checkpointing \
  --network_module networks.lora_qwen_image_21 \
  --network_dim "$NETWORK_DIM" --network_alpha "$NETWORK_ALPHA" \
  --optimizer_type AdamW --learning_rate "$LEARNING_RATE" \
  --lr_scheduler constant_with_warmup --lr_warmup_steps 50 \
  --timestep_sampling shift --discrete_flow_shift 3.0 \
  --weighting_scheme none \
  --max_train_steps "$MAX_TRAIN_STEPS" \
  --save_every_n_steps "$SAVE_EVERY_N_STEPS" \
  --max_data_loader_n_workers 2 --persistent_data_loader_workers \
  --seed "$SEED" \
  --logging_dir logs --log_with tensorboard \
  --output_dir "$OUTPUT_DIR" --output_name "$OUTPUT_NAME" \
  "${extra[@]}" \
  "$@"

log "Feito. Checkpoints em $OUTPUT_DIR/. Testa com: scripts/07_generate.sh $OUTPUT_DIR/${OUTPUT_NAME}.safetensors"
