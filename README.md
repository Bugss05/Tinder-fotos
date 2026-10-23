# Tinder-fotos

Treino local de um LoRA de identidade para o **Qwen-Image 2.1**, numa RX 9070 XT (16 GB) em Linux com ROCm, usando o [musubi-tuner](https://github.com/kohya-ss/musubi-tuner).

O objetivo final é gerar fotos realistas com a minha cara. Este repo cobre a primeira fase: preparar o dataset e treinar o LoRA.

## Hardware alvo

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT, 16 GB (gfx1201, RDNA4) |
| CPU | Ryzen 7 5800X3D |
| RAM | 32 GB DDR4 |
| SO | Linux com ROCm 7.2.x |

## Como a memória cabe nos 16 GB

| Componente | Tamanho | Quando é usado |
|---|---|---|
| DiT Qwen-Image 2.1 (7.1B) | ~14 GB bf16, **~7 GB em fp8** | treino e geração |
| Text encoder Qwen3-VL-8B | ~16 GB bf16, **~8 GB em fp8** | só na cache (passo 5) e na geração |
| VAE | pequeno | só na cache e na geração |

O truque é fazer a cache dos latents e dos embeddings antes do treino. Assim, durante o treino só o DiT está carregado, em fp8, com gradient checkpointing e alguns blocos em block swap.

## Passo a passo

Tudo se configura em [`config.env`](config.env). No mínimo muda `DATASET_SRC` para a pasta das tuas fotos.

```bash
scripts/00_check_system.sh      # ROCm, grupos, RAM, disco
scripts/01_setup_env.sh         # venv Py3.12 + PyTorch ROCm (AMD) + musubi-tuner
scripts/02_download_models.sh   # ~56 GB para o cache do Hugging Face
scripts/03_prepare_dataset.sh   # EXIF, resize, duplicados, legenda base
scripts/04_caption.sh           # legendas com Qwen3-VL-4B  -> REVER À MÃO
scripts/05_cache.sh             # latents + embeddings do text encoder
scripts/06_train_lora.sh        # treino
scripts/07_generate.sh output/lora_afnsx-step00000750.safetensors   # testar
```

Para acompanhar o treino: `tensorboard --logdir logs` (dentro do venv) e `amdgpu_top` ou `nvtop` noutro terminal para ver a VRAM.

## Dataset: o que importa

- **20 a 40 fotos** boas valem mais do que 100 médias.
- Variedade: grandes planos, meio corpo, corpo inteiro; luz de dia e interior; roupas e fundos diferentes; expressões diferentes.
- Nada de óculos de sol, chapéus a tapar a cara, filtros ou fotos de grupo.
- As legendas descrevem o que **não** é identidade (roupa, sítio, luz, pose). A cara e o corpo não se descrevem: é isso que o trigger (`afnsx man`) tem de aprender.

## Afinar

| Sintoma | O que mudar |
|---|---|
| Out of memory | subir `BLOCKS_TO_SWAP` (12, 16...) ou `RESOLUTION=768` |
| Sobra VRAM e quero mais velocidade | baixar `BLOCKS_TO_SWAP` (4, depois 0) |
| Não se parece comigo | mais passos, ou `NETWORK_DIM=32` |
| Parece-me sempre com a mesma roupa ou fundo | legendas pouco descritivas, ou checkpoint demasiado tardio |
| O prompt de controlo (sem trigger) também tem a minha cara | overfit: usa um checkpoint anterior ou baixa o learning rate |

Os checkpoints ficam a cada 250 passos; normalmente o melhor está entre 1000 e 1750 passos, não no fim. Compara-os todos com o `07_generate.sh` (usa seeds fixas).

## Problemas conhecidos em ROCm

- `hipErrorNoBinaryForGpu`: a versão do ROCm no sistema não bate certo com as wheels (7.2.1). Atualiza o ROCm ou muda `ROCM_WHEEL_BASE`/`ROCM_WHEELS` no `config.env`.
- Erro numpy ao importar torch: `uv pip install "numpy<2"` dentro do venv.
- Primeira execução lenta nas convoluções do VAE: é o MIOpen a compilar kernels; já está em `MIOPEN_FIND_MODE=FAST`.
- O suporte ao Qwen-Image 2.1 no musubi-tuner é experimental e está no branch `dev`, fixado num commit em `config.env`.

## Notas

- O repo é público: `data/`, `output/`, imagens e pesos estão no `.gitignore`. Não os adiciones à força.
- O Qwen-Image 2.1 está sob a *Qwen Research License* (uso não comercial), que se estende aos LoRAs treinados sobre ele.
