# Tinder-fotos

Treino local de um LoRA de identidade para o **Qwen-Image 2.1**, numa RX 9070 XT (16 GB) em Linux com ROCm. Usa o [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) para treinar e o **Qwen3.5-9B** como VLM, tanto para legendar as fotos como para as descrever.

O objetivo final é gerar fotos realistas com a minha cara. Este repo cobre a primeira fase: preparar o dataset, legendá-lo, treinar o LoRA, testá-lo e exportá-lo para o ComfyUI.

**Quem faz o quê:**

| Modelo | Papel | Quando |
|---|---|---|
| Qwen3.5-9B (VLM) | Lê fotos e escreve texto: legendas e descrições. **Não gera imagens.** | Só antes do treino (passos 5 e 6) |
| Qwen-Image 2.1 | Gera as imagens. É este que o LoRA ensina a desenhar a tua cara. | Treino, testes e depois no ComfyUI |

---

## 1. Antes de começar (o que tens de ter)

| Requisito | Porquê | Como verificar |
|---|---|---|
| Linux com **ROCm 7.2.x** instalado | As wheels de PyTorch usadas são as da AMD para ROCm 7.2.1 | `cat /opt/rocm/.info/version` |
| Utilizador nos grupos `render` e `video` | Sem isto o PyTorch não vê a GPU | `id -nG` |
| `uv` e `git` | O `uv` cria o venv e instala o Python 3.12 sozinho | `uv --version` |
| **~100 GB de disco livre** | Qwen-Image 2.1 ~47 GB, Qwen3.5-9B ~19 GB, venv e wheels ~10 GB, cache e checkpoints | `df -h .` |
| 32 GB de RAM e **16 GB de swap** | A VLM é lida em bf16 para a RAM (~19 GB) antes de ser quantizada; o swap é rede de segurança | `free -g` |
| A pasta com as tuas fotos | Ver secção 4 | |

O `scripts/00_check_system.sh` verifica tudo isto de uma vez.

## 2. Passo a passo

**Passo 0: configurar.** Abre o [`config.env`](config.env) e muda pelo menos `DATASET_SRC` para a pasta das tuas fotos. O resto tem valores pensados para a tua máquina.

**Passo 1: verificar o sistema.**
```bash
scripts/00_check_system.sh
```
Se falhar alguma coisa, o script diz o quê e como resolver. Não avances sem estar tudo `[ok]`.

**Passo 2: criar o ambiente** (~10 min, só uma vez).
```bash
scripts/01_setup_env.sh
```
No fim faz um teste na GPU. Tens de ver `GPU ok: True`, `bf16 SDPA: ok` e `fp8 storage: ok`. Se não vires, para aqui.

**Passo 3: descarregar os modelos** (~66 GB, só uma vez, retoma se for interrompido).
```bash
scripts/02_download_models.sh
```

**Passo 4: preparar as fotos.**

Se escolheste as fotos por número, copia-as primeiro para a pasta `DATASET_SRC`:
```bash
python3 scripts/select_photos.py escolhidas.txt /media/bugss/EOS_DIGITAL/DCIM ~/fotos-lora
```

```bash
scripts/03_prepare_dataset.sh
```
Copia as fotos para `data/train/` já rodadas, redimensionadas e com nome `img_001.jpg`, `img_002.jpg`... Lê os avisos que aparecem (duplicados, fotos pequenas, desfocadas) e apaga à mão as más (o `.jpg` e o `.txt` com o mesmo nome).

**Passo 5 (opcional): pedir uma segunda opinião à VLM.**
```bash
scripts/describe.sh data/train -q "Avalia de 1 a 10 para um dataset de LoRA de identidade: nitidez da cara, luz, se a cara está tapada. Sê crítico." -o avaliacao.txt
```

**Passo 6: legendar.**
```bash
scripts/04_caption.sh
```
Escreve um `.txt` por foto, do género `afnsx man stands on a pier at sunset, wearing a white linen shirt, warm golden light`. **Depois abre os `.txt` e corrige-os.** É o passo que mais pesa na qualidade final. Para refazer tudo: `scripts/04_caption.sh --overwrite`.

**Passo 7: cache** (~5 a 10 min).
```bash
scripts/05_cache.sh
```
Passa as fotos pelo VAE e as legendas pelo text encoder, uma vez só. Tens de repetir isto sempre que mudares fotos, legendas ou `RESOLUTION`.

**Passo 8: treinar** (horas; ver secção 6).
```bash
scripts/06_train_lora.sh
```
Para acompanhar: `tensorboard --logdir logs` (com o venv ativo: `source .venv/bin/activate`) e `amdgpu_top` noutro terminal para ver a VRAM.

**Passo 9: testar os checkpoints.**
```bash
for f in output/lora_afnsx-step*.safetensors output/lora_afnsx.safetensors; do scripts/07_generate.sh "$f"; done
```
Cada checkpoint gera as mesmas 5 imagens com as mesmas seeds, em `samples/`. Compara-os lado a lado e escolhe o melhor (normalmente não é o último).

**Passo 10: levar o LoRA para o ComfyUI.**
```bash
scripts/08_export_comfyui.sh output/lora_afnsx-step00001250.safetensors ~/ComfyUI
```
Converte o checkpoint escolhido e copia-o para `~/ComfyUI/models/loras/`. Ver secção 8.

## 3. O que faz cada ficheiro

**Configuração**

| Ficheiro | O que faz |
|---|---|
| `config.env` | Único sítio onde se mexe: pasta das fotos, trigger, hiperparâmetros, block swap, modelos, versões. Todos os scripts o leem. |
| `models.env` | Gerado pelo passo 3. Guarda onde ficaram os modelos no disco. Não vai para o git. |
| `configs/test_prompts.template.txt` | Prompts usados pelo `07_generate.sh`. Inclui um prompt **sem** trigger para detetar overfit. `{TRIGGER}` e `{CLASS}` são substituídos automaticamente. |
| `configs/sample_prompts.template.txt` | Igual, mas para amostras durante o treino (desligado por defeito). |
| `configs/dataset.toml` | Gerado pelo `05_cache.sh`. Diz ao musubi onde estão as fotos, a cache e a resolução. |

**Scripts (por ordem de uso)**

| Ficheiro | O que faz |
|---|---|
| `scripts/common.sh` | Partilhado por todos: carrega o `config.env`, ativa o venv e define as variáveis de ambiente do ROCm (alocador de memória, kernels de atenção para RDNA, MIOpen rápido). Não se corre diretamente. |
| `scripts/00_check_system.sh` | Verifica ROCm, GPU gfx1201, grupos, RAM, swap e disco. Não instala nada. |
| `scripts/01_setup_env.sh` | Cria `.venv` com Python 3.12, instala o PyTorch ROCm da AMD, clona o musubi-tuner no commit fixado e instala-o sem deixar trocar o torch por um de CUDA. Acaba com um teste real na GPU. |
| `scripts/02_download_models.sh` | Descarrega o Qwen-Image 2.1 e o Qwen3.5-9B para o cache do Hugging Face e escreve o `models.env`. |
| `scripts/select_photos.py` | Copia as fotos escolhidas a partir de um txt com números (`IMG_0123.JPG` = `123`; aceita intervalos `120-130`). Usar antes do passo 4 para encher a pasta `DATASET_SRC`. |
| `scripts/03_prepare_dataset.sh` + `prepare_dataset.py` | Corrige a rotação EXIF, converte para RGB, limita o lado maior a 1536 px, deteta quase-duplicados (rajadas) com um hash perceptual, avisa sobre fotos pequenas ou desfocadas, e cria a legenda base. |
| `scripts/vlm.py` | Carrega a VLM e quantiza-a para FP8 (ver secção 5). Usado pelos dois scripts abaixo. |
| `scripts/04_caption.sh` + `caption.py` | Legenda cada foto com um prompt que descreve roupa, sítio, luz e pose, mas **não** a cara nem o corpo, e mete o trigger no sítio de "the man". Por defeito só mexe nas legendas que ainda são a base. |
| `scripts/describe.sh` + `describe.py` | VLM de uso geral: descreve uma foto ou uma pasta, responde a perguntas (`-q`), grava respostas (`-o`), ou modo interativo (`-i`) para fazer vários pedidos sem recarregar o modelo. Responde em português. |
| `scripts/05_cache.sh` | Gera o `dataset.toml` e faz a cache dos latents (VAE) e dos embeddings (text encoder Qwen3-VL-8B em fp8). |
| `scripts/06_train_lora.sh` | Treina o LoRA com o DiT em fp8, gradient checkpointing e block swap. Aceita `--resume`. |
| `scripts/07_generate.sh` | Gera imagens de teste com um checkpoint e uma força (`multiplier`) à escolha. |
| `scripts/08_export_comfyui.sh` | Converte o LoRA para o formato que o ComfyUI lê por inteiro e copia-o para lá (ver secção 8). |

## 4. Fotos: o que importa

- **20 a 40 fotos** boas valem mais do que 100 médias.
- Variedade: grandes planos, meio corpo, corpo inteiro; luz de dia e interior; roupas e fundos diferentes; expressões diferentes.
- Nada de óculos de sol, chapéus a tapar a cara, filtros, ou fotos com outras pessoas.
- As legendas descrevem o que **não** é identidade. A cara e o corpo não se descrevem: é isso que o trigger (`afnsx man`) tem de aprender.

## 5. Como tudo cabe nos 16 GB

Nunca há dois modelos grandes na GPU ao mesmo tempo:

| Fase | O que está na GPU | VRAM aprox. |
|---|---|---|
| Legendas / describe | Qwen3.5-9B em FP8 | ~12 GB |
| Cache (passo 7) | VAE, depois Qwen3-VL-8B em fp8 | ~10 GB |
| Treino | Só o DiT (7.1B) em fp8 + LoRA + ativações | ~13-15 GB com `BLOCKS_TO_SWAP=8` |
| Geração | Text encoder, depois DiT, depois VAE | ~12 GB |

**Porquê o Qwen3.5-9B:** é o modelo multimodal mais forte que cabe na tua placa com FP8. Em bf16 precisaria de ~19 GB. O degrau seguinte (Qwen3.5-27B) precisa de ~28 GB mesmo em FP8. Abaixo de FP8 não fui, como pediste.

**Como é feito o FP8:** o `vlm.py` guarda os pesos de cada camada Linear em `float8_e4m3fn`, com uma escala por canal, e converte-os para bf16 no momento de multiplicar. Medi um erro de ~2.6% nos pesos e na saída de cada camada, o mesmo que o `--fp8_scaled` do musubi. Não uso os checkpoints `-FP8` oficiais porque esses dependem de kernels CUDA/Triton que não estão garantidos em RDNA4.

## 6. Afinar

| Sintoma | O que mudar no `config.env` |
|---|---|
| Out of memory no treino | subir `BLOCKS_TO_SWAP` (12, 16...) ou `RESOLUTION=768` |
| Sobra VRAM e quero mais velocidade | baixar `BLOCKS_TO_SWAP` (4, depois 0) |
| Não se parece comigo | mais passos, ou `NETWORK_DIM=32` |
| Aparece sempre com a mesma roupa ou fundo | legendas pouco descritivas, ou checkpoint demasiado tardio |
| O prompt de controlo (sem trigger) também tem a minha cara | overfit: usa um checkpoint anterior ou baixa o `LEARNING_RATE` |

## 7. O que sai no fim

```
data/train/          img_001.jpg + img_001.txt, ...   fotos limpas e legendas
data/cache/          *.safetensors                    latents e embeddings (descartável)
output/              lora_afnsx-step00000250.safetensors
                     lora_afnsx-step00000500.safetensors
                     ...                              um checkpoint a cada 250 passos
                     lora_afnsx.safetensors           LoRA final (passo 2000)
                     lora_afnsx-step..._comfyui.safetensors   versão para o ComfyUI (passo 10)
logs/                                                 curvas de loss para o TensorBoard
samples/<checkpoint>_x1.0/   *.png                    5 imagens de teste por checkpoint
avaliacao.txt                                         se usaste o describe com -o
```

O produto final é **um ficheiro `.safetensors` com algumas dezenas de MB** (rank 16). É o LoRA que carregas no ComfyUI (ou no `07_generate.sh`) para gerar fotos com `afnsx man` no prompt.

## 8. Depois do treino: usar no ComfyUI

O ComfyUI é a interface de nós (caixas ligadas por setas) que corre localmente e abre no browser. O LoRA treinado aqui funciona lá, mas **tem de ser convertido primeiro** com o `08_export_comfyui.sh`.

Porquê: o ComfyUI carrega o Qwen-Image 2.1 com duas camadas de cada MLP fundidas numa só (`gate_up`). No formato nativo do musubi, o ComfyUI não reconhece essas duas camadas e ignora-as **sem dar erro**. O LoRA carregava, mas mais fraco (testei o mapeamento de chaves do ComfyUI: só 5 de 7 camadas por bloco entravam). Depois de convertido entram as 7.

No ComfyUI:
1. Base do Qwen-Image 2.1 em `models/diffusion_models/` (bf16 ou Q8_0 GGUF; os pesos do Hugging Face usados para treinar estão noutro formato e não servem diretamente), text encoder fp8 em `models/text_encoders/` e VAE em `models/vae/`.
2. LoRA convertido em `models/loras/`.
3. No workflow do Qwen-Image 2.1, mete um nó **LoraLoaderModelOnly** entre o loader do modelo e o sampler, com força entre 0.8 e 1.0.
4. Prompt com o trigger: `a photo of afnsx man ...`.
5. Para iterar mais depressa podes juntar um LoRA turbo do 2.1, mas como LoRA separado e nunca fundido no modelo. Para as fotos finais, tira-o e usa os ~40 passos normais.

O ComfyUI pode correr no mesmo Linux (com o PyTorch ROCm) ou no Windows. O ficheiro do LoRA é o mesmo.

## 9. Problemas conhecidos

- `hipErrorNoBinaryForGpu`: o ROCm do sistema não bate certo com as wheels (7.2.1). Atualiza o ROCm ou muda `ROCM_WHEEL_BASE` e `ROCM_WHEELS` no `config.env`.
- Erro de numpy ao importar o torch: `uv pip install "numpy<2"` com o venv ativo.
- Primeira execução lenta no VAE: o MIOpen está a compilar kernels. Já está em `MIOPEN_FIND_MODE=FAST`.
- Avisos `falling back to its reference PyTorch implementation` no Qwen3.5: normais. A atenção linear corre em PyTorch puro em vez dos kernels otimizados (feitos para CUDA). Funciona, só é mais lenta; para 40 fotos não importa. Não instales `flash-linear-attention` à força.
- Se o Qwen3.5 não funcionar de todo em ROCm: usa o plano B descrito no `config.env` (Qwen3-VL-8B que já vem com o Qwen-Image, sem download extra).
- O suporte ao Qwen-Image 2.1 no musubi-tuner é experimental e está no branch `dev`, fixado num commit no `config.env`.

## Notas

- O repo é público: `data/`, `output/`, `samples/`, imagens e pesos estão no `.gitignore`. Nunca os adiciones com `git add -f`.
- O Qwen-Image 2.1 está sob a *Qwen Research License* (uso não comercial), que se estende aos LoRAs treinados sobre ele. O Qwen3.5-9B é Apache-2.0.
