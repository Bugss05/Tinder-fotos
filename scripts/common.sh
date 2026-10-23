#!/usr/bin/env bash
# Carregado por todos os outros scripts. Não corras isto diretamente.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=../config.env
source "$REPO_ROOT/config.env"

MUSUBI_DIR="$REPO_ROOT/musubi-tuner"
VENV_DIR="$REPO_ROOT/.venv"
MODELS_ENV="$REPO_ROOT/models.env"

# ---------- Ambiente ROCm ----------
# (expandable_segments não é suportado nesta versão do ROCm: só gerava um aviso.)
# Ativa os kernels AOTriton (flash / mem-efficient SDPA) em RDNA.
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL="${TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL:-1}"
# Evita que o MIOpen passe minutos a fazer tuning das convoluções do VAE.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

log()  { printf '\033[1;36m[%s]\033[0m %s\n' "$(basename "$0")" "$*"; }
warn() { printf '\033[1;33m[aviso]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[erro]\033[0m %s\n' "$*" >&2; exit 1; }

activate_venv() {
  [[ -f "$VENV_DIR/bin/activate" ]] || die "Falta o venv. Corre primeiro: scripts/01_setup_env.sh"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

load_models() {
  [[ -f "$MODELS_ENV" ]] || die "Falta models.env. Corre primeiro: scripts/02_download_models.sh"
  # shellcheck disable=SC1090
  source "$MODELS_ENV"
  [[ -d "$QWEN_DIR/transformer" ]] || die "Modelo não encontrado em $QWEN_DIR"
  DIT_PATH="$QWEN_DIR/transformer"
  VAE_PATH="$QWEN_DIR/vae"
  TE_PATH="$QWEN_DIR/text_encoder"
}

load_vlm_dir() {
  [[ -f "$MODELS_ENV" ]] || die "Falta models.env. Corre primeiro: scripts/02_download_models.sh"
  # shellcheck disable=SC1090
  source "$MODELS_ENV"
  [[ -n "${VLM_DIR:-}" && -d "$VLM_DIR" ]] || die "VLM em falta. Corre scripts/02_download_models.sh (sem --no-vlm)."
}
