#!/usr/bin/env bash
# Cria o venv (Python 3.12), instala PyTorch ROCm da AMD e o musubi-tuner
# fixado no commit com suporte ao Qwen-Image 2.1.
source "$(dirname "$0")/common.sh"

command -v uv  >/dev/null || die "Instala o uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git >/dev/null || die "Falta o git"

# ---------- venv ----------
if [[ ! -d "$VENV_DIR" ]]; then
  log "A criar venv Python 3.12 em .venv"
  uv venv --python 3.12 "$VENV_DIR"
fi
activate_venv

# ---------- PyTorch ROCm ----------
if python -c 'import torch,sys; sys.exit(0 if torch.version.hip else 1)' 2>/dev/null; then
  log "PyTorch ROCm já instalado: $(python -c 'import torch; print(torch.__version__)')"
else
  log "A instalar PyTorch ROCm (wheels AMD)"
  wheel_dir="$REPO_ROOT/.wheels"
  mkdir -p "$wheel_dir"
  local_wheels=()
  for w in "${ROCM_WHEELS[@]}"; do
    file="${w//%2B/+}"
    [[ -f "$wheel_dir/$file" ]] || curl -fL --retry 3 -o "$wheel_dir/$file" "$ROCM_WHEEL_BASE/$w"
    local_wheels+=("$wheel_dir/$file")
  done
  uv pip install "${local_wheels[@]}"
  rm -rf "$wheel_dir"   # ~5 GB que já não fazem falta
fi

# ---------- musubi-tuner ----------
if [[ ! -d "$MUSUBI_DIR/.git" ]]; then
  log "A clonar musubi-tuner ($MUSUBI_BRANCH)"
  git clone --branch "$MUSUBI_BRANCH" "$MUSUBI_REPO" "$MUSUBI_DIR"
fi
git -C "$MUSUBI_DIR" fetch --quiet origin "$MUSUBI_BRANCH"
git -C "$MUSUBI_DIR" checkout --quiet "$MUSUBI_COMMIT"
log "musubi-tuner em $(git -C "$MUSUBI_DIR" log -1 --format='%h %s')"

# Instala o musubi sem deixar o resolver trocar o torch ROCm por um CUDA.
torch_pin="torch==$(python -c 'import torch; print(torch.__version__)')"
log "A instalar dependências do musubi-tuner (mantendo $torch_pin)"
uv pip install -e "$MUSUBI_DIR" "$torch_pin"

# Logs do treino. (O CLI `hf` já vem com o huggingface-hub do musubi.)
uv pip install tensorboard "$torch_pin"
command -v hf >/dev/null || die "CLI 'hf' não encontrado no venv"

# ---------- accelerate ----------
python - <<'EOF'
from accelerate.utils import write_basic_config
write_basic_config(mixed_precision="bf16")
EOF

# ---------- verificação ----------
log "Verificação final"
python - <<'EOF'
import torch
print("torch      :", torch.__version__, "| HIP", torch.version.hip)
print("GPU ok     :", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("dispositivo:", p.name, getattr(p, "gcnArchName", ""), f"{p.total_memory/2**30:.1f} GiB")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    q = torch.nn.functional.scaled_dot_product_attention(x[None,None], x[None,None], x[None,None])
    f8 = x.to(torch.float8_e4m3fn).to(torch.bfloat16)
    torch.cuda.synchronize()
    print("bf16 SDPA  : ok", tuple(q.shape))
    print("fp8 storage: ok (erro médio %.4f)" % (f8 - x).abs().mean().item())
EOF

log "Pronto. Próximo: scripts/02_download_models.sh"
