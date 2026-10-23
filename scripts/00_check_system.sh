#!/usr/bin/env bash
# Verifica se o sistema está pronto: ROCm, GPU, grupos, RAM e disco.
# Não instala nada.
source "$(dirname "$0")/common.sh"

ok=1
check() { if eval "$2" >/dev/null 2>&1; then echo "  [ok]   $1"; else echo "  [FALHA] $1"; ok=0; fi; }

log "Sistema"
check "Linux x86_64"                  '[[ "$(uname -sm)" == "Linux x86_64" ]]'
check "rocminfo instalado (ROCm)"    'command -v rocminfo'
# Guarda o output primeiro: com `set -o pipefail`, `rocminfo | grep -q` falha
# por SIGPIPE quando o grep sai cedo, mesmo que a GPU lá esteja.
rocm_out=$(rocminfo 2>&1 || true)
check "GPU gfx1201 (RX 9070 XT) visível no ROCm" 'grep -q gfx1201 <<<"$rocm_out"'
gpus=$(grep -o 'gfx[0-9a-f]\+' <<<"$rocm_out" | grep -vx 'gfx[0-9]\{2\}' | sort -u | tr '\n' ' ')  # ignora 'gfx12' de 'gfx12-generic'
echo "  GPUs vistas pelo ROCm: ${gpus:-nenhuma}"
if [[ -z "$gpus" ]]; then
  echo "  diagnóstico:"
  [[ -e /dev/kfd ]] && echo "    /dev/kfd existe" || echo "    /dev/kfd NÃO existe (driver amdgpu/KFD não carregado)"
  lsmod | grep -q '^amdgpu' && echo "    módulo amdgpu carregado" || echo "    módulo amdgpu NÃO carregado"
  echo "    kernel: $(uname -r)  (RDNA4 precisa de >= 6.12 ou do amdgpu-dkms do ROCm)"
  grep -m3 -i 'error\|permission\|denied' <<<"$rocm_out" | sed 's/^/    rocminfo: /'
fi
check "utilizador no grupo render"   'id -nG | grep -qw render'
check "utilizador no grupo video"    'id -nG | grep -qw video'
check "uv instalado"                 'command -v uv'
check "git instalado"                'command -v git'
check "curl instalado"               'command -v curl'

if command -v rocminfo >/dev/null; then
  rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null || echo "?")
  echo "  versão ROCm: $rocm_ver (wheels configuradas para 7.2.x)"
fi

ram_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "  RAM: ${ram_gb} GB"
(( ram_gb >= 30 )) || { warn "Menos de 32 GB de RAM: block swap e o load do DiT vão sofrer."; }

swap_gb=$(awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
echo "  Swap: ${swap_gb} GB"
(( swap_gb >= 16 )) || warn "Recomendo >=16 GB de swap como rede de segurança no carregamento dos modelos."

existing() { local p="$1"; while [[ ! -e "$p" ]]; do p=$(dirname "$p"); done; echo "$p"; }
free_at() { df -BG --output=avail "$(existing "$1")" | tail -1 | tr -dc '0-9'; }
dev_of()  { df --output=source "$(existing "$1")" | tail -1; }
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
repo_gb=$(free_at "$REPO_ROOT"); hf_gb=$(free_at "$hf_home")
echo "  Disco livre no repo: ${repo_gb} GB (venv ~10 GB + cache + checkpoints; precisa de >=20)"
echo "  Disco livre para modelos ($hf_home): ${hf_gb} GB (Qwen-Image ~47 GB + VLM ~19 GB)"
(( repo_gb >= 20 )) || warn "Pouco espaço para o venv e os checkpoints."
if [[ "$(dev_of "$REPO_ROOT")" == "$(dev_of "$hf_home")" ]]; then
  (( hf_gb >= 100 )) || warn "Modelos e venv no mesmo disco com <100 GB livres. Aponta HF_HOME para outro disco."
else
  (( hf_gb >= 75 )) || warn "Pouco espaço para os modelos em $hf_home."
fi
[[ -w "$(existing "$hf_home")" ]] || warn "Sem permissão de escrita em $(existing "$hf_home") (HF_HOME=$hf_home)."

echo
if (( ok )); then
  log "Tudo certo. Próximo: scripts/01_setup_env.sh"
else
  cat <<'EOF'
Algo falhou. Pistas:
  - ROCm: instala com o amdgpu-install da AMD (ROCm 7.2.x, Ubuntu 24.04 é o caminho mais testado).
  - Grupos: sudo usermod -aG render,video "$USER"  e depois reinicia a sessão.
  - curl:  sudo apt install curl   (não uses o snap)
  - uv:    curl -LsSf https://astral.sh/uv/install.sh | sh   e depois: source ~/.local/bin/env
EOF
  exit 1
fi
