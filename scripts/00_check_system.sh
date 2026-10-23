#!/usr/bin/env bash
# Verifica se o sistema está pronto: ROCm, GPU, grupos, RAM e disco.
# Não instala nada.
source "$(dirname "$0")/common.sh"

ok=1
check() { if eval "$2" >/dev/null 2>&1; then echo "  [ok]   $1"; else echo "  [FALHA] $1"; ok=0; fi; }

log "Sistema"
check "Linux x86_64"                  '[[ "$(uname -sm)" == "Linux x86_64" ]]'
check "rocminfo instalado (ROCm)"    'command -v rocminfo'
check "GPU gfx1201 (RX 9070 XT) visível no ROCm" 'rocminfo | grep -q gfx1201'
check "utilizador no grupo render"   'id -nG | grep -qw render'
check "utilizador no grupo video"    'id -nG | grep -qw video'
check "uv instalado"                 'command -v uv'
check "git instalado"                'command -v git'

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

free_gb=$(df -BG --output=avail "$REPO_ROOT" | tail -1 | tr -dc '0-9')
echo "  Disco livre aqui: ${free_gb} GB (modelos ~55 GB + venv ~10 GB + cache)"
(( free_gb >= 80 )) || warn "Pouco espaço. O HF cache fica em \${HF_HOME:-~/.cache/huggingface}; muda HF_HOME se precisares."

echo
if (( ok )); then
  log "Tudo certo. Próximo: scripts/01_setup_env.sh"
else
  cat <<'EOF'
Algo falhou. Pistas:
  - ROCm: instala com o amdgpu-install da AMD (ROCm 7.2.x, Ubuntu 24.04 é o caminho mais testado).
  - Grupos: sudo usermod -aG render,video "$USER"  e depois reinicia a sessão.
  - uv:    curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 1
fi
