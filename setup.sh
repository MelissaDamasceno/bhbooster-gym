#!/usr/bin/env bash
#
# setup_runpod.sh — Prepara um container "pelado" (RunPod / Ubuntu) para TREINAR
# a caminhada do Booster Gym (Isaac Gym + PyTorch CUDA), do zero.
#
# Pressupostos:
#   - Container Linux x86_64 com GPU NVIDIA e driver já instalado (padrão RunPod GPU).
#   - NÃO existe conda/mamba/python/pip pré-instalados (instala tudo).
#
# O Isaac Gym (Preview 4) NÃO pode ser baixado sem login na NVIDIA, então você
# precisa fornecer o tarball de uma destas formas (a 1ª encontrada vence):
#   1) ./setup_runpod.sh /caminho/IsaacGym_Preview_4_Package.tar.gz
#   2) export ISAACGYM_TARBALL=/caminho/IsaacGym_Preview_4_Package.tar.gz
#   3) export ISAACGYM_URL=https://...   (link direto/pré-assinado seu, ex. S3)
#   4) Deixar IsaacGym_Preview_4_Package.tar.gz na raiz do repo.
#   5) Já ter a pasta ./isaacgym extraída.
#
# Uso:
#   bash setup_runpod.sh [caminho-do-tarball-do-isaacgym]
#
# Depois, em cada novo shell, ative o ambiente com:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate boostergym
# e treine:
#   python train.py --task=T1 --headless
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV_NAME="${ENV_NAME:-boostergym}"
PY_VERSION="3.8"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACGYM_TARBALL="${1:-${ISAACGYM_TARBALL:-}}"
ISAACGYM_URL="${ISAACGYM_URL:-}"

log()  { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }
warn() { printf '\n\033[1;33m[!] %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[x] %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Sanity / GPU
# ---------------------------------------------------------------------------
log "Verificando GPU NVIDIA"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || warn "nvidia-smi falhou; o treino exige GPU NVIDIA."
else
    warn "nvidia-smi não encontrado. Isaac Gym precisa de GPU NVIDIA com driver."
fi

# ---------------------------------------------------------------------------
# 1. Pacotes de sistema (apt)
# ---------------------------------------------------------------------------
log "Instalando dependências de sistema (apt)"
export DEBIAN_FRONTEND=noninteractive
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
    wget curl ca-certificates git bzip2 xz-utils \
    build-essential \
    libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libosmesa6 libgomp1 ffmpeg
$SUDO rm -rf /var/lib/apt/lists/* || true

# ---------------------------------------------------------------------------
# 2. Miniconda
# ---------------------------------------------------------------------------
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    log "Instalando Miniconda em $CONDA_DIR"
    tmp_installer="$(mktemp --suffix=.sh)"
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$tmp_installer"
    bash "$tmp_installer" -b -p "$CONDA_DIR"
    rm -f "$tmp_installer"
else
    log "Miniconda já presente em $CONDA_DIR"
fi

# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda config --set always_yes yes --set changeps1 no

# ---------------------------------------------------------------------------
# 3. Ambiente conda + PyTorch/CUDA + numpy
# ---------------------------------------------------------------------------
if conda env list | grep -qE "^\s*$ENV_NAME\s"; then
    log "Ambiente conda '$ENV_NAME' já existe"
else
    log "Criando ambiente conda '$ENV_NAME' (Python $PY_VERSION)"
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi

conda activate "$ENV_NAME"

log "Instalando numpy 1.21.6 + PyTorch 2.0 (CUDA 11.8)"
conda install -n "$ENV_NAME" -y \
    numpy=1.21.6 pytorch=2.0 pytorch-cuda=11.8 \
    -c pytorch -c nvidia

# ---------------------------------------------------------------------------
# 4. Isaac Gym
# ---------------------------------------------------------------------------
ISAACGYM_DIR="$REPO_DIR/isaacgym"

if [ ! -d "$ISAACGYM_DIR/python" ]; then
    # Resolve tarball
    if [ -z "$ISAACGYM_TARBALL" ]; then
        if [ -f "$REPO_DIR/IsaacGym_Preview_4_Package.tar.gz" ]; then
            ISAACGYM_TARBALL="$REPO_DIR/IsaacGym_Preview_4_Package.tar.gz"
        elif [ -n "$ISAACGYM_URL" ]; then
            log "Baixando Isaac Gym de ISAACGYM_URL"
            ISAACGYM_TARBALL="$REPO_DIR/IsaacGym_Preview_4_Package.tar.gz"
            wget -q "$ISAACGYM_URL" -O "$ISAACGYM_TARBALL"
        fi
    fi

    [ -n "$ISAACGYM_TARBALL" ] && [ -f "$ISAACGYM_TARBALL" ] || die \
"Isaac Gym não encontrado.
   Baixe 'IsaacGym_Preview_4_Package.tar.gz' em https://developer.nvidia.com/isaac-gym/download
   (precisa de login NVIDIA) e rode:
       bash setup_runpod.sh /caminho/IsaacGym_Preview_4_Package.tar.gz
   ou:  export ISAACGYM_URL=<link-direto-seu> && bash setup_runpod.sh"

    log "Extraindo Isaac Gym ($ISAACGYM_TARBALL)"
    tar -xzf "$ISAACGYM_TARBALL" -C "$REPO_DIR"
else
    log "Isaac Gym já extraído em $ISAACGYM_DIR"
fi

log "Instalando Isaac Gym (pip -e)"
pip install -e "$ISAACGYM_DIR/python"

# ---------------------------------------------------------------------------
# 5. LD_LIBRARY_PATH (libpython3.8) via activate.d
# ---------------------------------------------------------------------------
log "Configurando LD_LIBRARY_PATH no ambiente conda"
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"
cat > "$ACT_DIR/env_vars.sh" <<'EOF'
export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
EOF
cat > "$DEACT_DIR/env_vars.sh" <<'EOF'
export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH:-}
unset OLD_LD_LIBRARY_PATH
EOF
# aplica já nesta sessão
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$CONDA_PREFIX/lib"

# ---------------------------------------------------------------------------
# 6. Dependências Python do projeto
# ---------------------------------------------------------------------------
log "Instalando dependências do requirements.txt"
pip install -r "$REPO_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 7. Smoke test
# ---------------------------------------------------------------------------
log "Verificando instalação"
python - <<'PY'
import torch, isaacgym, numpy
print("torch        :", torch.__version__)
print("CUDA disp.   :", torch.cuda.is_available())
print("GPU          :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
print("numpy        :", numpy.__version__)
print("isaacgym     : ok")
PY

cat <<EOF

============================================================
 Ambiente pronto.

 Em cada novo shell, ative o ambiente:
   source $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME

 Treinar a caminhada (headless, recomendado no RunPod):
   cd $REPO_DIR
   python train.py --task=T1 --headless

 Acompanhar (TensorBoard):
   tensorboard --logdir logs --bind_all
============================================================
EOF
