#!/usr/bin/env bash
# run.sh - un solo comando: deps -> personas -> build de todo -> launch
set +u
export DEBIAN_FRONTEND=noninteractive

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$DIR"
while [ "$WS" != "/" ] && [ ! -f "$WS/install/setup.bash" ] && [ ! -d "$WS/src" ]; do WS="$(dirname "$WS")"; done
[ -d "$WS/src" ] || WS="${HRI_WS:-$HOME/hri_vision_ws}"

PEOPLE_DIR="${HRI_PEOPLE_DIR:-$HOME/hri_people}"
BACKUPS="$HOME/hri_people_backups"
B=$'\e[1m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; N=$'\e[0m'

# --- root o sudo (en Docker normalmente eres root y no hay sudo) ---
if [ "$(id -u)" -eq 0 ]; then SUDO=""; else SUDO="sudo"; fi

ensure_ros() {
  [ -f /opt/ros/jazzy/setup.bash ] && return 0
  echo "${B}[ros]${N} ROS 2 Jazzy no está instalado. Instalando (varios minutos)..."
  . /etc/os-release
  if [ "$VERSION_CODENAME" != "noble" ]; then
    echo "${R}✘ Se necesita Ubuntu 24.04 (noble) para ROS 2 Jazzy; esto es $VERSION_CODENAME.${N}"; exit 1
  fi
  $SUDO apt-get update -qq
  $SUDO apt-get install -y software-properties-common curl ca-certificates gnupg lsb-release zstd || exit 1
  $SUDO add-apt-repository -y universe
  local ver
  ver=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F\" '{print $4}')
  [ -n "$ver" ] || { echo "${R}✘ No pude consultar la versión de ros-apt-source (¿sin internet?)${N}"; exit 1; }
  curl -fL -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ver}/ros2-apt-source_${ver}.${VERSION_CODENAME}_all.deb" || exit 1
  $SUDO dpkg -i /tmp/ros2-apt-source.deb || exit 1
  $SUDO apt-get update -qq
  $SUDO apt-get install -y ros-jazzy-ros-base ros-dev-tools || { echo "${R}✘ Falló la instalación de ROS${N}"; exit 1; }
  echo "  ${G}✔${N} ROS 2 Jazzy instalado"
}

build_all() {
  echo "${B}[build]${N} Compilando todos los paquetes..."
  (cd "$WS" && colcon build 2>&1 | grep -v "WARNING:colcon\|underlay\|override\|^$\|If you understand\|may be promoted")
  if [ ! -f "$WS/install/setup.bash" ]; then
    echo "${R}✘ Falló el build.${N}"; exit 1
  fi
  source "$WS/install/setup.bash"
}

install_deps() {
  local req="$DIR/requirements.txt"
  [ -f "$req" ] || { echo "${R}✘ No encuentro $req${N}"; exit 1; }
  local mark="$HOME/.hri_deps_$(md5sum "$req" | cut -c1-8)"
  [ -f "$mark" ] && { echo "  ${G}✔${N} Dependencias al día"; return; }

  echo "${B}[deps]${N} Instalando dependencias (primera vez, tarda varios minutos)..."
  $SUDO apt-get update -qq
  $SUDO apt-get install -y python3-pip python3-dev build-essential portaudio19-dev \
      curl ca-certificates zstd nano python3-opencv ros-jazzy-cv-bridge ros-jazzy-vision-msgs \
      || { echo "${R}✘ apt falló${N}"; exit 1; }

  python3 -m pip install --break-system-packages --ignore-installed -r "$req" || { echo "${R}✘ pip falló${N}"; exit 1; }
  python3 -m pip install --break-system-packages --no-deps mediapipe==0.10.18
  python3 -m pip install --break-system-packages absl-py attrs flatbuffers matplotlib

  # Quitar cualquier opencv de pip para que gane el del sistema (cv_bridge)
  python3 -m pip uninstall -y --break-system-packages opencv-python opencv-python-headless opencv-contrib-python 2>/dev/null

  # Verificación
  python3 - <<'PY' || { echo "${R}✘ Entorno inválido${N}"; exit 1; }
import cv2, numpy
assert cv2.__file__.startswith('/usr/lib'), f'cv2 incorrecto: {cv2.__file__}'
assert numpy.__version__.startswith('1.'), f'numpy {numpy.__version__}'
import mediapipe, insightface, chromadb
print('entorno OK')
PY
  touch "$mark"
}

ensure_ollama() {
  if ! command -v ollama >/dev/null; then
    echo "${B}[ollama]${N} Instalando Ollama..."
    curl -fsSL https://ollama.com/install.sh | $SUDO sh || echo "  ${Y}…${N} No se pudo instalar Ollama (el RAG usará solo retrieval)"
  fi
  command -v ollama >/dev/null || return 0
  if curl -s --max-time 2 http://localhost:11434 >/dev/null 2>&1; then
    echo "  ${G}✔${N} Ollama activo"
  else
    echo "  ${Y}…${N} Iniciando Ollama"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 15); do
      curl -s --max-time 1 http://localhost:11434 >/dev/null 2>&1 && break
      sleep 1
    done
  fi
  ollama list 2>/dev/null | grep -q "qwen2.5:1.5b" || ollama pull qwen2.5:1.5b
}

[ -t 1 ] && clear
echo "${B}=====================================================${N}"
echo "${B}   HRI / Visión - Reto Avanzado Candidates${N}"
echo "${B}=====================================================${N}"
echo "  Workspace: $WS"

ensure_ros
source /opt/ros/jazzy/setup.bash
install_deps
ensure_ollama
mkdir -p "$PEOPLE_DIR"
echo

# Compilar siempre antes del menu (incremental) para que el codigo nuevo este instalado
build_all

# 1) Demo desde cero (opcional)
read -p "¿Empezar desde cero? Respalda y vacía personas y base RAG [s/N]: " a
if [[ "$a" =~ ^[sS]$ ]]; then
  ts=$(date +%Y%m%d_%H%M%S)
  mkdir -p "$BACKUPS"
  [ -d "$PEOPLE_DIR" ] && mv "$PEOPLE_DIR" "$BACKUPS/people_$ts"
  [ -d "$HOME/rag_db" ] && mv "$HOME/rag_db" "$BACKUPS/rag_db_$ts"
  mkdir -p "$PEOPLE_DIR"
  echo "${G}Listo.${N} Respaldo en $BACKUPS"
fi

# 2) Personas (menú interactivo; 0 para salir y continuar)
echo
echo "${B}[personas]${N} Registra o edita personas. Elige 0 para continuar."
"$DIR/people.sh"

# 3) Build de TODO
echo
build_all

# 4) Launch
echo
echo "${B}[launch]${N} Iniciando el sistema (Ctrl+C para detener)"
pkill -f rag_service_node 2>/dev/null
sleep 1
(sleep 15; xdg-open http://localhost:8080 >/dev/null 2>&1) &
ros2 launch identity_fusion identity_fusion.launch.py