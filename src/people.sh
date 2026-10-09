#!/usr/bin/env bash
# Manage enrolled people. Usage: ./people.sh <command> ...  (./people.sh -h)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${HRI_WS:-}"
if [ -z "$WS" ]; then
  d="$DIR"
  while [ "$d" != "/" ]; do
    if [ -f "$d/install/setup.bash" ]; then WS="$d"; break; fi
    d="$(dirname "$d")"
  done
fi
[ -n "$WS" ] || { echo "No encontré el workspace (install/setup.bash). Compila o define HRI_WS."; exit 1; }
source /opt/ros/jazzy/setup.bash
source "$WS/install/setup.bash"
exec python3 -m hri_common.cli "$@"