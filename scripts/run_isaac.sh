#!/bin/bash
# Run an Isaac Lab script from the repo root, detached-friendly: logs to a file and appends the exit code.
#   setsid nohup scripts/run_isaac.sh <log> <script.py> [args...] > /dev/null 2>&1 < /dev/null &
log=$1; shift
cd "$(dirname "$0")/.." || exit 1
export OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD
source "${CONDA_ROOT:-/media/storage/ayush/miniconda3}/etc/profile.d/conda.sh"
conda activate isaaclab
"${ISAACLAB:-/media/storage/ayush/IsaacLab}/isaaclab.sh" -p "$@" > "$log" 2>&1
echo "EXIT $?" >> "$log"
