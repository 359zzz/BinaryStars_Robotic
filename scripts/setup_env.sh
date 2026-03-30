#!/bin/bash
# BinaryStars_Real IPC environment setup.
#
# Source this before running experiments:
#   source scripts/setup_env.sh
#
# Or run experiments directly:
#   ./scripts/setup_env.sh run scripts/run_openarm_coupling.py --config home

set -euo pipefail

# ── CAN interface names ──────────────────────────────────────────────
export OLD_RIGHT=can0   # Right follower arm
export OLD_LEFT=can1    # Left follower arm
export NEW_RIGHT=can2   # Right leader arm
export NEW_LEFT=can3    # Left leader arm

# ── Python environment ───────────────────────────────────────────────
export LEROBOT_PY=/home/whz/miniconda/envs/lerobot/bin/python
export PYTHONPATH="/home/whz/下载/Evo-RL/src${PYTHONPATH:+:$PYTHONPATH}"

# ── Project root (auto-detect from this script's location) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BSREAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="${BSREAL_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# ── CAN bus initialisation ───────────────────────────────────────────
init_can() {
    echo "Initialising CAN FD interfaces..."
    for IF in ${OLD_RIGHT} ${OLD_LEFT} ${NEW_RIGHT} ${NEW_LEFT}; do
        if ip link show "$IF" &>/dev/null; then
            sudo ip link set "$IF" down 2>/dev/null || true
            sudo ip link set "$IF" type can bitrate 1000000 dbitrate 5000000 fd on
            sudo ip link set "$IF" up
            echo "  $IF: UP (1M/5M FD)"
        else
            echo "  $IF: not found, skipping"
        fi
    done
}

# ── CAN status check ────────────────────────────────────────────────
check_can() {
    echo "CAN interface status:"
    for IF in ${OLD_RIGHT} ${OLD_LEFT}; do
        if ip link show "$IF" &>/dev/null; then
            STATE=$(ip -d link show "$IF" | grep -oP 'state \K\S+' || echo "unknown")
            echo "  $IF: $STATE"
        else
            echo "  $IF: NOT FOUND"
        fi
    done
}

# ── Run a Python script with correct environment ────────────────────
run() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 run <script.py> [args...]"
        exit 1
    fi
    echo "Running: $LEROBOT_PY $@"
    echo "PYTHONPATH=$PYTHONPATH"
    exec $LEROBOT_PY "$@"
}

# ── Main ─────────────────────────────────────────────────────────────
case "${1:-}" in
    init)
        init_can
        ;;
    check)
        check_can
        ;;
    run)
        shift
        run "$@"
        ;;
    "")
        # Sourced — just export variables
        echo "BinaryStars_Real environment loaded."
        echo "  BSREAL_ROOT=$BSREAL_ROOT"
        echo "  LEROBOT_PY=$LEROBOT_PY"
        echo "  PYTHONPATH=$PYTHONPATH"
        ;;
    *)
        echo "Usage:"
        echo "  source $0           # export environment variables"
        echo "  $0 init             # initialise CAN FD interfaces (needs sudo)"
        echo "  $0 check            # check CAN interface status"
        echo "  $0 run <script> ... # run a Python script with correct env"
        ;;
esac
