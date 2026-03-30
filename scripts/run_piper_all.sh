#!/bin/bash
# Run all Piper experiments (separate from OpenArm).
#
# Usage:
#   ./scripts/run_piper_all.sh           # full run
#   ./scripts/run_piper_all.sh --dry-run # theory only
#
# Prerequisites:
#   source scripts/setup_env.sh
#   ./scripts/setup_env.sh init

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${LEROBOT_PY:-python3}"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run"

# Piper follower default port
PIPER_PORT="${PIPER_FOLLOWER:-can2}"

echo "=========================================="
echo " Piper 6-DOF Experiment Suite"
echo " $(date)"
echo " Port: $PIPER_PORT"
echo " Dry-run: ${DRY_RUN:-no}"
echo "=========================================="

# ── Preflight ────────────────────────────────────────────────────────
if [ -z "$DRY_RUN" ]; then
    echo ""
    echo ">>> Preflight check"
    $PYTHON scripts/preflight.py --robot piper --port "$PIPER_PORT"
fi

# ── Experiment 4: Single-arm coupling (4 configs × 6 joints) ────────
echo ""
echo ">>> Experiment 4: Piper single-arm coupling (all configs)"
$PYTHON scripts/run_piper_coupling.py \
    --port "$PIPER_PORT" --config all $DRY_RUN

# ── Experiment 5: Elbow sweep ────────────────────────────────────────
echo ""
echo ">>> Experiment 5: Piper elbow configuration sweep"
$PYTHON scripts/run_piper_sweep.py \
    --port "$PIPER_PORT" $DRY_RUN

echo ""
echo "=========================================="
echo " Piper experiments complete!"
echo " Results: $ROOT/results/piper_*.json"
echo "=========================================="
