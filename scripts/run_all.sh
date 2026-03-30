#!/bin/bash
# Run all BinaryStars_Real experiments in sequence.
#
# Usage:
#   ./scripts/run_all.sh           # full run (connects to robot)
#   ./scripts/run_all.sh --dry-run # theoretical predictions only
#
# Prerequisites:
#   source scripts/setup_env.sh
#   ./scripts/setup_env.sh init    # initialise CAN interfaces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

PYTHON="${LEROBOT_PY:-python3}"
DRY_RUN=""
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN="--dry-run"

echo "=========================================="
echo " BinaryStars_Real Experiment Suite"
echo " $(date)"
echo " Python: $PYTHON"
echo " Dry-run: ${DRY_RUN:-no}"
echo "=========================================="

# ── Step 0: Preflight ────────────────────────────────────────────────
if [ -z "$DRY_RUN" ]; then
    echo ""
    echo ">>> Step 0: Preflight check"
    $PYTHON scripts/preflight.py --robot openarm --port can0 --side right
    echo "    Preflight OK"
fi

# ── Step 1: Single-arm coupling (5 configs × 7 joints) ──────────────
echo ""
echo ">>> Step 1: OpenArm single-arm coupling (all configs)"
$PYTHON scripts/run_openarm_coupling.py --config all --port can0 --side right $DRY_RUN

# ── Step 2: Dual-arm Lemma 3 ────────────────────────────────────────
if [ -z "$DRY_RUN" ]; then
    echo ""
    echo ">>> Step 2: Dual-arm Lemma 3 verification"
    $PYTHON scripts/run_dual_arm_lemma3.py --right-port can0 --left-port can1
fi

# ── Step 3: Elbow configuration sweep ───────────────────────────────
echo ""
echo ">>> Step 3: Elbow configuration sweep"
$PYTHON scripts/run_openarm_sweep.py --port can0 --side right $DRY_RUN

# ── Step 4: Piper (optional, position-only) ─────────────────────────
# Uncomment if Piper is connected:
# echo ""
# echo ">>> Step 4: Piper single-arm coupling"
# $PYTHON scripts/run_piper_coupling.py --config all $DRY_RUN

echo ""
echo "=========================================="
echo " All experiments complete!"
echo " Results saved to: $ROOT/results/"
echo "=========================================="
