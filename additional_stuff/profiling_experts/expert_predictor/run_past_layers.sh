#!/usr/bin/env bash
# Thin wrapper: run the past-layer (cross-iteration history) experiment.
# Sweeps window N=1..24. All args are forwarded to run_seq.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_seq.sh" --mode layers "$@"
