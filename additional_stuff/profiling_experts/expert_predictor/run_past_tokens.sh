#!/usr/bin/env bash
# Thin wrapper: run the past-token (same-layer history) experiment.
# Sweeps window W=1..10. All args are forwarded to run_seq.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_seq.sh" --mode tokens "$@"
