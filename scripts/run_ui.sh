#!/usr/bin/env bash
set -euo pipefail

STREAMLIT_CMD=${STREAMLIT_CMD:-streamlit}
SCRIPT_PATH="ui/experiment_dashboard.py"

if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Cannot find $SCRIPT_PATH. Run from repository root." >&2
  exit 1
fi

exec "$STREAMLIT_CMD" run "$SCRIPT_PATH" "$@"
