#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <experiment_config.yaml> [additional python args...]" >&2
  exit 1
fi

CONFIG_PATH="$1"
shift || true

python run_experiments.py --config "$CONFIG_PATH" "$@"
