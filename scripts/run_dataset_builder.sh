#!/usr/bin/env bash
set -euo pipefail

# Discover repositories and save dataset
python dataset_builder.py discover \
  --name python-webapps \
  --language Python \
  --topic webapp \
  --max-stars 50 \
  --limit 50

# Convert dataset to experiment configuration
python dataset_builder.py to-experiment \
  --dataset evaluation_reports/datasets/python-webapps.json \
  --output experiments/python-webapps-experiment.yaml \
  --name python-webapps-evaluation \
  --description "Evaluate models on Python web application repositories" \
  --models config/models.yaml \
  --top-n 10 \
  --repetitions 1 \
  --cleanup per_run
