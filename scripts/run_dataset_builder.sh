#!/usr/bin/env bash
set -euo pipefail

# Scripted dataset creation for H1 validation (autonomous generation on real web apps).

DATASET_NAME="h1-webapps"
DATASET_PATH="evaluation_reports/datasets/${DATASET_NAME}.json"
MODELS_FILE="${MODELS_FILE:-config/models.yaml}"
EXPERIMENT_PATH="experiments/${DATASET_NAME}-experiment.yaml"

# Discover candidate repositories (recent web apps across popular languages)
python dataset_builder.py discover \
  --name "${DATASET_NAME}" \
  --query "webapp pushed:>=2025-01-01" \
  --language Python \
  --language JavaScript \
  --language TypeScript \
  --topic webapp \
  --min-stars 0 \
  --max-stars 10 \
  --limit 10 \
  --sort pushed \
  --order desc \
  --note "H1 dataset: recent web applications for functional validation" \
  --fetch-multiple 4 \
  --shuffle

## Convert dataset into an experiment definition focused on H1 validation
python dataset_builder.py to-experiment \
  --dataset "${DATASET_PATH}" \
  --output "${EXPERIMENT_PATH}" \
  --name "${DATASET_NAME}-evaluation" \
  --description "H1 validation on real-world Dockerized web apps" \
  --models "${MODELS_FILE}" \
  --top-n 20 \
  --repetitions 2 \
  --cleanup per_run
