#!/usr/bin/env bash
set -euo pipefail

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "[deploy] Authenticating with local ZenML server..."
if ! uv run zenml logout --local >/dev/null 2>&1; then
  echo "[deploy] No existing session to log out from."
fi
uv run zenml login --local

echo "[deploy] Ensuring MLflow experiment tracker exists..."
if ! uv run zenml experiment-tracker describe mlflow_tracker >/dev/null 2>&1; then
  uv run zenml experiment-tracker register mlflow_tracker \
    --flavor=mlflow \
    --tracking_uri='file:./mlruns'
fi

echo "[deploy] Ensuring MLflow model deployer exists..."
if ! uv run zenml model-deployer describe mlflow >/dev/null 2>&1; then
  uv run zenml model-deployer register mlflow --flavor=mlflow
fi

echo "[deploy] Configuring local_mlflow_stack..."
if ! uv run zenml stack describe local_mlflow_stack >/dev/null 2>&1; then
  uv run zenml stack register local_mlflow_stack \
    -o default \
    -a default \
    -e mlflow_tracker \
    -d mlflow \
    --set
else
  uv run zenml stack update local_mlflow_stack \
    -e mlflow_tracker \
    -d mlflow
  uv run zenml stack set local_mlflow_stack
fi

uv run zenml stack describe

TEST_CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/deployment-config.XXXXXX")"
trap 'rm -f "${TEST_CONFIG_FILE}"' EXIT

echo "[deploy] Creating test configuration at ${TEST_CONFIG_FILE}..."
TEST_CONFIG_FILE="${TEST_CONFIG_FILE}" uv run python - <<'PY'
import os
from pathlib import Path

import yaml

base = Path("config.yaml")
config = yaml.safe_load(base.read_text())

config["train"]["epochs"] = 1
config["evaluation"]["min_precision"] = 0.05
config["evaluation"]["min_recall"] = 0.05
config["evaluation"]["min_f1"] = 0.05
config["evaluation"]["min_accuracy"] = 0.05

Path(os.environ["TEST_CONFIG_FILE"]).write_text(
    yaml.safe_dump(config, sort_keys=False)
)
PY

echo "[deploy] Running deployment pipeline with test configuration..."
uv run run_deployment.py --config "${TEST_CONFIG_FILE}"
