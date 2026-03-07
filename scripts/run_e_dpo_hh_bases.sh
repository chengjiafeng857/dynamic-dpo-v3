#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_e_dpo_hh_bases.sh <hf_namespace> [base_config]

Runs Epsilon-DPO sequentially for HH helpful-base and harmless-base, then pushes
each trained model to Hugging Face Hub via dpo_training.hub_model_id.

Arguments:
  hf_namespace  Hugging Face user or org name used in hub_model_id.
  base_config   Base e-DPO config to clone per run. Default: config_e_dpo.yaml

Environment overrides:
  NPROC_PER_NODE  torchrun local world size. Default: 4
  NNODES          torchrun node count. Default: 1
  OUTPUT_ROOT     Per-run output root. Default: e_dpo_hh_outputs
  LOG_ROOT        Log root. Default: logs
  REPO_PREFIX     Optional prefix for generated run_name / hub repo names.

Example:
  HF_TOKEN=... scripts/run_e_dpo_hh_bases.sh my-hf-user config_e_dpo.yaml
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

HF_NAMESPACE="${HF_NAMESPACE:-${1:-}}"
BASE_CONFIG="${BASE_CONFIG:-${2:-config_e_dpo.yaml}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-e_dpo_hh_outputs}"
LOG_ROOT="${LOG_ROOT:-logs}"
REPO_PREFIX="${REPO_PREFIX:-}"

if [[ -z "${HF_NAMESPACE}" ]]; then
  echo "Missing hf_namespace." >&2
  usage >&2
  exit 1
fi

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Base config not found: ${BASE_CONFIG}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}"

timestamp="$(date +%Y%m%d_%H%M%S)"
torchrun_log_root="${LOG_ROOT}/torchrun_${timestamp}"
mkdir -p "${torchrun_log_root}"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/e_dpo_hh_bases.XXXXXX")"
cleanup() {
  local train_exit_code=$?
  rm -rf "${tmp_dir}"

  if [[ ${train_exit_code} -eq 0 ]]; then
    echo "Training completed successfully."
  else
    echo "Training failed with exit code ${train_exit_code}."
  fi

  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RUNPOD_POD_ID is not set. Are you running in RunPod?"
    echo "Skipping auto-shutdown."
    return
  fi

  echo "Shutting down pod ${RUNPOD_POD_ID} in 10 seconds..."
  sleep 10
  runpodctl stop pod "${RUNPOD_POD_ID}" || true
}
trap cleanup EXIT

for dataset_data_dir in helpful-base harmless-base; do
  mapfile -t run_meta < <(
    uv run python - "${BASE_CONFIG}" "${tmp_dir}" "${dataset_data_dir}" "${HF_NAMESPACE}" "${OUTPUT_ROOT}" "${REPO_PREFIX}" <<'PY'
from pathlib import Path
import re
import sys

import yaml

base_config_path = Path(sys.argv[1])
tmp_dir = Path(sys.argv[2])
dataset_data_dir = sys.argv[3]
hf_namespace = sys.argv[4]
output_root = Path(sys.argv[5])
repo_prefix = sys.argv[6]

config = yaml.safe_load(base_config_path.read_text())
policy_name = str(config["policy_name"])
model_slug = re.sub(r"[^a-z0-9]+", "-", policy_name.split("/", 1)[-1].lower()).strip("-")
dataset_slug = f"hh-{dataset_data_dir}"
run_name = f"{dataset_slug}-{model_slug}-e-dpo"
if repo_prefix:
    run_name = f"{repo_prefix}-{run_name}"
hub_model_id = f"{hf_namespace}/{run_name}"
output_dir = output_root / run_name

dataset_cfg = config.setdefault("dataset", {})
dataset_cfg["data_dir"] = dataset_data_dir

dpo_cfg = config.setdefault("dpo_training", {})
dpo_cfg["run_name"] = run_name
dpo_cfg["save_dir"] = str(output_dir)
dpo_cfg["hub_model_id"] = hub_model_id

temp_config_path = tmp_dir / f"{run_name}.yaml"
with temp_config_path.open("w", encoding="ascii") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=False)

for value in (str(temp_config_path), run_name, hub_model_id, str(output_dir)):
    print(value)
PY
  )

  temp_config="${run_meta[0]}"
  run_name="${run_meta[1]}"
  hub_model_id="${run_meta[2]}"
  output_dir="${run_meta[3]}"
  run_log="${LOG_ROOT}/${run_name}_${timestamp}.log"
  worker_log_dir="${torchrun_log_root}/${run_name}"

  if [[ -e "${output_dir}" ]]; then
    echo "Output directory already exists, refusing to overwrite: ${output_dir}" >&2
    exit 1
  fi

  echo "[E-DPO-HH] Starting ${dataset_data_dir} -> ${hub_model_id}"

  PYTHONUNBUFFERED=1 \
  PYTHONFAULTHANDLER=1 \
  TOKENIZERS_PARALLELISM=false \
  NCCL_ASYNC_ERROR_HANDLING=1 \
  NCCL_DEBUG=WARN \
  TRANSFORMERS_VERBOSITY=warning \
  HF_DATASETS_VERBOSITY=warning \
  ACCELERATE_LOG_LEVEL=warning \
  uv run torchrun \
    --standalone \
    --nnodes="${NNODES}" \
    --nproc-per-node="${NPROC_PER_NODE}" \
    --rdzv-backend=c10d \
    --max-restarts=0 \
    --log-dir "${worker_log_dir}" \
    --redirects 3 \
    --local-ranks-filter=0 \
    scripts/run_e_dpo.py \
    --config "${temp_config}" \
    --output_dir "${output_dir}" \
    2>&1 | tee "${run_log}"

  echo "[E-DPO-HH] Completed ${dataset_data_dir} -> ${hub_model_id}"
done

echo "[E-DPO-HH] All runs completed successfully."
