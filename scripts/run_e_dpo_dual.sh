#!/usr/bin/env bash
set -euo pipefail

HELPFUL_CONFIG="${HELPFUL_CONFIG:-config_e_dpo_helpful.yaml}"
HARMLESS_CONFIG="${HARMLESS_CONFIG:-config_e_dpo_harmless.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-e_dpo_hh_outputs}"
LOG_ROOT="${LOG_ROOT:-logs}"
REPO_PREFIX="${REPO_PREFIX:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_e_dpo_dual.sh [--prepare-only] <hf_namespace>

Description:
  Runs e-DPO sequentially for helpful and harmless HH base configs using
  the original checked-in YAML files directly (no temporary YAML files).
  The script validates each config includes the expected:
    - dpo_training.hub_model_id

Options:
  --prepare-only   Validate hub_model_id and print planned launch info only.
  -h, --help       Show this help message.

Environment overrides:
  HELPFUL_CONFIG   (default: config_e_dpo_helpful.yaml)
  HARMLESS_CONFIG  (default: config_e_dpo_harmless.yaml)
  NPROC_PER_NODE   (default: 4)
  NNODES           (default: 1)
  OUTPUT_ROOT      (default: e_dpo_hh_outputs)
  LOG_ROOT         (default: logs)
  REPO_PREFIX      (optional prefix prepended to run name)
EOF
}

PREPARE_ONLY=0
HF_NAMESPACE=""

while (($# > 0)); do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "$HF_NAMESPACE" ]]; then
        HF_NAMESPACE="$1"
      else
        echo "Unexpected extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$HF_NAMESPACE" ]]; then
  echo "Missing required argument: <hf_namespace>" >&2
  usage >&2
  exit 1
fi

for cfg in "$HELPFUL_CONFIG" "$HARMLESS_CONFIG"; do
  if [[ ! -f "$cfg" ]]; then
    echo "Config not found: $cfg" >&2
    exit 1
  fi
done

timestamp="$(date +%Y%m%d_%H%M%S)"
torchrun_root="${LOG_ROOT}/torchrun_${timestamp}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$torchrun_root"

compute_run_name() {
  local base_name="$1"
  if [[ -n "$REPO_PREFIX" ]]; then
    printf "%s-%s" "$REPO_PREFIX" "$base_name"
    return
  fi
  printf "%s" "$base_name"
}

validate_hub_model_id() {
  local config_path="$1"
  local expected_hub_model_id="$2"

  uv run python - "$config_path" "$expected_hub_model_id" <<'PY'
import sys
import yaml

config_path, expected_hub_model_id = sys.argv[1:]

with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

if not isinstance(config, dict):
    raise SystemExit(f"Top-level YAML must be a mapping: {config_path}")

dpo_training = config.get("dpo_training")
if not isinstance(dpo_training, dict):
    raise SystemExit("Expected 'dpo_training' mapping in config")

actual_hub_model_id = dpo_training.get("hub_model_id")

errors = []
if actual_hub_model_id != expected_hub_model_id:
    errors.append(
        f"dpo_training.hub_model_id mismatch in {config_path}: "
        f"expected '{expected_hub_model_id}', found '{actual_hub_model_id}'"
    )

if errors:
    for error in errors:
        print(error, file=sys.stderr)
    raise SystemExit(1)
PY
}

run_one() {
  local label="$1"
  local config_path="$2"
  local base_run_name="$3"
  local run_name
  run_name="$(compute_run_name "$base_run_name")"

  local output_dir="${OUTPUT_ROOT}/${run_name}"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to run: output directory already exists: $output_dir" >&2
    exit 1
  fi

  local hub_model_id="${HF_NAMESPACE}/${run_name}"
  local run_log="${LOG_ROOT}/${run_name}_${timestamp}.log"
  local torchrun_log_dir="${torchrun_root}/${run_name}"

  validate_hub_model_id "$config_path" "$hub_model_id"

  echo "[${label}] config=${config_path}"
  echo "[${label}] run_name=${run_name}"
  echo "[${label}] hub_model_id=${hub_model_id}"
  echo "[${label}] output_dir=${output_dir}"
  echo "[${label}] torchrun_log_dir=${torchrun_log_dir}"
  echo "[${label}] run_log=${run_log}"

  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    echo "[${label}] prepare-only mode; training skipped."
    return
  fi

  uv run torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --log_dir "$torchrun_log_dir" \
    scripts/run_e_dpo.py \
    --config "$config_path" \
    --output_dir "$output_dir" 2>&1 | tee "$run_log"
}

run_one "helpful" "$HELPFUL_CONFIG" "hh-helpful-base-e-dpo"
run_one "harmless" "$HARMLESS_CONFIG" "hh-harmless-base-e-dpo"

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "Prepare-only run complete."
else
  echo "Dual e-DPO run complete."
fi
