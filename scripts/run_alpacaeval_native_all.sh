#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ALPACAEVAL_DIR="${ALPACAEVAL_DIR:-${REPO_ROOT}/eval/alpacaeval}"
CONFIG_DIR="${CONFIG_DIR:-${ALPACAEVAL_DIR}/configs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/alpacaeval}"
ANNOTATORS_CONFIG="${ANNOTATORS_CONFIG:-weighted_alpaca_eval_gpt4_turbo}"
DRY_RUN=0
CONTINUE_ON_ERROR=0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_alpacaeval_native_all.sh [--dry-run] [--continue-on-error]

Description:
  Runs native AlpacaEval evaluate_from_model for every YAML config under
  eval/alpacaeval/configs/.

Environment overrides:
  ALPACAEVAL_DIR      Default: <repo>/eval/alpacaeval
  CONFIG_DIR          Default: $ALPACAEVAL_DIR/configs
  OUTPUT_ROOT         Default: <repo>/outputs/alpacaeval
  ANNOTATORS_CONFIG   Default: weighted_alpaca_eval_gpt4_turbo
EOF
}

while (($# > 0)); do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --continue-on-error)
      CONTINUE_ON_ERROR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$CONFIG_DIR" ]]; then
  echo "Config directory not found: $CONFIG_DIR" >&2
  exit 1
fi

shopt -s nullglob
config_paths=("$CONFIG_DIR"/*.yaml)
shopt -u nullglob

if [[ "${#config_paths[@]}" -eq 0 ]]; then
  echo "No YAML files found under: $CONFIG_DIR" >&2
  exit 1
fi

trim_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "$value"
}

read_pretty_name() {
  local config_path="$1"
  local first_key
  local pretty_name

  first_key="$(sed -n 's/^\([^[:space:]][^:]*\):.*/\1/p' "$config_path" | head -n 1)"
  pretty_name="$(sed -n 's/^  pretty_name:[[:space:]]*//p' "$config_path" | head -n 1)"

  if [[ -n "$pretty_name" ]]; then
    trim_quotes "$pretty_name"
    return
  fi

  if [[ -n "$first_key" ]]; then
    trim_quotes "$first_key"
    return
  fi

  echo "Could not determine pretty_name for ${config_path}" >&2
  exit 1
}

failures=()

for config_path in "${config_paths[@]}"; do
  config_name="$(basename "$config_path")"
  pretty_name="$(read_pretty_name "$config_path")"
  output_path="${OUTPUT_ROOT}/${pretty_name}/results"
  relative_config_path="configs/${config_name}"

  command=(
    uv run python -m alpaca_eval.main evaluate_from_model
    --model_configs "$relative_config_path"
    --annotators_config "$ANNOTATORS_CONFIG"
    --output_path "$output_path"
  )

  echo "[alpacaeval] config=${config_name} pretty_name=${pretty_name}"
  echo "[alpacaeval] output_path=${output_path}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[alpacaeval] dry-run command='
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi

  if (
    cd "$ALPACAEVAL_DIR"
    "${command[@]}"
  ); then
    echo "[alpacaeval] completed ${config_name}"
  else
    echo "[alpacaeval] failed ${config_name}" >&2
    failures+=("${config_name}")
    if [[ "$CONTINUE_ON_ERROR" -ne 1 ]]; then
      exit 1
    fi
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  printf '[alpacaeval] failed configs: %s\n' "${failures[*]}" >&2
  exit 1
fi
