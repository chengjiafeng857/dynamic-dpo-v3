#!/usr/bin/env bash
set -euo pipefail

LLAMA_CONFIG="${LLAMA_CONFIG:-config_e_dpo_ultrafeedback_llama.yaml}"
QWEN_CONFIG="${QWEN_CONFIG:-config_e_dpo_ultrafeedback_qwen.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
STOP_RUNPOD_AFTER_RUN="${STOP_RUNPOD_AFTER_RUN:-auto}"
RUNPOD_STOP_COMMAND="${RUNPOD_STOP_COMMAND:-runpodctl stop pod \"\$RUNPOD_POD_ID\"}"

should_stop_runpod() {
  case "$(printf '%s' "$STOP_RUNPOD_AFTER_RUN" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    auto) [[ -n "${RUNPOD_POD_ID:-}" ]] ;;
    *)
      echo "Invalid STOP_RUNPOD_AFTER_RUN value: $STOP_RUNPOD_AFTER_RUN" >&2
      exit 1
      ;;
  esac
}

stop_runpod() {
  if ! should_stop_runpod; then
    echo "[runpod] shutdown skipped"
    return
  fi

  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    echo "RUNPOD_POD_ID is not set. Are you running in RunPod?"
    echo "Skipping auto-shutdown."
    return
  fi

  echo "Shutting down pod ${RUNPOD_POD_ID} in 10 seconds..."
  sleep 10
  echo "[runpod] stopping pod with command: ${RUNPOD_STOP_COMMAND}"
  bash -lc "$RUNPOD_STOP_COMMAND"
}

on_exit() {
  local exit_code="$?"
  if [[ "$exit_code" -eq 0 ]]; then
    echo "UltraFeedback e-DPO runs complete."
  else
    echo "UltraFeedback e-DPO run failed with exit code ${exit_code}."
  fi
  stop_runpod || true
  exit "$exit_code"
}

run_one() {
  local label="$1"
  local config_path="$2"

  if [[ ! -f "$config_path" ]]; then
    echo "Config not found: $config_path" >&2
    exit 1
  fi

  echo "[${label}] config=${config_path}"
  uv run torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    scripts/run_e_dpo.py \
    --config "$config_path"
}

trap on_exit EXIT

run_one "llama" "$LLAMA_CONFIG"
run_one "qwen" "$QWEN_CONFIG"
