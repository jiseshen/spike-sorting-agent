#!/usr/bin/env bash
set -euo pipefail

# Run VLM unit decision test (no_rag vs rag) for:
# - Qwen3.5-4B
# - Gemma4-E4B
# Then generate comparison plots.

REPO_DIR="${REPO_DIR:-$HOME/SpikeSorting}"
cd "${REPO_DIR}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WAIT_SECONDS="${WAIT_SECONDS:-180}"
SERVE_BACKEND="${SERVE_BACKEND:-native}"   # native | apptainer
USE_UV_RUN="${USE_UV_RUN:-true}"
MANAGE_VLLM="${MANAGE_VLLM:-true}"         # false => assume external vLLM already running
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

# Apptainer defaults
APPTAINER_SIF="${APPTAINER_SIF:-$HOME/vllm-openai_gemma4.sif}"
APPTAINER_HF_CACHE_BIND="${APPTAINER_HF_CACHE_BIND:-$HOME/.cache/huggingface:/root/.cache/huggingface}"
APPTAINER_TMP_BIND="${APPTAINER_TMP_BIND:-$HOME/vllm_cache:/tmp}"
APPTAINER_OUTPUT_BIND="${APPTAINER_OUTPUT_BIND:-${REPO_DIR}/output:/workspace/output}"

# Eval defaults
CHANNELS_STR="${CHANNELS_STR:-CH3 CH20 CH30 CH31}"
AUTO_DISCARD_THRESHOLD="${AUTO_DISCARD_THRESHOLD:-500}"
RAG_TOP_K="${RAG_TOP_K:-3}"
RAG_WAVEFORM_WEIGHT="${RAG_WAVEFORM_WEIGHT:-0.7}"
RAG_FEATURE_WEIGHT="${RAG_FEATURE_WEIGHT:-0.3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"
MAX_STEPS_PER_CHANNEL="${MAX_STEPS_PER_CHANNEL:-0}"  # >0 means [TEMP SMOKE]
ROLLING_WINDOW="${ROLLING_WINDOW:-10}"

# Models
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-Qwen/Qwen3.5-4B}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-Qwen/Qwen3.5-4B}"
QWEN_REASONING_PARSER="${QWEN_REASONING_PARSER:-}"
GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-google/gemma-4-E4B-it}"
GEMMA_SERVED_MODEL_NAME="${GEMMA_SERVED_MODEL_NAME:-google/gemma-4-E4B-it}"
GEMMA_REASONING_PARSER="${GEMMA_REASONING_PARSER:-gemma4}"

RESULT_ROOT="${RESULT_ROOT:-output/rag_backbone_eval}"
QWEN_OUT="${QWEN_OUT:-${RESULT_ROOT}/qwen35_4b}"
GEMMA_OUT="${GEMMA_OUT:-${RESULT_ROOT}/gemma4_e4b}"
PLOTS_OUT="${PLOTS_OUT:-${RESULT_ROOT}/plots}"
SERVE_LOG_DIR="${SERVE_LOG_DIR:-${RESULT_ROOT}/serve_logs}"
mkdir -p "${SERVE_LOG_DIR}" "${RESULT_ROOT}"

VLLM_PID=""
CURRENT_VLLM_LOG=""

cleanup() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[cleanup] stopping vLLM pid=${VLLM_PID}"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

activate_local_env() {
  if [[ "${SERVE_BACKEND}" == "native" ]] && [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
  fi
  export VLLM_BASE_URL="http://${HOST}:${PORT}/v1"
  export VLM_EXTRA_BODY_JSON="${VLM_EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
}

run_module() {
  local script="$1"
  shift
  if [[ "${USE_UV_RUN}" == "true" ]]; then
    PYTHONUNBUFFERED=1 uv run python3 "${script}" "$@"
  else
    python3 -u "${script}" "$@"
  fi
}

wait_vllm_ready() {
  local base_url="http://${HOST}:${PORT}/v1/models"
  local elapsed=0
  while (( elapsed < WAIT_SECONDS )); do
    if curl -sS --max-time 3 "${base_url}" >/dev/null 2>&1; then
      echo "[vllm] ready at ${base_url}"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "[error] vLLM not reachable at ${base_url} within ${WAIT_SECONDS}s"
  if [[ -n "${CURRENT_VLLM_LOG}" ]] && [[ -f "${CURRENT_VLLM_LOG}" ]]; then
    echo "[error] tail of ${CURRENT_VLLM_LOG}:"
    tail -n 100 "${CURRENT_VLLM_LOG}" || true
  fi
  return 1
}

start_vllm_bg() {
  local model_path="$1"
  local served_model_name="$2"
  local log_file="$3"
  local reasoning_parser="$4"

  local trust_flag=""
  local parser_flag=""
  local serve_model_path="${model_path}"
  if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    trust_flag="--trust-remote-code"
  fi
  if [[ -n "${reasoning_parser}" ]]; then
    parser_flag="--reasoning-parser ${reasoning_parser}"
  fi

  if [[ "${SERVE_BACKEND}" == "apptainer" ]]; then
    if [[ "${model_path}" == output/* ]]; then
      serve_model_path="/workspace/${model_path}"
    elif [[ "${model_path}" == "${REPO_DIR}/output/"* ]]; then
      serve_model_path="/workspace/output/${model_path#${REPO_DIR}/output/}"
    fi
  else
    if [[ "${model_path}" != /* ]] && [[ -e "${model_path}" ]]; then
      serve_model_path="$(cd "$(dirname "${model_path}")" && pwd)/$(basename "${model_path}")"
    fi
  fi

  echo "[vllm] backend=${SERVE_BACKEND} model=${serve_model_path} served=${served_model_name}"
  CURRENT_VLLM_LOG="${log_file}"
  if [[ "${SERVE_BACKEND}" == "apptainer" ]]; then
    if [[ ! -f "${APPTAINER_SIF}" ]]; then
      echo "[error] APPTAINER_SIF not found: ${APPTAINER_SIF}"
      exit 1
    fi
    nohup apptainer exec --nv \
      -B "${APPTAINER_OUTPUT_BIND}" \
      -B "${APPTAINER_HF_CACHE_BIND}" \
      -B "${APPTAINER_TMP_BIND}" \
      "${APPTAINER_SIF}" \
      vllm serve "${serve_model_path}" \
      --served-model-name "${served_model_name}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      ${trust_flag} \
      ${parser_flag} >"${log_file}" 2>&1 &
  else
    if ! command -v vllm >/dev/null 2>&1; then
      echo "[error] vllm command not found in current shell."
      echo "[hint] activate your serving env first, or set SERVE_BACKEND=apptainer."
      exit 1
    fi
    nohup vllm serve "${serve_model_path}" \
      --served-model-name "${served_model_name}" \
      --host 0.0.0.0 \
      --port "${PORT}" \
      --max-model-len "${VLLM_MAX_MODEL_LEN}" \
      ${trust_flag} \
      ${parser_flag} >"${log_file}" 2>&1 &
  fi
  VLLM_PID=$!
  wait_vllm_ready
}

stop_vllm_bg() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid=${VLLM_PID}"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  VLLM_PID=""
  CURRENT_VLLM_LOG=""
}

run_unit_compare_once() {
  local served_model_name="$1"
  local output_base="$2"
  shift 2
  local channels=("$@")

  local cmd=(
    scripts/test/test_vlm_unit_decisions.py
    --provider vllm
    --model "${served_model_name}"
    --run-mode both
    --output-base "${output_base}"
    --channels
  )
  for ch in "${channels[@]}"; do
    cmd+=("${ch}")
  done
  cmd+=(
    --auto-discard-threshold "${AUTO_DISCARD_THRESHOLD}"
    --rag-top-k "${RAG_TOP_K}"
    --rag-waveform-weight "${RAG_WAVEFORM_WEIGHT}"
    --rag-feature-weight "${RAG_FEATURE_WEIGHT}"
    --temperature "${TEMPERATURE}"
    --reasoning-effort "${REASONING_EFFORT}"
  )
  if (( MAX_STEPS_PER_CHANNEL > 0 )); then
    cmd+=(--max-steps-per-channel "${MAX_STEPS_PER_CHANNEL}")
  fi

  run_module "${cmd[@]}"
}

run_one_model() {
  local label="$1"
  local model_path="$2"
  local served_model_name="$3"
  local reasoning_parser="$4"
  local output_base="$5"
  local log_file="$6"
  shift 6
  local channels=("$@")

  echo "========================================"
  echo "[${label}] output=${output_base}"
  echo "========================================"

  mkdir -p "${output_base}"
  if [[ "${MANAGE_VLLM}" == "true" ]]; then
    start_vllm_bg "${model_path}" "${served_model_name}" "${log_file}" "${reasoning_parser}"
  fi
  run_unit_compare_once "${served_model_name}" "${output_base}" "${channels[@]}"
  if [[ "${MANAGE_VLLM}" == "true" ]]; then
    stop_vllm_bg
  fi
}

read -r -a CHANNELS <<< "${CHANNELS_STR}"

echo "========================================"
echo "RAG Backbone Unit Compare"
echo "repo=${REPO_DIR}"
echo "channels=${CHANNELS_STR}"
echo "manage_vllm=${MANAGE_VLLM} backend=${SERVE_BACKEND}"
echo "result_root=${RESULT_ROOT}"
echo "========================================"

deactivate 2>/dev/null || true
activate_local_env

run_one_model \
  "Qwen3.5-4B" \
  "${QWEN_MODEL_PATH}" \
  "${QWEN_SERVED_MODEL_NAME}" \
  "${QWEN_REASONING_PARSER}" \
  "${QWEN_OUT}" \
  "${SERVE_LOG_DIR}/qwen_vllm.log" \
  "${CHANNELS[@]}"

deactivate 2>/dev/null || true
activate_local_env

run_one_model \
  "Gemma4-E4B" \
  "${GEMMA_MODEL_PATH}" \
  "${GEMMA_SERVED_MODEL_NAME}" \
  "${GEMMA_REASONING_PARSER}" \
  "${GEMMA_OUT}" \
  "${SERVE_LOG_DIR}/gemma_vllm.log" \
  "${CHANNELS[@]}"

run_module scripts/plot/plot_rag_backbone_unit_compare.py \
  --qwen-dir "${QWEN_OUT}" \
  --gemma-dir "${GEMMA_OUT}" \
  --channels "${CHANNELS[@]}" \
  --rolling-window "${ROLLING_WINDOW}" \
  --out-dir "${PLOTS_OUT}"

echo "========================================"
echo "Done."
echo "Qwen results : ${QWEN_OUT}"
echo "Gemma results: ${GEMMA_OUT}"
echo "Plots        : ${PLOTS_OUT}"
echo "Serve logs   : ${SERVE_LOG_DIR}"
echo "========================================"
