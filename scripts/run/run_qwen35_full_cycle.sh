#!/usr/bin/env bash
set -euo pipefail

# End-to-end no-tmux workflow:
# 1) run base vLLM in background (native or apptainer)
# 2) wait and run base unit test
# 3) stop vLLM
# 4) run finetune via uv run
# 5) run finetuned vLLM in background
# 6) wait and run finetuned unit test
# 7) stop vLLM

REPO_DIR="${HOME}/SpikeSorting"
cd "${REPO_DIR}"

# ==========================
# Configurable parameters
# ==========================
PORT="${PORT:-8000}"
WAIT_SECONDS="${WAIT_SECONDS:-150}"
EVAL_CHANNELS="${EVAL_CHANNELS:-CH3,CH20,CH30,CH31}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-512}"
EVAL_ENABLE_THINKING="${EVAL_ENABLE_THINKING:-false}"
HOST="${HOST:-127.0.0.1}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
BASE_SERVED_MODEL_NAME="${BASE_SERVED_MODEL_NAME:-${BASE_MODEL}}"
FT_OUTPUT_DIR="${FT_OUTPUT_DIR:-output/finetune_qwen35_4b_vision_lora}"
FT_MERGED_MODEL="${FT_MERGED_MODEL:-${FT_OUTPUT_DIR}/merged_16bit}"
FT_SERVED_MODEL_NAME="${FT_SERVED_MODEL_NAME:-finetuned-model}"
FINETUNE_MODEL_NAME="${FINETUNE_MODEL_NAME:-unsloth/Qwen3.5-4B}"
TRAIN_JSONL="${TRAIN_JSONL:-output/finetune_dataset/finetune_dataset_reason_action.jsonl}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-output/finetune_dataset}"

UNIT_TEST_SCRIPT="scripts/test/eval_unit_actions_from_dataset.py"
FINETUNE_SCRIPT="scripts/finetune/train_qwen35_unsloth.py"

BASE_EVAL_OUT="${BASE_EVAL_OUT:-output/unit_test_dataset_eval_base_allch}"
FT_EVAL_OUT="${FT_EVAL_OUT:-output/unit_test_dataset_eval_ft_allch}"
BASE_EVAL_RAW_LOG="${BASE_EVAL_RAW_LOG:-output/vllm_logs/base_eval_raw_full.log}"

SERVE_LOG_DIR="${SERVE_LOG_DIR:-output/vllm_logs}"
mkdir -p "${SERVE_LOG_DIR}"
USE_UV_RUN="${USE_UV_RUN:-true}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
SKIP_FINETUNE_IF_MERGED_EXISTS="${SKIP_FINETUNE_IF_MERGED_EXISTS:-true}"

# Serving backend:
# - native: vllm serve ...
# - apptainer: apptainer exec ... vllm serve ...
SERVE_BACKEND="${SERVE_BACKEND:-native}"  # native | apptainer
APPTAINER_SIF="${APPTAINER_SIF:-$HOME/vllm-openai_gemma4.sif}"
APPTAINER_HF_CACHE_BIND="${APPTAINER_HF_CACHE_BIND:-$HOME/.cache/huggingface:/root/.cache/huggingface}"
APPTAINER_TMP_BIND="${APPTAINER_TMP_BIND:-$HOME/vllm_cache:/tmp}"
APPTAINER_OUTPUT_BIND="${APPTAINER_OUTPUT_BIND:-${REPO_DIR}/output:/workspace/output}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-}"

VLLM_PID=""

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
  # Respect caller-provided thinking config; default to non-thinking.
  export VLM_EXTRA_BODY_JSON="${VLM_EXTRA_BODY_JSON:-{\"chat_template_kwargs\":{\"enable_thinking\":false}}}"
}

activate_venv_train() {
  :
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

eval_artifact_exists() {
  local out_dir="$1"
  [[ -f "${out_dir}/summary_accuracy.json" ]] && [[ -f "${out_dir}/detail_predictions.csv" ]]
}

start_vllm_bg() {
  local model_path="$1"
  local log_file="$2"
  local served_model_name="$3"
  local serve_model_path="$model_path"
  local trust_flag=""
  local parser_flag=""
  if [[ "${TRUST_REMOTE_CODE}" == "true" ]]; then
    trust_flag="--trust-remote-code"
  fi
  if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
    parser_flag="--reasoning-parser ${VLLM_REASONING_PARSER}"
  fi

  # Map finetuned local output path to container-visible path when using apptainer.
  # Example: output/foo/merged_16bit -> /workspace/output/foo/merged_16bit
  if [[ "${SERVE_BACKEND}" == "apptainer" ]]; then
    if [[ "${model_path}" == output/* ]]; then
      serve_model_path="/workspace/${model_path}"
    elif [[ "${model_path}" == "${REPO_DIR}/output/"* ]]; then
      serve_model_path="/workspace/output/${model_path#${REPO_DIR}/output/}"
    fi
  else
    # Native backend: convert local relative model path to absolute when applicable.
    if [[ "${model_path}" != /* ]] && [[ -e "${model_path}" ]]; then
      serve_model_path="$(cd "$(dirname "${model_path}")" && pwd)/$(basename "${model_path}")"
    fi
  fi

  echo "[vllm] backend=${SERVE_BACKEND} model=${serve_model_path} served_model_name=${served_model_name}"
  if [[ "${SERVE_BACKEND}" == "apptainer" ]]; then
    if [[ ! -f "${APPTAINER_SIF}" ]]; then
      echo "[error] APPTAINER_SIF not found: ${APPTAINER_SIF}"
      exit 1
    fi
    # Pass HF_TOKEN into container only if set in host env.
    local env_prefix=""
    if [[ -n "${HF_TOKEN:-}" ]]; then
      env_prefix="APPTAINERENV_HF_TOKEN=${HF_TOKEN}"
    fi
    nohup env ${env_prefix} apptainer exec --nv \
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
    nohup vllm serve "${serve_model_path}" --served-model-name "${served_model_name}" --host 0.0.0.0 --port "${PORT}" ${trust_flag} ${parser_flag} >"${log_file}" 2>&1 &
  fi
  VLLM_PID=$!
  echo "[vllm] pid=${VLLM_PID}, log=${log_file}"
  echo "[vllm] waiting up to ${WAIT_SECONDS}s for readiness..."
  wait_vllm_ready "${log_file}"
}

wait_vllm_ready() {
  local log_file="$1"
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
  if [[ -f "${log_file}" ]]; then
    echo "[error] tail of ${log_file}:"
    tail -n 80 "${log_file}" || true
  fi
  return 1
}

stop_vllm_bg() {
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid=${VLLM_PID}"
    kill "${VLLM_PID}" || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  VLLM_PID=""
}

run_base_unit_test() {
  echo "[eval-base] running unit test on channels=${EVAL_CHANNELS}"
  local thinking_flag="--disable-thinking"
  if [[ "${EVAL_ENABLE_THINKING}" == "true" ]]; then
    thinking_flag="--enable-thinking"
  fi
  run_module "${UNIT_TEST_SCRIPT}" \
    --eval-channels "${EVAL_CHANNELS}" \
    --provider vllm \
    --model "${BASE_SERVED_MODEL_NAME}" \
    --max-tokens "${EVAL_MAX_TOKENS}" \
    ${thinking_flag} \
    --print-first-raw \
    --raw-log-file "${BASE_EVAL_RAW_LOG}" \
    --output-dir "${BASE_EVAL_OUT}"
}

run_ft_unit_test() {
  echo "[eval-ft] running unit test on channels=${EVAL_CHANNELS}"
  local thinking_flag="--disable-thinking"
  if [[ "${EVAL_ENABLE_THINKING}" == "true" ]]; then
    thinking_flag="--enable-thinking"
  fi
  run_module "${UNIT_TEST_SCRIPT}" \
    --eval-channels "${EVAL_CHANNELS}" \
    --provider vllm \
    --model "${FT_SERVED_MODEL_NAME}" \
    --max-tokens "${EVAL_MAX_TOKENS}" \
    ${thinking_flag} \
    --output-dir "${FT_EVAL_OUT}"
}

run_finetune() {
  echo "[finetune] training with CH3,CH20,CH31 -> eval CH30 model=${FINETUNE_MODEL_NAME}"
  echo "[finetune] input_jsonl=${TRAIN_JSONL}"
  run_module "${FINETUNE_SCRIPT}" \
    --input-jsonl "${TRAIN_JSONL}" \
    --dataset-root "${TRAIN_DATASET_ROOT}" \
    --train-channels CH3,CH20,CH31 \
    --eval-channels CH30 \
    --expert-only \
    --target-mode original \
    --model-name "${FINETUNE_MODEL_NAME}" \
    --output-dir "${FT_OUTPUT_DIR}" \
    --max-steps 200
}

echo "========================================"
echo "Qwen/Gemma Full Cycle Start"
echo "repo=${REPO_DIR}"
echo "host=${HOST} port=${PORT} backend=${SERVE_BACKEND}"
echo "========================================"

deactivate 2>/dev/null || true
activate_local_env
if [[ "${SKIP_IF_EXISTS}" == "true" ]] && eval_artifact_exists "${BASE_EVAL_OUT}"; then
  echo "[skip] base eval outputs exist at ${BASE_EVAL_OUT}; skipping base vLLM + eval."
else
  start_vllm_bg "${BASE_MODEL}" "${SERVE_LOG_DIR}/vllm_base.log" "${BASE_SERVED_MODEL_NAME}"
  run_base_unit_test
  stop_vllm_bg
fi

deactivate 2>/dev/null || true
activate_venv_train
if [[ "${SKIP_FINETUNE_IF_MERGED_EXISTS}" == "true" ]] && [[ -d "${FT_MERGED_MODEL}" ]]; then
  echo "[skip] merged finetuned model exists at ${FT_MERGED_MODEL}; skipping finetune."
else
  run_finetune
fi

deactivate 2>/dev/null || true
activate_local_env
if [[ ! -d "${FT_MERGED_MODEL}" ]]; then
  echo "[error] finetuned merged model not found: ${FT_MERGED_MODEL}"
  exit 1
fi
if [[ "${SKIP_IF_EXISTS}" == "true" ]] && eval_artifact_exists "${FT_EVAL_OUT}"; then
  echo "[skip] finetuned eval outputs exist at ${FT_EVAL_OUT}; skipping finetuned vLLM + eval."
else
  start_vllm_bg "${FT_MERGED_MODEL}" "${SERVE_LOG_DIR}/vllm_ft.log" "${FT_SERVED_MODEL_NAME}"
  run_ft_unit_test
  stop_vllm_bg
fi

echo "========================================"
echo "Done."
echo "Base eval: ${BASE_EVAL_OUT}"
echo "FT eval:   ${FT_EVAL_OUT}"
echo "Logs:      ${SERVE_LOG_DIR}"
echo "========================================"
