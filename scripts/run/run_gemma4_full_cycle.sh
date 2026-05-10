#!/usr/bin/env bash
set -euo pipefail

# Gemma4 wrapper for the generic full-cycle runner.
# It keeps the same pipeline order:
# base eval -> finetune -> finetuned eval

REPO_DIR="${HOME}/SpikeSorting"
cd "${REPO_DIR}"

export SERVE_BACKEND="${SERVE_BACKEND:-apptainer}"
export BASE_MODEL="${BASE_MODEL:-google/gemma-4-E4B-it}"
export BASE_SERVED_MODEL_NAME="${BASE_SERVED_MODEL_NAME:-google/gemma-4-E4B-it}"
export FINETUNE_MODEL_NAME="${FINETUNE_MODEL_NAME:-unsloth/gemma-4-E4B-it}"
export FT_OUTPUT_DIR="${FT_OUTPUT_DIR:-output/finetune_gemma4_e4b_vision_lora}"
export FT_MERGED_MODEL="${FT_MERGED_MODEL:-${FT_OUTPUT_DIR}/merged_16bit}"
export FT_SERVED_MODEL_NAME="${FT_SERVED_MODEL_NAME:-gemma4-ft}"
export BASE_EVAL_OUT="${BASE_EVAL_OUT:-output/unit_test_dataset_eval_base_gemma4}"
export FT_EVAL_OUT="${FT_EVAL_OUT:-output/unit_test_dataset_eval_ft_gemma4}"
export TRAIN_JSONL="${TRAIN_JSONL:-output/finetune_dataset/finetune_dataset_reason_action.jsonl}"
export TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-output/finetune_dataset}"

# Common defaults for cluster + apptainer serving
export PORT="${PORT:-8000}"
export HOST="${HOST:-127.0.0.1}"
export WAIT_SECONDS="${WAIT_SECONDS:-150}"
export EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-512}"
export USE_UV_RUN="${USE_UV_RUN:-true}"
export SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
export SKIP_FINETUNE_IF_MERGED_EXISTS="${SKIP_FINETUNE_IF_MERGED_EXISTS:-true}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-true}"
export VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-gemma4}"
export APPTAINER_SIF="${APPTAINER_SIF:-$HOME/vllm-openai_gemma4.sif}"
export APPTAINER_HF_CACHE_BIND="${APPTAINER_HF_CACHE_BIND:-$HOME/.cache/huggingface:/root/.cache/huggingface}"
export APPTAINER_TMP_BIND="${APPTAINER_TMP_BIND:-$HOME/vllm_cache:/tmp}"
export APPTAINER_OUTPUT_BIND="${APPTAINER_OUTPUT_BIND:-${REPO_DIR}/output:/workspace/output}"

echo "[gemma4] SERVE_BACKEND=${SERVE_BACKEND}"
echo "[gemma4] BASE_MODEL=${BASE_MODEL}"
echo "[gemma4] FINETUNE_MODEL_NAME=${FINETUNE_MODEL_NAME}"
echo "[gemma4] VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER}"
echo "[gemma4] APPTAINER_SIF=${APPTAINER_SIF}"

bash scripts/run/run_qwen35_full_cycle.sh "$@"
