#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${WEIGHT_DIR:?Set WEIGHT_DIR to the local model weight directory.}"
SERVE_MODEL_NAME="${SERVE_MODEL_NAME:-$(basename "${WEIGHT_DIR}")}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-fastapply7b}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8001}"
NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0,1}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}" \
  -e NCCL_P2P_DISABLE=1 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_SHM_DISABLE=1 \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${WEIGHT_DIR}:/models:ro" \
  -v "${WORKSPACE_DIR}:/workspace:ro" \
  vllm/vllm-openai:v0.15.0 \
  /models \
  --served-model-name "${SERVE_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --disable-custom-all-reduce \
  --generation-config vllm \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --host 0.0.0.0 \
  --port "${PORT}"

echo "Container started, waiting for service..."
sleep 5
docker logs --tail 20 "${CONTAINER_NAME}"