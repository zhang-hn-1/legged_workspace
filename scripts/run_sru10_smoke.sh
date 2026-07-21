#!/usr/bin/env bash

set -Eeuo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${WORKSPACE_ROOT}/.venv/bin/python"

GPU_ID="${GPU_ID:-2}"
NUM_ENVS="${NUM_ENVS:-64}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2}"

LOG_DIR="${LOG_DIR:-/tmp/legged_workspace_logs}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${LOG_DIR}/sru10_smoke_${TIMESTAMP}.log"

fail() {
    echo "[FAIL] $*" >&2
    exit 1
}

echo "======================================================================"
echo "SRU10 PPO Smoke Test"
echo "======================================================================"
echo "Workspace       : ${WORKSPACE_ROOT}"
echo "Python          : ${PYTHON_BIN}"
echo "Physical GPU    : ${GPU_ID}"
echo "Process GPU     : cuda:0"
echo "Task            : anymal_c_flat_sru10"
echo "Environments    : ${NUM_ENVS}"
echo "Iterations      : ${MAX_ITERATIONS}"
echo "Log file        : ${LOG_FILE}"
echo "======================================================================"

[[ -x "${PYTHON_BIN}" ]] || \
    fail "虚拟环境 Python 不存在：${PYTHON_BIN}"

[[ -f "${WORKSPACE_ROOT}/legged_gym/legged_gym/scripts/train.py" ]] || \
    fail "找不到 legged_gym 训练脚本"

[[ -f \
    "${WORKSPACE_ROOT}/legged_gym/legged_gym/envs/anymal_c/flat/anymal_c_flat_sru10_config.py" \
]] || fail "找不到 SRU10 任务配置"

[[ -f \
    "${WORKSPACE_ROOT}/rsl_rl/rsl_rl/modules/actor_critic_sru_lh.py" \
]] || fail "找不到 ActorCriticSRULH 适配器"

[[ -f \
    "${WORKSPACE_ROOT}/sru_rotunbot_study/rotunbot_sru/memory.py" \
]] || fail "找不到 SRU memory 实现"

mkdir -p "${LOG_DIR}"

export PYTHONPATH="${WORKSPACE_ROOT}/isaacgym/python:${WORKSPACE_ROOT}/legged_gym:${WORKSPACE_ROOT}/rsl_rl:${WORKSPACE_ROOT}/sru_rotunbot_study:${PYTHONPATH:-}"

echo
echo "===== 环境信息 ====="

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" - <<'PY'
from isaacgym import gymapi, gymtorch
import platform
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

if torch.cuda.device_count() != 1:
    raise RuntimeError(
        "smoke test 预期只有一张可见 GPU，"
        f"实际为 {torch.cuda.device_count()}"
    )

print("Process GPU 0:", torch.cuda.get_device_name(0))
print("PRE_FLIGHT_CHECK=PASS")
PY

echo
echo "===== 开始 PPO smoke test ====="

cd "${WORKSPACE_ROOT}/legged_gym"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" legged_gym/scripts/train.py \
    --task=anymal_c_flat_sru10 \
    --headless \
    --sim_device=cuda:0 \
    --rl_device=cuda:0 \
    --pipeline=gpu \
    --physx \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    2>&1 | tee "${LOG_FILE}"

echo
echo "===== 日志异常扫描 ====="

ERROR_PATTERN='Traceback|CUDA out of memory|illegal memory access|shape mismatch|failed to create sim|asset file not found'

if grep -Eiq "${ERROR_PATTERN}" "${LOG_FILE}"; then
    echo "[FAIL] 日志中发现关键错误："
    grep -Ein "${ERROR_PATTERN}" "${LOG_FILE}" || true
    exit 1
fi

if grep -Eiq '(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' "${LOG_FILE}"; then
    echo "[FAIL] 日志中发现独立的 NaN 或 Inf："
    grep -Ein \
        '(^|[^[:alnum:]_])(nan|inf)([^[:alnum:]_]|$)' \
        "${LOG_FILE}" || true
    exit 1
fi

echo "[PASS] 未发现关键异常"
echo
echo "SRU10_SMOKE_TEST=PASS"
echo "Log file: ${LOG_FILE}"
