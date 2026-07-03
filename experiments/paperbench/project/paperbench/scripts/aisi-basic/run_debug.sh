#!/bin/bash

set -e

# 获取脚本所在目录的绝对路径（paperbench目录）
PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# 构建 Docker 镜像
NEW_IMAGE="aisi-basic-agent-basic:latest"

# 参数解析
AGENT_ID="${1:-aisi-basic-agent-my-o3}"
JUDGE_MODEL="${2:-o3-mini}"
PAPER_SPLIT="${3:-dev}"

# 自动检测 NVIDIA GPU
HAS_NVIDIA_GPU=False
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=True
    fi
fi

# Change to PAPERBENCH_DIR before building Docker image
cd "${PAPERBENCH_DIR}"

echo ""
echo "==================== BUILD PHASE ===================="
echo "Building Docker image from directory: $(pwd)"
echo "Image name: ${NEW_IMAGE}"
echo ""

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

echo ""
echo "✓ Docker image build completed successfully"
echo ""

# 运行评估，增加超时时间以允许容器/kernel 启动（Mac 上可能较慢）
export ALCATRAZ_TIMEOUT=600

echo "==================== EVALUATION PHASE ===================="
echo "Starting paperbench evaluation with debug logging..."
echo "ALCATRAZ_TIMEOUT: ${ALCATRAZ_TIMEOUT}"
echo "Agent ID: ${AGENT_ID}"
echo "Judge Model: ${JUDGE_MODEL}"
echo "Paper Split: ${PAPER_SPLIT}"
echo ""
echo "NOTE: If this hangs, check:"
echo "  1. Docker daemon is running"
echo "  2. Container logs: docker logs <container_id>"
echo "  3. Container health: docker ps"
echo ""

python -m paperbench.nano.entrypoint \
    paperbench.paper_split="${PAPER_SPLIT}" \
    paperbench.solver.agent_id="${AGENT_ID}" \
    paperbench.judge.code_only=True \
    paperbench.judge.model="${JUDGE_MODEL}" \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image="${NEW_IMAGE}" \
    paperbench.solver.cluster_config.local_network=True \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    paperbench.solver.is_nvidia_gpu_env="${HAS_NVIDIA_GPU}"

echo ""
echo "✓ Evaluation completed"
