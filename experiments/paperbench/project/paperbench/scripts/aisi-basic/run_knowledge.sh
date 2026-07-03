#!/bin/bash

set -e

# 获取脚本所在目录的绝对路径（paperbench目录）
PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

# 构建 Docker 镜像
NEW_IMAGE="aisi-basic-agent-basic-knowledge:latest"

# 参数解析
AGENT_ID="${1:-aisi-basic-agent-basic-knowledge}"
JUDGE_MODEL="${2:-o3-mini}"
PAPER_SPLIT="${3:-dev}"

# 自动检测 NVIDIA GPU
HAS_NVIDIA_GPU=False
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=True
    fi
fi

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

# 运行评估，增加超时时间以允许容器/kernel 启动（Mac 上可能较慢）
export ALCATRAZ_TIMEOUT=600

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
