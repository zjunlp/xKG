#!/bin/bash

set -e

PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

NEW_IMAGE="aisi-basic-agent-basic:latest"

AGENT_ID="${1:-aisi-basic-agent-my-o3}"
JUDGE_MODEL="${2:-o3-mini}"
PAPER_SPLIT="${3:-dev}"

HAS_NVIDIA_GPU=False
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=True
    fi
fi

cd "${PAPERBENCH_DIR}"

echo "Building Docker image from directory: $(pwd)"
echo "Image name: ${NEW_IMAGE}"

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent

ENV HF_ENDPOINT="https://hf-mirror.com"

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF


python -m paperbench.nano.entrypoint \
    paperbench.paper_split="${PAPER_SPLIT}" \
    paperbench.solver.agent_id="${AGENT_ID}" \
    paperbench.judge.code_only=True \
    paperbench.judge.model="${JUDGE_MODEL}" \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image="${NEW_IMAGE}" \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    paperbench.solver.is_nvidia_gpu_env="${HAS_NVIDIA_GPU}"