#!/bin/bash

set -e

PAPERBENCH_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

XKG_ROOT="$(cd "${PAPERBENCH_DIR}/../../../.." && pwd)"

NEW_IMAGE="aisi-basic-agent-iterative-knowledge:latest"

AGENT_ID="${1:-aisi-basic-agent-iterative-knowledge}"
JUDGE_MODEL="${2:-o3-mini}"
PAPER_SPLIT="${3:-dev}"

HAS_NVIDIA_GPU=False
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        HAS_NVIDIA_GPU=True
    fi
fi

cd "${XKG_ROOT}"

echo "Building Knowledge Docker image from context: $(pwd)"
echo "Image name: ${NEW_IMAGE}"


docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-knowledge-agent

COPY ./experiments/paperbench/project/paperbench/paperbench/agents/aisi-basic-agent/ /home/agent/

COPY ./xKG/source/ /opt/xKG/xKG/source/
COPY ./xKG/config.yaml /opt/xKG/xKG/config.yaml
COPY ./pyproject.toml /opt/xKG/pyproject.toml

COPY ./xKG/storage/kg/ /opt/xKG/xKG/storage/kg/

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
