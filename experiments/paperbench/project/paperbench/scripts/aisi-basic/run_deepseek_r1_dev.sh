#!/bin/bash
# 任何命令执行失败则立即退出
set -e

NEW_IMAGE="aisi-basic-agent-custom-knowledge:latest"
# JUDGE_MODEL="DMXAPI-HuoShan-DeepSeek-V3"
JUDGE_MODEL="o3-mini-2025-01-31"
# JUDGE_MODEL="test"

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent:latest

ENV HTTP_PROXY="http://172.20.0.1:7890" \
    HTTPS_PROXY="http://172.20.0.1:7890" \
    NO_PROXY="localhost,172.20.0.1,127.0.0.1,hf-mirror.com,vip.dmxapi.com,pypi.tuna.tsinghua.edu.cn,openaipublic.blob.core.windows.net" \
    HF_ENDPOINT="https://hf-mirror.com" 

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

python -m paperbench.nano.entrypoint \
    paperbench.paper_split=dev \
    paperbench.solver.agent_id=aisi-basic-agent-my-r1 \
    paperbench.judge.code_only=True \
    paperbench.judge.model="${JUDGE_MODEL}" \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image="${NEW_IMAGE}" \
    paperbench.solver.cluster_config.local_network=True \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    paperbench.solver.is_nvidia_gpu_env=True  



