set -e

NEW_IMAGE="aisi-basic-agent-custom:latest"

JUDGE_MODEL="o3-mini"

docker build -t "${NEW_IMAGE}" -f - . <<'EOF'
FROM aisi-basic-agent:latest

ENV  HF_ENDPOINT="https://hf-mirror.com" 

COPY ./paperbench/agents/aisi-basic-agent/ /home/agent/
EOF

python -m paperbench.nano.entrypoint \
    paperbench.paper_split=dev \
    paperbench.solver.agent_id=aisi-basic-agent-my-o3 \
    paperbench.judge.code_only=True \
    paperbench.judge.model="${JUDGE_MODEL}" \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image="${NEW_IMAGE}" \
    paperbench.solver.cluster_config.local_network=True \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    paperbench.solver.is_nvidia_gpu_env=True \


