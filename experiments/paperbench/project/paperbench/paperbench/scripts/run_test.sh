python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    runner.recorder=nanoeval.json_recorder:json_recorder