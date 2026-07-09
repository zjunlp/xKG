# important parameters:
# - paperbench.paper_split: the split to use, debug for rice paper, dev for two papers, human for human evaluation, all for all papers
# - paperbench.solver: the solver to use, nano.eval:ExternalPythonCodingSolver for external python coding
# - paperbench.solver.agent_id: the agent id to use

python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=dummy \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=dummy:latest  \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder