import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path

from _basic_agent_iterative import basic_agent_iterative  # type: ignore
from _basic_agent_plus import basic_agent_plus  # type: ignore
from _execute import bash, python
from _file_reader import read_file_chunk, search_file  # type: ignore
from _knowledge import get_overview, get_similar_papers, get_similar_techniques, get_full_techniques
from inspect_ai import Task, eval, task  # type: ignore
from inspect_ai.dataset import Sample  # type: ignore
from inspect_ai.tool import web_browser  # type: ignore
from templates import additional_notes_template
from utils import get_gpu_generation

# PB constants
CODE_DIR = Path(os.environ["CODE_DIR"])
AGENT_DIR = Path(os.environ["AGENT_DIR"])
WORKSPACE_BASE = os.environ.get("WORKSPACE_BASE")
assert WORKSPACE_BASE is not None, "Environment variable 'WORKSPACE_BASE' is not set."

# aisi-specific settings
MAX_TIME_IN_HOURS = os.environ.get("MAX_TIME_IN_HOURS")
assert MAX_TIME_IN_HOURS is not None, "Environment variable 'MAX_TIME_IN_HOURS' is not set."
MODEL = os.environ.get("MODEL")
assert MODEL is not None, "Environment variable 'MODEL' is not set."
DISALLOW_SUBMIT = os.environ.get("DISALLOW_SUBMIT", False)
ITERATIVE_AGENT = os.environ.get("ITERATIVE_AGENT", False)
gpu_generation = get_gpu_generation()
type_of_processor = gpu_generation if gpu_generation else "CPU"

additional_notes = additional_notes_template.substitute(
    type_of_processor=type_of_processor,
    max_time_in_hours=MAX_TIME_IN_HOURS,
    workspace=CODE_DIR,
    workspace_base=WORKSPACE_BASE,
)

# 根据 KNOWLEDGE_ENABLED 选择 instructions 文件
knowledge_enabled = os.environ.get("KNOWLEDGE_ENABLED", "false").lower() == "true"
if knowledge_enabled:
    instructions_file = f"{WORKSPACE_BASE}/instructions_with_knowledge.txt"
else:
    instructions_file = f"{WORKSPACE_BASE}/instructions.txt"

with open(instructions_file, "r") as file:
    partial_instructions = file.read()

instructions = partial_instructions + additional_notes

# Add DeepSeek-specific hints if using DeepSeek model
if "deepseek" in MODEL.lower():
    hints_deepseek_path = Path(__file__).parent.parent / "instructions" / "hints_deepseek.txt"
    if hints_deepseek_path.exists():
        with open(hints_deepseek_path, "r") as f:
            instructions += "\n\n" + f.read()


@task
def pb_task():
    # 支持环境变量控制
    knowledge_enabled = os.environ.get("KNOWLEDGE_ENABLED", "false").lower() == "true"

    research_tools = []
    if knowledge_enabled:
        research_tools = [
            get_overview(),          # 工具1：获取论文概览
            get_similar_techniques(),  # 工具2：获取相似技术
            get_full_techniques()      # 工具3：获取完整技术实现
        ]

    if ITERATIVE_AGENT:
        solver = basic_agent_iterative(
            tools=[bash(), read_file_chunk()] + research_tools,
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT, # type: ignore
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
        )
    else:
        solver = basic_agent_plus(
            tools=[bash(), python(), read_file_chunk(), search_file()] + web_browser() + research_tools,
            max_attempts=1,
            disallow_submit=DISALLOW_SUBMIT, # type: ignore
            real_time_limit=int(float(MAX_TIME_IN_HOURS) * 60 * 60),
        )
    return Task(
        dataset=[Sample(input=instructions)],
        solver=solver,
        sandbox="local",
    )


def run():
    eval(pb_task(), model=MODEL, trace=True, reasoning_effort="high", reasoning_tokens=25600)


if __name__ == "__main__":
    run()
