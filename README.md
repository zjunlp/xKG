# ResearchKG
- 创建`python=3.10`环境

- 创建`.env`文件并将`.env.example`中的内容拷贝过去并填充完
  
- cmd所在的目录：`ResearchKG`，一切命令都在这个目录下执行

- 执行的命令：`python -m xkg.run_kg` 模块化运行 以便适配到其它项目中

- 如何一键集成到外部：
  - 根据`xkg/interface/retrieve.py`中定义的接口 直接调用
  - 使用方法：`/disk/disk_20T/luoyujie/ResearchKG/xkg/run.py`中咋用就咋用 非常方便
  - 用的话直接import函数就行，类似：
    ```
    from ResearchKG.xkg.schema import Node
    from ResearchKG.xkg.interface.retrieve import (
        initialize_kg,
        find_node_by_paper_title,
        find_similar_techniques,
        find_similar_papers
    )
    ```


# PaperBench

## 环境配置

本项目已为 Mac（ARM64）和 Linux（x86_64）平台进行了优化适配。按以下步骤配置 PaperBench 评估环境：

### 前置要求
- Docker Desktop（已安装并运行）
- Python 3.11
- Git

### 步骤 1：创建 PaperBench 专用环境

```bash
# 创建 Python 3.11 的 conda 环境
conda create -n paperbench python=3.11 -y
conda activate paperbench
```

### 步骤 2：安装 PaperBench 依赖包

```bash
# 进入 PaperBench 目录
cd experiments/paperbench/project

# 安装核心模块（按顺序安装）
pip install -e nanoeval/
pip install -e alcatraz/
pip install -e nanoeval_alcatraz/
pip install -e paperbench/

# 安装额外工具
pip install inspect-ai
```

### 步骤 3：配置环境变量

```bash
# 在 experiments/paperbench/project/paperbench 目录下
cd paperbench

# 复制并编辑环境配置
cp .env.example .env
# 编辑 .env 文件，填入你的 API keys:
# - OPENAI_API_KEY: 你的 OpenAI API 密钥
# - OPENAI_BASE_URL: OpenAI API 端点（可选，使用代理时设置）
# - GRADER_OPENAI_API_KEY: 评分器使用的 API 密钥（可选，默认使用 OPENAI_API_KEY）

# 配置 agent 环境变量
cp paperbench/agents/agent.env.example paperbench/agents/agent.env
# 编辑 paperbench/agents/agent.env，填入:
# - OPENAI_API_KEY: agent 使用的 API 密钥
# - HF_TOKEN: HuggingFace 令牌（某些论文需要）
```

### 步骤 4：构建 Docker 镜像

```bash
# 在 experiments/paperbench/project/paperbench 目录下执行
bash paperbench/scripts/build-docker-images.sh
```

脚本会自动检测你的平台（Mac ARM64 或 Linux x86_64）并构建相应的镜像：
- `pb-env:latest` - 基础环境
- `dummy:latest` - 测试 agent
- `aisi-basic-agent:latest` - BasicAgent
- `pb-grader:latest` - 评分器
- `pb-reproducer:latest` - 代码执行环境

### 步骤 5：验证环境

```bash
# 测试 dummy agent（验证端到端流程）
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench
conda activate paperbench
python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=dummy \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=dummy:latest \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

### 步骤 6：运行 PaperBench 评估

```bash
# 运行 BasicAgent（开发配置，5 分钟超时）
python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    runner.recorder=nanoeval.json_recorder:json_recorder

# 运行 Code-Dev 评估（跳过代码执行和结果验证，仅检查代码开发）
# 添加 paperbench.judge.code_only=True 标志来降低成本和执行时间
python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    paperbench.judge.code_only=True \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

### 查看评估结果

```bash
# 评估结果保存在 runs 目录下
# 每个 run 包含：
#   - metadata.json: 元信息
#   - pb_result.json: 评分结果
#   - run.log: 运行日志
#   - status.json: 运行状态

ls runs/
```

## 常见问题

### Docker 镜像构建失败
- 确保 Docker Desktop 正在运行
- 清理旧镜像：`docker system prune -a`
- 重新运行构建脚本

### API 调用超时
- 检查网络连接
- 确认 OPENAI_API_KEY 有效
- 如果使用代理，确认 OPENAI_BASE_URL 配置正确

### GPU 不可用
- 当前配置针对 CPU 优化
- 如需 GPU 支持，修改 Docker 配置中的 `is_nvidia_gpu_env: True`

## 平台适配说明

本项目已自动适配以下平台：
- **Mac ARM64（Apple Silicon）**: 自动检测并使用 arm64 Docker 镜像和 aarch64 Miniconda
- **Linux x86_64**: 自动检测并使用 amd64 Docker 镜像和 x86_64 Miniconda

无需手动配置，构建脚本会自动处理平台差异。