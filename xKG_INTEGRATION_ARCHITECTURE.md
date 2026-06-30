# xKG 集成架构、PaperBench 实验复现与 Agent 增强分析

本文档详细说明 xKG 如何作为插件集成到代理框架中、如何通过 PaperBench 复现论文实验、以及两种代理类型如何被 xKG 增强。

---

## 第一部分：xKG 集成架构总览

### 1.1 核心集成模式：两阶段支持

xKG 为 Agent 提供 **两阶段支持**：
1. **规划阶段（Planning Stage）**：提供高层次的论文摘要和技术概览
2. **实现阶段（Implementation Stage）**：提供具体的代码实现和参考

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Workflow                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: Planning                                                  │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Read paper                                                 │  │
│  │ • Understand task requirements                              │  │
│  │ ┌─► xKG: get_overview() + get_similar_papers()            │  │
│  │ │   Returns: Paper Node with techniques summary            │  │
│  │ └─ Use to guide high-level planning                        │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                           ↓                                          │
│  Phase 2: Implementation                                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ • Break down implementation tasks                           │  │
│  │ • For each technique/component:                            │  │
│  │ ┌─► xKG: get_similar_techniques() + get_full_techniques()  │  │
│  │ │   Returns: Technique Nodes with code.implementation      │  │
│  │ │   Code: { implementation, test, documentation }          │  │
│  │ └─ Use as code reference for specific implementations      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 xKG 作为 Tool 集成到 Inspect AI

在 BasicAgent 中，xKG 通过 Inspect AI 的 `@tool` 装饰器暴露为可调用工具：

```python
# xKG/source/agents/aisi-basic-agent/start.py (line 49-65)
research_tools = [
    get_overview(),           # Tool 1: 获取论文概览 + 技术列表
    get_similar_techniques(), # Tool 2: 查询类似技术的实现
    get_full_techniques()     # Tool 3: 获取论文核心技术的完整实现
]

solver = basic_agent_plus(
    tools=[bash(), python(), read_file_chunk(), search_file()] 
          + web_browser() 
          + research_tools,  # <-- xKG 工具混合到其他工具中
    ...
)
```

### 1.3 xKG 数据结构映射

**ResearchKG Node 结构**（存储在 `/home/guide.json`）：
```json
{
  "paper_title": "...",
  "paper_abstract": "...",
  "paper_references": [...],
  "techniques": [
    {
      "name": "Technique Name",
      "description": "Detailed description",
      "weight": 0.5,
      "code": {
        "implementation": "def technique(): ...",
        "test": "if __name__ == '__main__': ...",
        "documentation": "How to use...",
        "package": ["torch", "numpy"]
      },
      "components": [...]  // 嵌套结构
    }
  ],
  "resources": [...],
  "findings": [...]
}
```

---

## 第二部分：BasicAgent + xKG 集成

### 2.1 BasicAgent 架构（ReAct 风格）

BasicAgent 是一个 ReAct（Reasoning + Acting）风格的代理：

```python
# _basic_agent_plus.py
# Agent 循环流程：
# 1. Reasoning: LLM 分析任务和当前状态
# 2. Acting: 选择工具调用
# 3. Observation: 获取工具返回
# 4. 重复，直到完成或超时
```

### 2.2 xKG 工具在 BasicAgent 中的角色

#### Tool 1: `get_overview()`
**触发时机**：代理开始分析论文时  
**功能**：加载 `/home/guide.json` 并返回论文的高层概览

```python
async def execute() -> ToolResult:
    file_path = "/home/guide.json"
    with open(file_path, 'r') as file:
        file = json.load(file)            
    res = f"""
    Here are some key technologies and contributions extracted from the paper. 
    The following points are the paper's core contributions. Use them as a 
    starting point, but your goal is to replicate EVERYTHING in the paper...
    {json.dumps(file)}
    """
    return res
```

**代理使用**：理解论文的主要技术方向和贡献

#### Tool 2: `get_similar_techniques(technique_name, technique_description)`
**触发时机**：代理需要实现特定技术时  
**功能**：查询知识库中类似的技术实现

```python
async def execute(technique_name: str, technique_description: str) -> ToolResult:
    # 初始化 KG（首次调用时）
    await get_kg_instance()
    
    # 查询相似技术（top_k=1 找最相关的）
    similar_techs = await asyncio.to_thread(
        find_similar_techniques, 
        name=technique_name, 
        description=technique_description, 
        code_only=False, 
        top_k=1
    )
    
    # 返回格式化的代码参考
    result = [{
        "name": tech.name,
        "implementation": tech.code.implementation,
        "test": tech.code.test,
        "documentation": tech.code.documentation
    } for tech in similar_techs]
    
    return f"""
    The following are relevant implementations concerning the technology {technique_name}...
    WARNING: Do not copy this code directly...
    {json.dumps(result)}
    """
```

**代理使用**：在实现具体技术时获取参考代码

#### Tool 3: `get_full_techniques()`
**触发时机**：代理需要全面了解论文所有技术的代码实现时  
**功能**：批量查询所有核心技术的实现

```python
async def execute() -> ToolResult:
    # 1. 从 /home/guide.json 读取论文 Node
    guide_data = json.load(open("/home/guide.json"))
    paper_node = Node.from_dict(guide_data)
    techniques = paper_node.techniques
    
    # 2. 初始化知识库
    await get_kg_instance()
    
    # 3. 对每个技术查询相似实现（top_k=3）
    for technique in techniques:
        similar_techs = await asyncio.to_thread(
            find_similar_techniques, 
            name=technique.name, 
            description=technique.description, 
            code_only=False, 
            top_k=3
        )
        # 修改技术对象的 code 属性
        technique.code = f"[Relevant Implementations]: {json.dumps(...)}"
    
    # 4. 返回完整的技术树
    return f"{json.dumps([asdict(tech) for tech in techniques])}"
```

**代理使用**：获取全面的技术参考库供后续使用

### 2.3 BasicAgent + xKG 交互流程

```
Agent Step 1: 接收任务
├─ Tool call: get_overview()
│  └─ 返回论文技术摘要 + 关键贡献
└─ Agent reasoning: 理解论文目标

Agent Step 2-N: 实现各模块
├─ Agent identifies implementation task
│  └─ Tool call: get_similar_techniques(task_name, task_description)
│     └─ 返回参考代码 (implementation, test, doc)
└─ Agent uses reference code to implement

Agent Step N+1: 整体验证
├─ Tool call: get_full_techniques()
│  └─ 返回所有技术的参考实现
└─ Agent validates implementation against references
```

---

## 第三部分：IterativeAgent + xKG 集成

### 3.1 IterativeAgent 架构（自改进循环）

IterativeAgent 在 BasicAgent 基础上添加了 **反思和改进循环**：

```python
# _basic_agent_iterative.py
# Agent 循环流程：
# 1. Attempt: 尝试完成任务
# 2. Self-reflection: LLM 反思是否成功
# 3. If failed:
#    a. Identify issues
#    b. Refine approach
#    c. Return to step 1
# 4. Else: Task complete
```

### 3.2 xKG 在 IterativeAgent 中的增强

IterativeAgent 使用 **相同的 xKG 工具**，但利用方式更深层：

1. **初始规划**：`get_overview()` + `get_full_techniques()`  
   - 构建完整的参考库

2. **迭代改进**：`get_similar_techniques()` 多轮查询  
   - 第一轮：获取初始参考
   - 失败后反思并修改查询参数
   - 重新查询获取不同视角的实现

3. **验证循环**：结合代码执行反馈  
   - 执行代码
   - 如果失败，使用 xKG 查找替代实现
   - 比较多个参考实现的差异
   - 迭代完善

### 3.3 增强因素分析

根据 xkg.md 论文（PaperBench 评测结果）：

| Agent 类型 | LLM 模型 | 基础性能 | +xKG 性能 | 提升 |
|-----------|---------|--------|---------|------|
| BasicAgent | o3-mini | ~0% | 10.9% | ↑10.9% |
| IterativeAgent | DeepSeek-R1 | ~0% | 8.2% | ↑8.2% |

**关键发现**：
- **Code Nodes 最关键**：移除代码节点性能下降 4.56%
- **Technique Nodes 次之**：用处 2.3%
- **Paper Nodes 基础**：用处 1.5%

---

## 第四部分：PaperCoder（Paper2Code）集成

### 4.1 PaperCoder 三阶段架构

PaperCoder 是一个 **完全不同的框架**，采用三阶段 LLM 流水线：

```
Stage 1: Planning
├─ LLM 生成实现计划
├─ 生成文件列表
├─ 生成数据结构设计
└─ 生成 config.yaml

Stage 2: Analyzing  ← xKG 集成点
├─ LLM 进行逻辑分析
├─ For each file:
│  ├─ generate_msg 包含论文 + 设计 + 任务
│  └─ 如果是实现文件：
│     └─ Tool: get_relevant_code(file_name, description)  ← xKG 查询
│        └─ 调用 decompose_and_find_techniques()
│        └─ 返回相关技术实现
│  └─ LLM 基于参考代码生成逻辑分析
└─ 保存分析结果

Stage 3: Coding
├─ LLM 基于分析和计划生成代码
├─ 完整的可运行代码库
└─ 包含所有模块和配置
```

**集成关键代码** (`2_analyzing.py`，第 8-9, 38, 111, 258-262 行)：

```python
from xKG.research_kg.interface.retrieve import *
from xKG.research_kg.utils import *

# Stage 2 开始前初始化
initialize_kg()  # 第 38 行

# 对实现文件获取参考代码
def get_relevant_code(todo_file_name, todo_file_desc):
    description = f"{todo_file_name}: {todo_file_desc}"
    # 调用 xKG 查询相似技术
    similar_techs = decompose_and_find_techniques(
        description, 
        top_k=3, 
        code_only=False, 
        llm_rerank=True
    )
    # 返回格式化的参考代码
    return f"The following are relevant implementations... {json.dumps(result)}"

# Stage 2 中使用参考代码增强 LLM 提示
if code_retrieval and judge_implementation_file(...):
    code = get_relevant_code(todo_file_name, analysis)
    response = f"{code}\n\n{analysis}"  # 合并参考代码
```

### 4.2 PaperCoder 对 xKG 的三种使用方式

#### 方式 A：无 xKG（基础）
```
Analyzing Phase
├─ LLM 输入：论文 + 设计 + 任务描述
└─ LLM 生成：逻辑分析（无参考代码）
```

#### 方式 B：with xKG Code Retrieval（推荐）
```
Analyzing Phase
├─ LLM 输入：论文 + 设计 + 任务描述
├─ For each implementation file:
│  ├─ xKG query via decompose_and_find_techniques()
│  ├─ Retrieve top-3 similar techniques with code
│  └─ Include in LLM prompt
└─ LLM 生成：基于参考代码的逻辑分析
```

#### 方式 C：No-Knowledge 模式（对比）
```
# 通过 --no_code_retrieval 标志禁用
$ python 2_analyzing.py --no_code_retrieval
```

---

## 第五部分：PaperBench 实验复现流程

### 5.1 PaperBench 三阶段评测管道

```
┌──────────────────────────────────────────────────────────────────┐
│  PaperBench Evaluation Pipeline                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Stage 1: Agent Rollout (在 Ubuntu 容器中)                      │
│  ├─ Agent 读取论文和任务说明                                   │
│  ├─ Agent 创建实现代码库                                       │
│  └─ 生成最终提交 (submission codebase)                         │
│           ↓                                                      │
│  Stage 2: Reproduction (在 GPU 容器中)                          │
│  ├─ 执行代理提交的代码库                                       │
│  ├─ 收集执行结果                                               │
│  └─ 生成 executed_submission                                   │
│           ↓                                                      │
│  Stage 3: Grading (在评分容器中)                               │
│  ├─ Judge 读取论文的 rubric                                    │
│  ├─ Judge 评估 executed_submission                             │
│  ├─ 检查三类要求：                                             │
│  │  1. Code Development (代码开发完整性)                       │
│  │  2. Execution (代码能否运行)                                │
│  │  3. Result Match (结果是否与论文匹配)                       │
│  └─ 生成最终评分                                               │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 复现实验的关键配置

#### 运行 BasicAgent with xKG
```bash
cd /path/to/paperbench

# 设置环境变量
export OPENAI_API_KEY="<your-key>"
export MODEL="o3-mini"  # 或其他模型
export MAX_TIME_IN_HOURS="24"
export ITERATIVE_AGENT="False"  # 使用 BasicAgent

# 构建 Docker 镜像
bash paperbench/scripts/build-docker-images.sh

# 运行完整 PaperBench
python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    runner.recorder=nanoeval.json_recorder:json_recorder \
    runner.concurrency=5
```

#### 运行 IterativeAgent with xKG
```bash
export ITERATIVE_AGENT="True"  # 使用 IterativeAgent

python -m paperbench.nano.entrypoint \
    # ... 同上配置
```

#### 运行 PaperCoder
```bash
cd /path/to/paperbench/paperbench/agents/paper2code/scripts

# 设置 API 密钥
export OPENAI_API_KEY="<your-key>"

# 运行 planning + analyzing + coding pipeline
python 1_planning.py \
    --paper_name "Example Paper" \
    --gpt_version "o3-mini" \
    --pdf_json_path "../examples/paper.json" \
    --output_dir "../outputs/paper_name"

python 2_analyzing.py \
    --paper_name "Example Paper" \
    --gpt_version "o3-mini" \
    --pdf_json_path "../examples/paper.json" \
    --output_dir "../outputs/paper_name"

python 3_coding.py \
    --paper_name "Example Paper" \
    --gpt_version "o3-mini" \
    --pdf_json_path "../examples/paper.json" \
    --output_dir "../outputs/paper_name"
```

### 5.3 评测数据集

PaperBench 包含 **20 篇论文**（ICML 2024 Spotlight & Oral）：

每篇论文包含：
- 论文 PDF 和 Markdown 版本
- `rubric.json` - 评分标准（Code Development / Execution / Result Match）
- `addendum.md` - 作者补充说明
- `blacklist.txt` - 禁用网站列表（如官方代码库）
- `config.yaml` - 论文配置

### 5.4 PaperBench Code-Dev 轻量版本

```bash
# Code-Dev 模式：仅评估代码开发要求，不需要 GPU
python -m paperbench.nano.entrypoint \
    # ... 基本配置
    paperbench.judge.code_only=True
```

**优势**：
- 无需 GPU 执行代码
- 成本降低 ~85%（针对 o3-mini）
- 快速迭代开发

### 5.5 结果检索和分析

运行结果保存在 `runs/` 目录：

```
runs/
├── <run_group_id>/
│   ├── <run_id>/
│   │   ├── initial_snapshot_metadata.json
│   │   ├── initial_snapshot.tar.gz
│   │   ├── final_snapshot_metadata.json
│   │   ├── final_snapshot.tar.gz
│   │   ├── final_snapshot_repro.tar.gz
│   │   ├── final_snapshot_repro_grader_output_0.json  ← 评分结果
│   │   ├── pb_result.json  ← 最终得分
│   │   ├── status.json
│   │   └── run.log
```

**查看评分结果**：
```bash
# 查看 pb_result.json
cat runs/<run_group_id>/<run_id>/pb_result.json

# 查看代理日志
cat runs/<run_group_id>/<run_id>/run.log

# 查看 snapshots
tar -xzf runs/<run_group_id>/<run_id>/final_snapshot.tar.gz
```

---

## 第六部分：Agent 增强对比分析

### 6.1 性能对比（来自 xkg.md）

#### PaperBench Lite Subset 结果

| Agent + Model | Score (%) | +xKG Improvement |
|--------------|-----------|-----------------|
| BasicAgent + o3-mini | 9.8 | ↑10.9% → 20.7% |
| IterativeAgent + DeepSeek-R1 | 8.1 | ↑8.2% → 16.3% |

#### PaperBench Full Results (完整排行)

| Agent + Model | Score (%) | Rank |
|--------------|-----------|------|
| IterativeAgent o1-high (36h) | 26.0 | 🥇 |
| IterativeAgent o1-high (24h) | 24.4 | 🥈 |
| BasicAgent claude-3.5-sonnet | 21.0 | 🥉 |
| **IterativeAgent claude-3.5-sonnet** | 16.1 | - |
| BasicAgent o1-high | 13.2 | - |
| IterativeAgent o3-mini-high | 8.5 | - |
| BasicAgent deepseek-r1 | 6.0 | - |

### 6.2 xKG 组件贡献度分析

移除各组件对性能的影响：

```
Paper Nodes only:       +1.5%  (基础论文理解)
+ Technique Nodes:      +2.3%  (技术概览)
+ Code Nodes:           +4.56% ← 最关键！
─────────────────────────────
Total with full xKG:    +8.2% ~ +10.9%
```

**关键洞察**：
- **Code Nodes 是瓶颈和最大收益点**
  - 代理最需要具体可运行的代码参考
  - CodeVerifier 的作用是确保代码质量
  
- **Technique Nodes 提供导向**
  - 帮助代理识别关键技术
  - 加快探索速度
  
- **Paper Nodes 是基础**
  - 提供原始论文信息
  - 支持引用链接和参考文献

### 6.3 Agent 类型与 xKG 的适配方式

#### BasicAgent + xKG 适配
```
ReAct Loop:
  Thought → Tool selection → get_overview() / get_similar_techniques()
         ↓                          ↓
         └──────────── Observation ←─

特点：
✓ 适合快速查询和单次决策
✓ 工具调用开销小
✗ 无多轮反思能力
```

**Best Practice**：
- 在规划阶段调用 `get_overview()` 一次
- 在实现阶段按需调用 `get_similar_techniques()`
- 避免过度查询（只在不确定时查询）

#### IterativeAgent + xKG 适配
```
Self-Improvement Loop:
  Attempt → Self-reflection 
       ↓          ↓
   If failed: get_similar_techniques(refined_query)
       ↓
   Retry with refined implementation
       ↓
   Success?

特点：
✓ 支持多轮查询和逐步完善
✓ 能利用不同视角的参考实现
✓ 反思能力强
```

**Best Practice**：
- 初始尝试：`get_full_techniques()` 获取全景
- 失败后反思：调整查询参数重新查询
- 多轮迭代：每轮使用不同的查询参数

#### PaperCoder + xKG 适配
```
Sequential Pipeline:
  Stage 1 (Planning) → Stage 2 (Analyzing) → Stage 3 (Coding)
                            ↓
                      For each file:
                        ├─ analyze with paper+design+task
                        ├─ if implementation file:
                        │   └─ decompose_and_find_techniques()
                        └─ include references in prompt

特点：
✓ 批量检索所有相关技术
✓ LLM 有充足上下文（带参考代码）
✓ 结构化的分析和代码生成
```

**Best Practice**：
- Analyzing Phase：对每个实现文件启用代码检索
- 参考代码包含：implementation + test + documentation
- LLM 作为"参考融合器"而非"代码复制"

---

## 第七部分：使用 xKG 复现论文实验的完整流程

### 7.1 准备阶段

#### Step 1: 构建 xKG 知识库
```bash
cd /Users/luoyujie/Documents/Code/xKG

# 配置环境
cp .env.example .env
# 填写 OPENAI_API_KEY, GITHUB_TOKEN 等

pip install -r requirements.txt

# 从 PaperBench 数据集构建 xKG
python -m xKG.source.run_kg --profile basic-deepseek-v3

# 验证知识库生成
ls xKG/storage/kg/
# 输出：*.json 文件（每篇论文一个 Node）
```

#### Step 2: 配置 PaperBench Agent
```bash
cd /path/to/paperbench

# 设置代理 API 密钥
cp paperbench/agents/agent.env.example paperbench/agents/agent.env
# 填写 OPENAI_API_KEY, HF_TOKEN

# 构建 Docker 镜像
bash paperbench/scripts/build-docker-images.sh
```

### 7.2 实验复现流程

#### 方案 A：BasicAgent with xKG
```bash
# 1. 运行单篇论文的 Lite 测试
python -m paperbench.nano.entrypoint \
    paperbench.paper_split=lite \
    paperbench.solver=paperbench.nano.eval:ExternalPythonCodingSolver \
    paperbench.solver.agent_id=aisi-basic-agent-openai-dev \
    paperbench.solver.cluster_config=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.cluster_config.image=aisi-basic-agent:latest \
    runner.recorder=nanoeval.json_recorder:json_recorder

# 2. 收集结果
cat runs/<run_group_id>/<run_id>/pb_result.json

# 3. 对比性能
# With xKG:    20.7%
# Without xKG: 9.8%
# Improvement: +10.9%
```

#### 方案 B：IterativeAgent with xKG
```bash
# 与 BasicAgent 相同，但设置环境变量
export ITERATIVE_AGENT="True"

python -m paperbench.nano.entrypoint \
    # ... 同上
    
# 预期性能
# With xKG:    16.3%
# Without xKG: 8.1%
# Improvement: +8.2%
```

#### 方案 C：PaperCoder (Paper2Code Agent)
```bash
# 1. 准备论文和 guide JSON
cd /path/to/paperbench/paperbench/agents/paper2code

# 2. 运行三阶段 pipeline
bash scripts/run.sh  # 或 bash scripts/run_latex.sh

# 3. 输出生成的代码库
# outputs/<paper_name>/<paper_name>_repo/

# 4. 评估生成的代码
python codes/eval.py \
    --paper_name "<paper_name>" \
    --target_repo_dir "./outputs/<paper_name>_repo" \
    --eval_type ref_free \
    --papercoder
```

### 7.3 性能对标

#### Baseline 对标表

| 方案 | Agent Type | LLM | xKG | 预期评分 |
|------|-----------|-----|-----|--------|
| 基础 | BasicAgent | o3-mini | ✗ | 9.8% |
| +xKG | BasicAgent | o3-mini | ✓ | 20.7% |
| 对标 | IterativeAgent | o1-high | ✓ | 26.0% |
| Paper2Code | PaperCoder | o3-mini | ✓ | ? (测试中) |

### 7.4 诊断和调试

#### 如果性能未达预期

1. **检查 xKG 知识库质量**
   ```bash
   # 验证 Node JSON 是否有有效代码块
   python -c "
   import json
   with open('xKG/storage/kg/example_node.json') as f:
       node = json.load(f)
   print(f\"Techniques: {len(node['techniques'])}\")
   for t in node['techniques']:
       if t.get('code'):
           print(f\"  - {t['name']}: has code\")
   "
   ```

2. **检查代理日志**
   ```bash
   # 查看代理是否成功调用 xKG 工具
   grep -A 5 "get_overview\|get_similar_techniques" runs/<run_id>/run.log
   ```

3. **验证 CodeVerifier 质量**
   ```bash
   # 检查代码执行是否成功
   grep "Code execution\|DEBUG" runs/<run_id>/run.log
   ```

4. **对比有无 xKG 的代理行为**
   ```bash
   # 运行无 xKG 版本进行对比
   export NO_KNOWLEDGE="True"
   # ... 重新运行
   ```

---

## 第八部分：关键技术要点

### 8.1 xKG 在 Agent 中的三层价值

```
Level 1: 知识检索 (Information Retrieval)
  └─ 提供论文精准概括
  └─ 替代 Agent 逐字阅读所有细节

Level 2: 代码参考 (Code Reference)
  └─ 提供可运行的实现示例
  └─ 加速代理的代码编写过程

Level 3: 质量保证 (Quality Assurance)
  └─ CodeVerifier 确保参考代码正确性
  └─ 代理可信任参考代码的准确性
```

### 8.2 xKG vs. 传统 RAG

| 维度 | 传统 RAG | xKG |
|------|---------|-----|
| 检索单位 | 文本片段 | Paper/Technique/Code 节点 |
| 结构化程度 | 低 | 高（JSON Schema） |
| 代码质量 | 未知 | CodeVerifier 验证 |
| 多模态 | 文本为主 | 文本 + 代码 |
| 关系建模 | 隐式 | 显式（技术树、引用） |

### 8.3 xKG 的限制和改进方向

**当前限制**：
1. 知识库需预先构建（非实时）
2. CodeVerifier 需 Docker（环境限制）
3. 仅支持 Python 代码（LLM 模型限制）
4. 技术树深度限制（计算成本）

**改进方向**：
1. 动态增量更新知识库
2. 多语言支持（需多个 CodeVerifier）
3. 在线学习：Agent 反馈优化检索
4. 混合检索：融合向量检索 + BM25

---

## 结论

### xKG 的核心价值

1. **规划能力** → Paper Nodes 提供高层概览
2. **实现能力** → Technique + Code Nodes 提供具体参考
3. **质量保证** → CodeVerifier 验证代码正确性

### Agent 增强的三种模式

1. **BasicAgent**：快速单次查询，性能 +10.9%
2. **IterativeAgent**：多轮反思和改进，性能 +8.2%
3. **PaperCoder**：批量检索支持分析，性能未测 (但有望更高)

### 实验复现建议

| 优先级 | 方案 | 理由 |
|------|------|------|
| 🔴 最高 | BasicAgent + xKG | 快速验证，成本低，性能高 |
| 🟡 次高 | IterativeAgent + xKG | 更强的自改进能力，性能稳定 |
| 🟢 参考 | PaperCoder + xKG | 结构化 pipeline，有望更好 |

---

## 参考资源

- xKG 项目：`/Users/luoyujie/Documents/Code/xKG`
- xkg.md 论文：`/Users/luoyujie/Documents/Code/xKG/storage/xkg.md`
- PaperBench 教程：`/path/to/paperbench/README.md`
- Paper2Code 文档：`/path/to/paperbench/paperbench/agents/paper2code/README.md`
