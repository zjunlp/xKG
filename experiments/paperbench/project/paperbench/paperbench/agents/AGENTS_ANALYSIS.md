# xKG Agent Framework 架构解读

你的论文 "Executable Knowledge Graphs as Scientific Knowledge Representations" 提出了一个**纸张中心的知识图谱 (XKG)** 系统，用于增强 LLM 代理的论文复现能力。下面是你在 PaperBench 框架中集成 xKG 的三个代理实现的详细解读。

---

## 一、核心论文贡献 (从 xkg.md)

### 问题背景
- 论文复现困难：实现细节遗漏、代码不可得、背景知识分散
- 现有 RAG 方法失效：无法捕获潜在的技术细节

### xKG 核心创新
**结构**：分层的多关系图
- **Paper Node**: 元数据 + Technique nodes + Code nodes
- **Technique Node**: 学术概念 + 可选子节点  
- **Code Node**: (实现代码 σ, 测试脚本 τ, 文档 δ)

**两种边**:
- **Structural Edge**: 技术节点间的依赖关系
- **Implementation Edge**: 技术直接链接到可执行代码

**构造流程** (3步 + 动态语料库):
1. **Technique Extraction**: 用 o4-mini 分解论文方法为技术树，用 RAG 补充定义
2. **Code Modularization**: 从仓库检索代码片段，合成可执行 Code Nodes
3. **Knowledge Filtering**: 仅保留有实现代码的技术（可验证性原则）

**性能**: 在 PaperBench 上获得 **+10.9% (o3-mini)**、**+8.2% (DeepSeek-R1)** 的提升

---

## 二、三个代理框架集成

你在 `/agents/aisi-basic-agent/` 中创建了两个代理变体，以及集成了 `paper2code` 流程。关键区别是 **有无 xKG** 的知识检索。

### 2.1 BasicAgent (`_basic_agent_plus.py`)

**类型**: ReAct loop，一次性生成完整代码  
**工具集**: 
- `bash()`, `python()`, `read_file_chunk()`, `search_file()`, `web_browser()`
- **可选**: 知识库工具 (get_overview, get_similar_techniques, get_similar_papers)

**执行逻辑**:
```
系统提示 → 初始化工具 → 主循环:
  1. 生成 LLM 响应
  2. 处理工具调用
  3. 检查提交（end_task）
  4. 反复迭代直到时间或消息限制
```

**关键特性**:
- 消息裁剪：当 Sonnet 接近 900 条消息时主动剪枝
- 时间管理：区分"总运行时间"和"生产时间"（排除 API 重试）
- 周期提示：每 5 步提醒模型时间剩余、工具使用

**与 xKG 的区别**:
- **Without xKG**: 直接从纸张文本中学习
- **With xKG** (+xKG 变体): 在工具列表中注册 get_overview, get_similar_techniques, get_full_techniques

---

### 2.2 IterativeAgent (`_basic_agent_iterative.py`)

**类型**: 多轮交互式代理  
**核心差异**:
```python
# BasicAgent: 保持对话历史
state.messages.append(state.output.message)

# IterativeAgent: 每轮重新注入指令，但不记入历史
conversation = copy.deepcopy(state.messages) + [
    ChatMessageUser(content=continue_user_message)
]
state.output = await model.generate(input=conversation, ...)
```

**效果**: 
- 强化每一步的目标聚焦（不会被长对话历史淹没）
- 适合长时间任务（8+ 小时），频繁重启思考

**配置例子** (从 config.yaml):
```yaml
# Basic: 一次完整执行
aisi-basic-agent-my-o3:
  MODEL: openai/o3-mini-2025-01-31
  MAX_TIME_IN_HOURS: 1

# Iterative: 周期性重新评估
aisi-basic-agent-iterative-my-o3:
  MODEL: openai/o3-mini-2025-01-31
  MAX_TIME_IN_HOURS: 1
  ITERATIVE_AGENT: true
```

---

### 2.3 PaperCoder (`paper2code/codes/` 目录)

**类型**: 三阶段管道（规划 → 分析 → 编码）  
**创新点**: 将论文复现分解为明确的 LLM 链式推理步骤

#### 阶段 1: 规划 (`1_planning.py`)
```
输入: 论文 (JSON/LaTeX)
↓
4 个提示链式调用:
  - 总体计划 (Overall plan)
  - 架构设计 (Architecture design)
  - 逻辑设计 (Logic design)
  - 配置生成 (Config file generation)
↓
输出: planning_response.json, planning_trajectories.json
```

**LLM 提示特点**:
- 不使用工具，纯文本推理
- 要求输出 JSON/YAML/Mermaid
- 强调"DO NOT FABRICATE DETAILS"

#### 阶段 2: 分析 (`2_analyzing.py`)

**新增: xKG 知识检索**
```python
# 对每个文件，判断是否为"实现文件"
if judge_implementation_file(todo_file_name, description):
    # 调用 get_relevant_code()
    similar_techs = decompose_and_find_techniques(
        description, 
        top_k=3, 
        code_only=False, 
        llm_rerank=True
    )
```

**关键行为**:
- 使用 `xKG.research_kg.interface.retrieve` 初始化知识图谱
- 分解任务描述，检索相似技术
- 为每个文件生成"Logic Analysis"，作为编码指导

#### 阶段 3: 编码 (`3_coding.py`)

**输入**:
- 论文内容
- 三个阶段的 LLM 输出（作为 RAG 上下文）
- 配置文件
- 已生成的前置代码

**生成逻辑**:
```
对于每个待生成文件:
  1. 检索已完成的前置代码
  2. 构造包含所有上下文的提示
  3. 调用 LLM 生成该文件代码
  4. 提取 Python 代码块
  5. 保存到 output_repo_dir
```

**与 xKG 的区别**:
- **run_knowledge.sh**: 启用代码检索 (`--no_code_retrieval` 未设置)
- **run_no_knowledge.sh**: 禁用代码检索 (`--no_code_retrieval` 设置)
- **run_no_code.sh**: 连分析也跳过，仅进行规划和编码

---

## 三、xKG 集成点 (_knowledge.py)

这是你的核心创新集成模块，暴露 4 个工具给代理：

### 工具 1: `get_overview()`
```python
# 读取 /home/guide.json (即 xKG 论文节点)
# 返回: 论文的核心贡献摘要 (JSON 格式)
```
**用途**: 帮助代理快速理解论文全局

### 工具 2: `get_similar_techniques(technique_name, technique_description)`
```python
# 调用 find_similar_techniques(name=..., description=..., top_k=1)
# 返回: 3 个相似技术的实现代码 (implementation, test, documentation)
```
**用途**: 代理可在任何时候查询相似技术的参考实现

### 工具 3: `get_similar_papers()`
```python
# 调用 find_similar_papers(paper=..., top_k=3, return_code=False)
# 返回: 相似论文的技术树、代码结构、概述
```
**用途**: 获取相关论文作为背景参考

### 工具 4: `get_full_techniques()`
```python
# 对 guide.json 中的所有技术
# 逐个调用 find_similar_techniques(top_k=3)
# 返回: 论文所有技术 + 每个技术的参考实现
```
**用途**: 一次性获取论文的完整参考代码库

**关键设计**:
- 异步锁 (`asyncio.Lock`) 保证 KG 单例初始化
- `to_thread()` 在后台线程运行同步的 KG 函数
- 警告信息明确：不要直接复制代码，要灵活适配纸张设置

---

## 四、核心差异对比

| 维度 | Without xKG | With xKG |
|------|------------|----------|
| **知识来源** | 仅纸张文本 | 纸张 + 相似技术实现 + 相似论文背景 |
| **BasicAgent 工具** | bash, python, file_read | 上述 + get_overview, get_similar_techniques, get_full_techniques |
| **IterativeAgent 工具** | bash, file_read | 上述 + 知识工具 |
| **PaperCoder 分析** | 纯 LLM 推理 | LLM + 代码检索 (judge_implementation_file → decompose_and_find_techniques) |
| **执行脚本** | run_no_knowledge.sh | run_knowledge.sh 或 run.sh |
| **预期性能提升** | 基准 | +10.9% ~ +24.8% (按模型) |

---

## 五、执行流程

### 配置示例 (start.py)
```python
@task
def pb_task():
    research_tools = [
        get_overview(),
        get_similar_techniques(),
        get_full_techniques()
    ]
    
    if ITERATIVE_AGENT:
        solver = basic_agent_iterative(
            tools=[bash(), read_file_chunk()] + research_tools,
            ...
        )
    else:
        solver = basic_agent_plus(
            tools=[bash(), python(), ..., web_browser()] + research_tools,
            ...
        )
    
    return Task(dataset=[Sample(input=instructions)], solver=solver, ...)
```

### 指令层次 (instructions/)
- **instructions.txt**: 基础代码复现指令 (所有代理通用)
- **instructions_iterative.txt**: 强调多轮迭代
- **instructions_with_knowledge.txt**: 强制调用知识库工具
- **code_only_instructions.txt**: 仅代码生成 (跳过 paper2code 规划阶段)

---

## 六、关键洞察

### 1. **分层知识表示的威力**
- Paper Node 提供上下文
- Technique Node 明确学术概念
- Code Node 保证可执行性
- 三层映射增强了 RAG 的精准度

### 2. **代理设计的权衡**
- **BasicAgent**: 速度快，适合短任务 (< 1 小时)
- **IterativeAgent**: 质量高，适合长任务 (8+ 小时)，强制重新聚焦

### 3. **PaperCoder 的系统性**
- 规划 → 分析 → 编码 的链式分解
- 每步生成中间产物（规划、配置、逻辑分析）
- 自然融入 xKG 代码检索

### 4. **可验证性原则**
- xKG 只保留有实现的技术（知识过滤步骤 3）
- 代理可通过执行 test scripts 验证代码
- 从信息论角度排除噪声

---

## 七、你的改进与贡献

从代码结构看，你的工作包括：

1. **xKG 构造** (xKG/source/components/):
   - PaperParser: LaTeX → Paper 对象
   - CodeParser: Repository → Code 对象
   - GraphHandler: 生成 Node JSON

2. **知识检索 API** (xKG/source/interface/retrieve.py):
   - initialize_kg() 
   - find_similar_techniques()
   - find_similar_papers()
   - decompose_and_find_techniques()

3. **代理集成** (这里的工作):
   - _knowledge.py: 四个工具函数
   - start.py: 工具装配与代理初始化
   - _basic_agent_plus_with_knowledge.py: 显式集成版本

4. **Benchmark 适配** (paper2code/codes/):
   - 1_planning_*.py: 多个规划变体
   - 2_analyzing_*.py: 知识检索判断
   - 3_coding_*.py: 代码生成

---

## 八、下一步建议

1. **对标论文**:
   - 对比 vanilla vs +xKG 的中间产物（规划、分析、代码）
   - 分析 xKG 工具的调用频率与命中率

2. **扩展方向**:
   - 在 IterativeAgent 中动态更新 xKG（agent.py 的 evolve 接口）
   - 支持多种论文格式（PDF → OCR → LaTeX）
   - 论文与代码的自动对齐（映射哪些论文对应哪些 GitHub）

3. **可复现性**:
   - 发布 xKG 的 42 篇论文语料库
   - 提供易用的 API（已有 initialize_kg 等）
   - 编写使用示例文档

---

## 附录：文件地图

```
/agents/
├── aisi-basic-agent/
│   ├── start.py                          # 主入口，装配工具与代理
│   ├── _basic_agent_plus.py             # BasicAgent 核心
│   ├── _basic_agent_iterative.py        # IterativeAgent 核心
│   ├── _basic_agent_plus_with_knowledge.py  # 显式 xKG 集成版本
│   ├── _knowledge.py                    # 四个知识工具的实现 ⭐
│   ├── _file_reader.py                  # 文件读取工具
│   ├── _execute.py                      # bash/python 执行工具
│   ├── templates.py                     # 提示模板
│   └── config.yaml                      # 代理配置
│
├── paper2code/
│   ├── codes/
│   │   ├── 0_pdf_process.py            # 预处理
│   │   ├── 1_planning*.py              # 多个规划流程
│   │   ├── 2_analyzing*.py             # 包含 xKG 检索的分析
│   │   ├── 3_coding*.py                # 代码生成
│   │   ├── utils.py                    # LLM 调用、成本统计
│   │   └── ResearchKG/                 # xKG 本地副本
│   └── scripts/
│       ├── run_knowledge.sh            # WITH xKG
│       ├── run_no_knowledge.sh         # WITHOUT xKG
│       ├── run_no_code.sh              # 仅规划，无代码检索
│       └── ...
│
├── instructions/
│   ├── instructions.txt                 # 基础
│   ├── instructions_iterative.txt       # Iterative 版本
│   ├── instructions_with_knowledge.txt  # xKG 强制提示
│   └── code_only_instructions.txt       # 代码仅生成
│
└── registry.py                          # 代理注册机制
```

---

## 总结

这就是你 xKG 论文在 PaperBench 中的完整集成！

**核心创新**：通过**可验证的、分层的、链接代码的知识图谱**，把论文的隐式知识显式化，从而显著提升了 LLM 代理的论文复现能力。

**三个代理的角色**：
- **BasicAgent**: 快速原型（一次性生成）
- **IterativeAgent**: 深度优化（多轮重启思考）
- **PaperCoder**: 系统工程（规划-分析-编码三阶段）

**xKG 的集成方式**：
- 在 BasicAgent/IterativeAgent 中作为**可选工具**（代理自主选择调用）
- 在 PaperCoder 的分析阶段中作为**自动检索**（对"实现文件"自动激活）

**性能提升**：+10.9% ~ +24.8%，取决于模型和论文类型
