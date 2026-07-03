# xKG 论文实验启动脚本总览

本目录包含用于复现 xKG 论文实验的完整启动脚本。

## 目录结构

```
scripts/
├── aisi-basic/           ← BasicAgent 和 IterativeAgent 启动脚本
│   ├── run_o3_with_xkg.sh              # o3-mini + WITH xKG
│   ├── run_o3_without_xkg.sh           # o3-mini + WITHOUT xKG (Baseline)
│   ├── run_deepseek_r1_with_xkg.sh     # DeepSeek-R1 + WITH xKG
│   ├── run_deepseek_r1_without_xkg.sh  # DeepSeek-R1 + WITHOUT xKG
│   └── README.md
│
├── paper2code/           ← Paper2Code (PaperCoder) 启动脚本
│   ├── run_with_xkg.sh      # WITH xKG
│   ├── run_without_xkg.sh   # WITHOUT xKG (Baseline)
│   ├── run_comparison.sh    # 完整消融实验
│   └── README.md
│
└── README.md             ← 本文件
```

## 快速开始

### 1. 环境准备

```bash
# 安装 xKG 包
cd /Users/luoyujie/Documents/Code/xKG
pip install -e .

# 设置 API 密钥
export OPENAI_API_KEY="<your-openai-key>"
export GITHUB_TOKEN="<your-github-token>"  # 可选，用于访问私有仓库
```

### 2. 运行实验

#### 方案 A：BasicAgent + o3-mini（推荐快速测试）

```bash
cd scripts/aisi-basic

# 运行 WITH xKG 版本
bash run_o3_with_xkg.sh

# 对比：运行 WITHOUT xKG 基线
bash run_o3_without_xkg.sh
```

**预期性能**：
- WITH xKG：20.7%
- WITHOUT xKG：9.8%
- 提升：**+10.9%**

#### 方案 B：IterativeAgent + DeepSeek-R1（更强的自改进）

```bash
cd scripts/aisi-basic

# 运行 WITH xKG 版本
bash run_deepseek_r1_with_xkg.sh

# 对比：运行 WITHOUT xKG 基线
bash run_deepseek_r1_without_xkg.sh
```

**预期性能**：
- WITH xKG：16.3%
- WITHOUT xKG：8.1%
- 提升：**+8.2%**

#### 方案 C：Paper2Code（最强的代码生成能力）

```bash
cd scripts/paper2code

# 完整对比实验（同时运行 with/without xKG）
bash run_comparison.sh \
    "Paper Title" \
    "o3-mini" \
    "./paper.json" \
    "./results" \
    "./guide.json"

# 或分别运行
bash run_with_xkg.sh "Paper Title" "o3-mini" "./paper.json" "./output" "./guide.json"
bash run_without_xkg.sh "Paper Title" "o3-mini" "./paper.json" "./output_baseline"
```

**预期性能**：
- WITH xKG：53.21%
- WITHOUT xKG：42.31%
- 提升：**+10.9%**

## 实验矩阵

| 脚本 | Agent 类型 | LLM 模型 | xKG | 预期性能 | 文件 |
|------|-----------|---------|-----|--------|------|
| 1 | BasicAgent | o3-mini | ✓ | **20.7%** ↑10.9% | `aisi-basic/run_o3_with_xkg.sh` |
| 2 | BasicAgent | o3-mini | ✗ | 9.8% | `aisi-basic/run_o3_without_xkg.sh` |
| 3 | IterativeAgent | DeepSeek-R1 | ✓ | **16.3%** ↑8.2% | `aisi-basic/run_deepseek_r1_with_xkg.sh` |
| 4 | IterativeAgent | DeepSeek-R1 | ✗ | 8.1% | `aisi-basic/run_deepseek_r1_without_xkg.sh` |
| 5 | PaperCoder | o3-mini | ✓ | **53.21%** ↑10.9% | `paper2code/run_with_xkg.sh` |
| 6 | PaperCoder | o3-mini | ✗ | 42.31% | `paper2code/run_without_xkg.sh` |

## 脚本详细说明

### aisi-basic/ 目录

针对 BasicAgent 和 IterativeAgent 的脚本。这些脚本使用 Docker 容器运行，配合 PaperBench 评测框架。

详见：[aisi-basic/README.md](./aisi-basic/README.md)

**关键特性**：
- 通过环境变量 `XKG_ENABLED` 控制是否启用 xKG
- 通过 `ITERATIVE_AGENT` 环境变量选择 Agent 类型
- 支持不同 LLM 模型（o3-mini, DeepSeek-R1 等）

### paper2code/ 目录

针对 Paper2Code（PaperCoder）的脚本。这些脚本在本地运行，采用三阶段流水线。

详见：[paper2code/README.md](./paper2code/README.md)

**关键特性**：
- 三阶段流水线：Planning → Analyzing → Coding
- Analyzing 阶段是 xKG 的集成点（检索相关技术实现）
- 支持本地运行和快速迭代

## 核心改进点

### BasicAgent / IterativeAgent 中 xKG 的作用

```python
# 在 start.py 中（支持环境变量控制）
research_tools = []
if os.environ.get("XKG_ENABLED", "true").lower() == "true":
    research_tools = [
        get_overview(),          # 工具1：获取论文概览
        get_similar_techniques(),  # 工具2：查询相似技术实现
        get_full_techniques()      # 工具3：获取完整技术树
    ]

tools = [...base_tools...] + research_tools
```

**增强效果**：
- Agent 可调用 xKG 工具了解论文全貌
- 在实现具体技术时获取参考代码
- 降低 Agent 的代码生成难度

### Paper2Code 中 xKG 的作用

```python
# 在 Stage 2 (Analyzing) 中
for filename in task_list:
    analysis = llm_analyze(...)  # LLM 分析
    
    if is_core_implementation_file:
        # ← xKG 增强点
        similar_techs = decompose_and_find_techniques(
            f"{filename}: {analysis}",
            top_k=3,
            code_only=False
        )
        
        # 将代码参考注入到分析文本中
        augmented_analysis = f"{analysis}\n\nRelevant implementations: {json.dumps(...)}"
    
    save(f"{filename}_simple_analysis.txt", augmented_analysis)
```

**增强效果**：
- Stage 2 分析时包含具体代码参考
- Stage 3 生成代码时，LLM 可看到现成的实现示例
- 代码生成的保真度大幅提升

## 环境变量速查表

| 变量名 | 说明 | 示例值 |
|-------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | `sk-...` |
| `GITHUB_TOKEN` | GitHub 访问令牌 | `ghp_...` |
| `XKG_ENABLED` | 是否启用 xKG（aisi-basic） | `true` / `false` |
| `ITERATIVE_AGENT` | 是否使用 IterativeAgent（aisi-basic） | `true` / `false` |
| `XKG_KG_PATH` | xKG 知识库路径（全局） | `/path/to/kg` |
| `XKG_CONFIG_PATH` | xKG 配置文件路径（全局） | `/path/to/config.yaml` |
| `HF_ENDPOINT` | Hugging Face 镜像源 | `https://hf-mirror.com` |
| `MODEL` | aisi-basic 使用的 LLM 模型 | `o3-mini` |

## 完整消融实验工作流

要完整复现论文的消融实验，建议按以下顺序执行：

### 步骤 1：BasicAgent + o3-mini

```bash
cd scripts/aisi-basic

# 1.1 WITH xKG
echo "=== Test 1: o3-mini WITH xKG ==="
bash run_o3_with_xkg.sh
# 预期性能：20.7%

# 1.2 WITHOUT xKG
echo "=== Test 2: o3-mini WITHOUT xKG (Baseline) ==="
bash run_o3_without_xkg.sh
# 预期性能：9.8%

# 性能提升：+10.9%
```

### 步骤 2：IterativeAgent + DeepSeek-R1

```bash
# 2.1 WITH xKG
echo "=== Test 3: DeepSeek-R1 WITH xKG ==="
bash run_deepseek_r1_with_xkg.sh
# 预期性能：16.3%

# 2.2 WITHOUT xKG
echo "=== Test 4: DeepSeek-R1 WITHOUT xKG (Baseline) ==="
bash run_deepseek_r1_without_xkg.sh
# 预期性能：8.1%

# 性能提升：+8.2%
```

### 步骤 3：Paper2Code

```bash
cd ../paper2code

echo "=== Test 5: Paper2Code WITH vs WITHOUT xKG ==="
bash run_comparison.sh \
    "Test Paper" \
    "o3-mini" \
    "./test.json" \
    "./results" \
    "./guide.json"

# 性能提升：+10.9% (42.31% → 53.21%)
```

## 性能汇总

完成所有实验后，您将看到以下性能对标：

```
╔═══════════════════════════════════════════════════════════════════╗
║                    xKG 性能对标汇总                               ║
╠════════════════════╦═════════════╦═════════════╦═════════════════╣
║    配置            ║  无 xKG     ║   有 xKG    ║    性能提升      ║
╠════════════════════╬═════════════╬═════════════╬═════════════════╣
║ BasicAgent + o3    ║    9.8%     ║   20.7%     ║   ↑10.9%        ║
║ IterativeAgent+DS  ║    8.1%     ║   16.3%     ║   ↑8.2%         ║
║ PaperCoder + o3    ║   42.31%    ║   53.21%    ║   ↑10.9%        ║
╚════════════════════╩═════════════╩═════════════╩═════════════════╝
```

## 故障排除

### 常见错误 1：xKG 导入失败

```
ModuleNotFoundError: No module named 'xKG'
```

**解决**：
```bash
cd /Users/luoyujie/Documents/Code/xKG
pip install -e .
```

### 常见错误 2：Docker 镜像未找到

```
Error: image not found: aisi-basic-agent:latest
```

**解决**：确保已构建基础镜像
```bash
docker build -t aisi-basic-agent:latest \
  -f Dockerfile \
  paperbench/agents/aisi-basic-agent/
```

### 常见错误 3：KG 数据路径错误

```
FileNotFoundError: /home/guide.json
```

**解决**：设置正确的 KG 路径
```bash
export XKG_KG_PATH="/Users/luoyujie/Documents/Code/xKG/storage/kg"
```

## 参考资源

- **论文全文**：`/Users/luoyujie/Documents/Code/xKG/xkg.md`
- **集成架构**：`/Users/luoyujie/Documents/Code/xKG/xKG_INTEGRATION_ARCHITECTURE.md`
- **实现计划**：`/Users/luoyujie/Documents/Code/xKG/.claude/plan.md`
- **主 xKG 包**：`/Users/luoyujie/Documents/Code/xKG/`

## 扩展阅读

### 三类 Agent 对比

| Agent 类型 | 特点 | 自改进能力 | xKG 增强效果 |
|-----------|------|----------|------------|
| BasicAgent | 简单 ReAct 循环 | 无 | ↑10.9% |
| IterativeAgent | 带反思的自改进 | 强 | ↑8.2% |
| PaperCoder | 三阶段流水线 | 中等 | ↑10.9% |

### xKG 三层价值

1. **知识检索** - 提供论文精准概括，替代逐字阅读
2. **代码参考** - 提供可运行的实现示例，加速代码生成
3. **质量保证** - CodeVerifier 确保参考代码的正确性

## 许可证与致谢

本实验脚本基于 xKG 论文提供。

**相关出版物**：
> Luo, Y., Yu, Z., Wang, X., et al. (2026). "What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations." *arXiv preprint arXiv:2510.17795*

---

**最后更新**：2026-06-30  
**维护者**：Yujie Luo <luo.yj@zju.edu.cn>
