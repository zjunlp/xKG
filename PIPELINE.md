# xKG 构建 Pipeline 详细文档

本文档描述从原始论文+代码仓库到最终知识图谱节点（Node JSON）的完整执行流程，粒度精确到每个函数调用、参数、输入输出格式。

---

## 整体流程

```
paper_dir/          code_dir/
    └─ *.tex            └─ *.py, *.md ...
    └─ {id}.json (meta)
         │                   │
         ▼                   ▼
   PaperParser.run()   CodeParser.run()
         │                   │
         ▼                   ▼
      Paper object       Code object
              │               │
              └───────┬───────┘
                      ▼
          GraphHandler.generate_node()
                      │
                      ▼
          {title}_graph_unverified.json   ← 最终输出
```

---

## Phase 1: 解析论文 — `PaperParser.run()`

**模块路径**: `xkg/src/components/paper_parser.py`

### 调用方式
```python
from xkg.src.components import PaperParser
paper_parser = PaperParser(model=config['model'])
paper = paper_parser.run(
    paper_path="/path/to/storage/raw/paper/2506.01234v1",  # 论文目录
    paper_format="latex",   # 目前只支持 "latex"
    save_path="/path/to/storage/raw/paper"  # 可选，保存中间JSON
)
```

### 前置要求：目录结构
```
storage/raw/paper/2506.01234v1/
    ├── 2506.01234v1.json     ← 必须存在！元信息文件（arxiv元数据）
    ├── main.tex              ← 主tex文件（含\documentclass）
    ├── sections/
    │   ├── intro.tex
    │   └── method.tex
    └── references.bib        ← 或 references.bbl
```

**元信息 JSON 格式**（`{dir_name}.json`）：
```json
{
    "title": "Paper Title",
    "abstract": "Abstract text...",
    "url": "https://arxiv.org/abs/2506.01234",
    "year": 2025,
    "authors": ["Author A", "Author B"]
}
```

### 内部执行步骤

#### Step 1.1 — `LatexExtractor.run()` 结构化提取

1. 读取 `{paper_path}/{dir_name}.json` → `Paper` 对象（title, abstract, url, year, authors）
2. `_find_main_tex(paper_dir)`: 遍历目录找含 `\documentclass` 的 `.tex` 文件
3. `_load_tex_content(main_tex)`: 递归追踪 `\input{...}` / `\include{...}` 合并所有 `.tex` 内容
4. `_construct_latex_node(content)`: 用 `pylatexenc` 解析 `\begin{document}...\end{document}` 区间，返回节点树
5. `_extract_sections(nodes)`: 提取 `\section` / `\subsection`，构建两级 `Section` 树
   - `Section(id="1", name="Introduction", content="...", subsection=[Section(id="1.1", ...)])`
6. `_extract_labeled_environments(nodes, "fig")` / `("tab")`: 提取带 `\label` 的图表 LaTeX 源码
7. `_extract_equations(nodes)`: 提取 `equation` / `align` 环境
8. `_inject_references(sections, figures, tables)`: 在 section content 中遇到 `\ref{label}` 时，将对应图表 LaTeX 注入其后
9. `_find_references(paper_dir)`:
   - 优先找 `.bib` 文件 → 用 `bibtexparser` 提取标题列表
   - 找不到 `.bib` 则找 `.bbl` → LLM 提取（`_prompt_get_references`）
10. 设置 `paper.sections`, `paper.equations`, `paper.references`

保存（若 `save_path` 不为 None）：`{save_path}/{dir_name}.json`

#### Step 1.2 — `LLMExtractor.run()` 语义提取

**输入**: 上一步的 `Paper` 对象（已有 sections）

1. **提取 GitHub URL**：
   - 用正则 `https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+` 扫描所有 sections 文本
   - LLM 调用（`_prompt_get_github_url`）从候选链接中选出官方仓库链接 → `paper.code_url`
   - 使用模型：`config['model']`

2. **提取 introduction**：
   - 找 section.name 含 "introduction" 的节 → `paper.introduction`
   - 找不到则用 `paper.abstract`

3. **提取 contributions**（关键步骤）：
   - LLM 调用（`_prompt_get_contributions`）
   - 使用模型：`config['paper_model']`（专用论文模型）
   - 系统提示：`CONTRIBUTION_EXTRACTOR_PROMPT`
   - 输入：title + sections JSON + equations JSON
   - 输出解析为 `List[Contribution]`，过滤掉 `type` 不在 `["Methodology", "Technique", "Finding", "Resource"]` 的项

   **`Contribution` 数据结构**：
   ```python
   @dataclass
   class Contribution:
       name: str           # 简洁学术术语
       type: str           # "Methodology" | "Technique" | "Finding" | "Resource"
       description: str    # 详细实现说明
       components: List[Contribution]  # 子 Technique（可递归嵌套）
   ```

保存（若 `save_path` 不为 None）：`{save_path}/{sanitize(paper.title)}_llm.json`

### Phase 1 输出：`Paper` 对象
```python
@dataclass
class Paper:
    title: str
    abstract: str
    url: str
    year: int
    authors: List[str]
    sections: List[Section]           # 两级section树
    references: List[str]             # 参考文献标题列表
    equations: List[str]              # ["eq:1: \begin{equation}...", ...]
    introduction: str                 # 引言文本
    code_url: str                     # GitHub 仓库链接
    contributions: List[Contribution] # 核心贡献（含层级结构）
```

---

## Phase 2: 解析代码库 — `CodeParser.run()`

**模块路径**: `xkg/src/components/code_parser.py`

### 调用方式
```python
from xkg.src.components import CodeParser
code_parser = CodeParser(model=config['model'])
code = code_parser.run(
    code_path="/path/to/storage/raw/code/author_reponame",
    save_path="/path/to/storage/raw/code"  # 可选
)
```

### 内部执行步骤

1. **`get_file_tree(code_path)`**:
   - 递归遍历目录，跳过 `.git`, `__pycache__`, `node_modules`, `.venv` 等
   - 收集所有相对路径 → `file_tree_str`（换行分隔）
   - 读取第一个 `README.md` → `readme_content`

2. **LLM 调用：生成 overview**（`_prompt_get_code_overview`）:
   - 输入：repo name, file_tree, readme
   - 输出：Markdown 格式的仓库概述（Overview / Architecture / Core Features）
   - 使用模型：`config['model']`
   - 系统提示：`REPOSITORY_ANALYZER_PROMPT`
   - → `code.overview`

3. **LLM 调用：关联论文标题**（`_prompt_get_relevant_paper`）:
   - 输入：repo name, readme
   - 输出：对应论文标题字符串 或 `None`
   - → `code.paper`

4. **收集源文件**（根据 config `code_rag.code_extensions` 和 `doc_extensions`）:
   - 代码文件（如 `.py`, `.js`, `.java`）→ `File(name, content, is_implementation=True)`
     - `is_implementation=False` 若路径含 "test"
   - 文档文件（如 `.md`, `.rst`）→ `File(name, content, is_implementation=False)`

5. 保存（若 `save_path`）：`{save_path}/{repo_name}_meta.json`

### Phase 2 输出：`Code` 对象
```python
@dataclass
class Code:
    name: str               # 仓库目录名，如 "author_reponame"
    readme: str
    file_tree: str          # 所有文件相对路径，换行分隔
    file_list: List[File]   # File(name, content, is_implementation)
    overview: str           # LLM 生成的结构化仓库描述
    paper: str              # 关联论文标题（可能为 None）
```

> **paper-only 模式**：直接传 `code=None` 给 `generate_node()`，跳过所有代码处理。

---

## Phase 3: 生成知识图谱节点 — `GraphHandler.generate_node()`

**模块路径**: `xkg/src/components/graph_handler.py`

### 调用方式
```python
from xkg.src.components import GraphHandler
graph_handler = GraphHandler(model=config['model'])
node = graph_handler.generate_node(
    paper=paper,                        # Phase 1 输出
    code=code,                          # Phase 2 输出，paper-only 传 None
    save_path=config['kg_path'],        # 保存目录
    llm_rerank=True,                    # 是否启用 LLM 对代码文件重排
    rewrite_paper=None,                 # None 时从 config paper_rag.rag 读取
    verify_code=None                    # None 时从 config code_rag.exec_check_code 读取
)
```

### 内部执行步骤

#### Step 3.1 — 过滤 contributions → Technique 树

```python
top_level_techniques = [
    _convert_to_technique(c)
    for c in paper.contributions
    if c.type in ["Methodology", "Technique"]
]
# "Finding" 和 "Resource" 不进入 technique 树，直接进 node.findings/resources
```

`Technique` 结构：
```python
@dataclass
class Technique:
    name: str
    description: str
    code: Optional[CodeBlock]  # 初始为 None，后续填充
    weight: float              # 初始为 0.0，后续分配
    components: List[Technique]  # 子技术（与 Contribution 树结构相同）
```

#### Step 3.2 — 用论文原文重写 Technique 描述（可选）

**触发条件**：`config['paper_rag']['rag'] == True`

对 technique 树的**所有节点**（递归展开，包含所有层级）：

```python
# 1. 准备 PaperRAG（基于 paper.sections 建 FAISS 索引，缓存到 .pkl 文件）
paper_rag.prepare_retriever(paper, db_path="{raw_data_path}/paper/{sanitize(title)}.pkl")

# 2. 对每个 technique 节点：
query = f"{tech.name}: {tech.description}"
excerpts = paper_rag(query)      # 返回相关 section 文本列表

# 3. LLM 调用（_prompt_rewrite_paper）
# 使用模型：config['paper_model']
# 无系统提示
# 输出：单段连续文本（学术风格），包裹在 ``` 中
tech.description = extracted_text  # 原地修改
```

**注意**：此步骤使用 `paper_model`，不是 `model`。

#### Step 3.3 — 给技术节点分配权重

```python
# 叶节点（无 components）权重 = 1 / 叶节点总数
# 父节点权重 = 所有子节点权重之和
# 按权重升序排序（叶节点先处理）
```

#### Step 3.4 — CodeRAG：为每个 Technique 检索相关代码文件

**触发条件**：`code` 不为 None

```python
# 准备 CodeRAG（基于 code.file_list 建检索索引，缓存到 .pkl 文件）
code_rag.prepare_retriever(code, db_path="{raw_data_path}/code/{code.name}.pkl")

# 对每个 technique 节点：
query = f"{tech.name}: {tech.description}"
initial_snippets = code_rag(query)   # 返回 List[FileSnippet]
```

**FileSnippet 结构**：
```python
@dataclass
class FileSnippet:
    file_name: str          # 相对路径，如 "src/model.py"
    code_list: List[str]    # 该文件的代码片段列表
```

若 `llm_rerank=True`，进一步 LLM 过滤（`_prompt_check_snippets`）：
- 使用模型：`config['code_model']`
- 系统提示：`REPOSITORY_ANALYZER_PROMPT`
- 输入：technique 定义、code.overview、所有 snippet 文件名+内容
- 输出：相关文件名列表（按相关性降序），或 `None`
- 配置：`top_files=5`（最终保留文件数），`top_complete_file=1`（前1个文件加载完整内容，受 `max_prompt_code_bytes` 限制）

```python
tech.code = json.dumps([asdict(fs) for fs in relevant_files])  # 暂存为 JSON 字符串
```

#### Step 3.5 — 代码重写：将散乱片段合并为可执行代码块

**触发条件**：`llm_rerank=True`（按权重升序，叶节点先处理）

对每个 technique 节点，调用 `_get_rewrited_code(paper, tech)`：

**情况 A：叶节点**（无子节点有 CodeBlock）
```
LLM 调用：_prompt_rewrite_code
使用模型：config['code_model']
系统提示：CODE_REWRITER_PROMPT
输入：
  - paper.abstract
  - technique name + description
  - tech.code（Step 3.4 得到的 JSON 字符串，即 FileSnippet 列表）
输出：两个 ``` 代码块
  - 第一个：Python 代码（含 # TEST BLOCK 注释分隔符）
  - 第二个：纯文本文档说明
```

**情况 B：父节点**（部分子节点已有 CodeBlock）
```
LLM 调用：_prompt_rewrite_restructure_code
输入追加：sub_techniques（子节点的 name + implementation + documentation）
逻辑：复用子节点代码，补全未覆盖部分
```

代码输出拆分：
```python
# 以 "# TEST BLOCK" 为分隔符，将代码分为实现部分和测试部分
implementation = code_above_test_block
test = code_below_test_block

tech.code = CodeBlock(
    implementation=implementation,
    test=test,
    documentation=second_backtick_block
)
```

若 LLM 返回 `None` 或只返回一个代码块，则 `tech.code = None`。

可选 LLM 保真检查（`config['code_rag']['llm_check_code'] == True`）：
- 调用 `_prompt_check_rewrite_code` 验证代码是否忠实于原始 snippets 和 technique 定义
- 检查失败则丢弃，`tech.code = None`

**最终 CodeBlock 结构**：
```python
@dataclass
class CodeBlock:
    implementation: str   # 核心实现代码（不含 TEST BLOCK 部分）
    test: str             # TEST BLOCK 之后的测试用例代码
    documentation: str    # 5-10 句的使用说明文本
    package: List[str]    # 依赖包列表（可能为空）
```

#### Step 3.6 — 组装 Node 对象

```python
node = Node(
    paper_title=paper.title,
    paper_abstract=paper.abstract,
    paper_references=paper.references,   # 参考文献标题列表
    code_file=code.file_tree if code else None,
    code_overview=code.overview if code else None,
    techniques=top_level_techniques,     # 带 CodeBlock 的完整 Technique 树
    resources=[f"{c.name}: {c.description}"
               for c in paper.contributions if c.type == "Resource"],
    findings=[f"{c.name}: {c.description}"
              for c in paper.contributions if c.type == "Finding"],
)
```

#### Step 3.7 — 保存到磁盘

```python
save_file = f"{save_path}/{sanitize(paper.title)}_graph_unverified.json"
json.dump(asdict(node), f, indent=2, ensure_ascii=False)
```

若 `verify_code=True`，还会额外保存：`_graph_verified.json`（经 Docker 执行验证后的版本）。

---

## Phase 4: 最终输出 — Node JSON 结构

```json
{
  "paper_title": "...",
  "paper_abstract": "...",
  "paper_references": ["ref title 1", "ref title 2"],
  "code_file": "src/model.py\nsrc/train.py\n...",
  "code_overview": "## Overview\n...",
  "techniques": [
    {
      "name": "Technique A",
      "description": "Detailed description...",
      "weight": 0.5,
      "code": {
        "implementation": "import torch\n...\ndef technique_a(...):\n    ...",
        "test": "if __name__ == '__main__':\n    ...",
        "documentation": "This module implements...",
        "package": ["torch", "numpy"]
      },
      "components": [
        {
          "name": "Sub-technique A1",
          "description": "...",
          "weight": 0.25,
          "code": { "implementation": "...", "test": "...", "documentation": "...", "package": [] },
          "components": []
        }
      ]
    }
  ],
  "resources": ["Dataset X: A benchmark for ..."],
  "findings": ["Finding Y: We empirically show that ..."]
}
```

---

## 关键配置项（`config.yaml`）

| 配置路径 | 说明 | 影响步骤 |
|---|---|---|
| `model` | 主模型 | CodeParser overview/paper, LLM rerank, 任务分解 |
| `paper_model` | 论文专用模型 | Contribution 提取, 描述重写 |
| `code_model` | 代码专用模型 | LLM rerank, 代码重写 |
| `paper_rag.rag` | 是否启用描述重写 | Step 3.2 |
| `code_rag.llm_rerank` | 是否启用 LLM 文件重排 | Step 3.4 |
| `code_rag.top_files` | 最终保留代码文件数 | Step 3.4 |
| `code_rag.top_complete_file` | 加载完整内容的文件数 | Step 3.4 |
| `code_rag.exec_check_code` | 是否 Docker 执行验证 | Step 3.7 |
| `code_rag.llm_check_code` | 是否 LLM 保真检查 | Step 3.5 |
| `max_prompt_code_bytes` | 单文件内容最大字节数（默认 52100） | Step 3.4 |
| `raw_data_path` | RAG 索引缓存目录 | Step 3.2, 3.4 |
| `kg_path` | Node JSON 保存目录 | Step 3.7 |

---

## 完整调用示例

```python
from xkg.src.utils import initialize_app, get_config
from xkg.src.components import PaperParser, CodeParser, GraphHandler

initialize_app()
config = get_config()

paper_parser = PaperParser(model=config['model'])
code_parser = CodeParser(model=config['model'])
graph_handler = GraphHandler(model=config['model'])

# Phase 1
paper = paper_parser.run(
    paper_path="/path/to/storage/raw/paper/2506.01234v1",
    paper_format="latex",
    save_path="/path/to/storage/raw/paper"
)

# Phase 2（paper-only 则跳过，传 code=None）
code = code_parser.run(
    code_path="/path/to/storage/raw/code/author_reponame",
    save_path="/path/to/storage/raw/code"
)

# Phase 3
node = graph_handler.generate_node(
    paper=paper,
    code=code,          # 或 None
    save_path="/path/to/storage/kg",
    llm_rerank=True
)
# 输出：/path/to/storage/kg/{sanitize(title)}_graph_unverified.json
```
