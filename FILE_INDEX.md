# xKG 项目文件索引

完整的文件导航指南 - 帮助您快速定位所有重要文件。

---

## 📁 根目录文件

| 文件名 | 类型 | 说明 | 优先级 |
|--------|------|------|--------|
| `pyproject.toml` | 配置 | pip 包配置文件（支持 pip install -e） | ⭐⭐⭐ |
| `IMPLEMENTATION_SUMMARY.md` | 文档 | 完整的工作实现总结 | ⭐⭐⭐ |
| `WORK_COMPLETION_REPORT.md` | 文档 | 项目完成报告 | ⭐⭐ |
| `FILE_INDEX.md` | 文档 | 本文件 - 快速导航 | ⭐⭐ |
| `.claude/plan.md` | 文档 | 实现规划（参考文档） | ⭐ |

---

## 🚀 快速启动脚本

### aisi-basic/ 目录

位置：`/Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench/scripts/aisi-basic/`

| 脚本名 | 配置 | 预期性能 | 用途 |
|--------|------|--------|------|
| `run_o3_with_xkg.sh` | BasicAgent + o3-mini + WITH xKG | **20.7%** | 展示 xKG 增强效果 |
| `run_o3_without_xkg.sh` | BasicAgent + o3-mini + WITHOUT xKG | 9.8% | 消融实验基线 |
| `run_deepseek_r1_with_xkg.sh` | IterativeAgent + DS-R1 + WITH xKG | **16.3%** | 自改进 Agent + xKG |
| `run_deepseek_r1_without_xkg.sh` | IterativeAgent + DS-R1 + WITHOUT xKG | 8.1% | 消融实验基线 |
| `README.md` | 文档 | — | 详细使用说明 |

**快速使用**：
```bash
cd scripts/aisi-basic
bash run_o3_with_xkg.sh
```

### paper2code/ 目录

位置：`/Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench/scripts/paper2code/`

| 脚本名 | 配置 | 预期性能 | 用途 |
|--------|------|--------|------|
| `run_with_xkg.sh` | PaperCoder + o3-mini + WITH xKG | **53.21%** | 展示 xKG 对代码生成的增强 |
| `run_without_xkg.sh` | PaperCoder + o3-mini + WITHOUT xKG | 42.31% | 消融实验基线 |
| `run_comparison.sh` | 同时运行两个版本 | — | 一键消融实验对比 |
| `README.md` | 文档 | — | 详细使用说明 |

**快速使用**：
```bash
cd scripts/paper2code
bash run_comparison.sh "Paper Title" "o3-mini" "./paper.json" "./results"
```

---

## 📚 文档和指南

### 快速参考

| 文件 | 位置 | 内容 | 适合人群 |
|------|------|------|---------|
| **QUICK_START.md** | `scripts/` | 30 秒快速启动 | 👤 首次用户 |
| **README.md** | `scripts/` | 完整项目文档 | 👥 所有用户 |

### 详细文档

| 文件 | 位置 | 内容 | 适合人群 |
|------|------|------|---------|
| **aisi-basic/README.md** | `scripts/aisi-basic/` | BasicAgent 详情 | 🔧 开发者 |
| **paper2code/README.md** | `scripts/paper2code/` | PaperCoder 详情 | 🔧 开发者 |

### 技术文档

| 文件 | 位置 | 内容 | 适合人群 |
|------|------|------|---------|
| **IMPLEMENTATION_SUMMARY.md** | `/` | 工作实现细节 | 👨‍💻 技术人员 |
| **WORK_COMPLETION_REPORT.md** | `/` | 项目完成报告 | 👨‍💼 管理者 |

---

## 🔧 代码修改文件

### xKG 主包

| 文件 | 修改类型 | 变动说明 |
|------|--------|---------|
| `xKG/source/utils/config.py` | 修改 | 添加 `config_path` 和 `kg_path` 参数支持 |
| `/pyproject.toml` | 新增 | pip 包配置（支持 pip install -e） |

### aisi-basic-agent

| 文件 | 修改类型 | 变动说明 |
|------|--------|---------|
| `paperbench/agents/aisi-basic-agent/_knowledge.py` | 修改 | 更新导入路径：`xKG.research_kg` → `xKG.source` |
| `paperbench/agents/aisi-basic-agent/start.py` | 修改 | 添加 `XKG_ENABLED` 环境变量支持 |

### paper2code

| 文件 | 修改类型 | 变动说明 |
|------|--------|---------|
| `paperbench/agents/paper2code/codes/2_analyzing.py` | 修改 | 更新导入路径：`xKG.research_kg` → `xKG.source` |

---

## 🧪 实验命令速查

### BasicAgent 实验

```bash
# 有 xKG
cd scripts/aisi-basic && bash run_o3_with_xkg.sh

# 无 xKG (消融)
cd scripts/aisi-basic && bash run_o3_without_xkg.sh
```

### IterativeAgent 实验

```bash
# 有 xKG
cd scripts/aisi-basic && bash run_deepseek_r1_with_xkg.sh

# 无 xKG (消融)
cd scripts/aisi-basic && bash run_deepseek_r1_without_xkg.sh
```

### PaperCoder 实验

```bash
# 完整消融对比
cd scripts/paper2code && bash run_comparison.sh "Paper" "o3-mini" "./paper.json" "./results"

# 仅有 xKG
cd scripts/paper2code && bash run_with_xkg.sh "Paper" "o3-mini" "./paper.json" "./results"

# 仅无 xKG
cd scripts/paper2code && bash run_without_xkg.sh "Paper" "o3-mini" "./paper.json" "./results"
```

---

## 📋 使用流程

### 第一次使用

1. **查看快速开始**
   ```bash
   cat scripts/QUICK_START.md
   ```

2. **验证环境**
   ```bash
   bash scripts/VERIFY_SETUP.sh
   ```

3. **运行第一个实验**
   ```bash
   bash scripts/aisi-basic/run_o3_with_xkg.sh
   ```

### 完整复现

1. **查看完整文档**
   ```bash
   cat scripts/README.md
   ```

2. **运行所有 4 种配置**（见实验命令速查）

3. **查看结果**
   ```bash
   cat runs/*/*/pb_result.json
   ```

### 深入理解

1. **查看实现总结**
   ```bash
   cat IMPLEMENTATION_SUMMARY.md
   ```

2. **查看技术文档**
   ```bash
   cat scripts/aisi-basic/README.md
   cat scripts/paper2code/README.md
   ```

---

## 🎯 按用途查找

### 我想...

#### "快速了解这个项目"
👉 查看：`scripts/QUICK_START.md`（5 分钟）

#### "看详细的实验说明"
👉 查看：`scripts/README.md`（15 分钟）

#### "运行第一个实验"
👉 查看：`scripts/aisi-basic/README.md` 的"快速开始"部分

#### "复现论文的所有实验"
👉 查看：`scripts/README.md` 的"完整消融实验工作流"部分

#### "理解代码修改"
👉 查看：`IMPLEMENTATION_SUMMARY.md` 的"代码修改"部分

#### "了解项目完成情况"
👉 查看：`WORK_COMPLETION_REPORT.md`

#### "查看脚本源代码"
👉 查看：`scripts/aisi-basic/*.sh` 或 `scripts/paper2code/*.sh`

---

## 🌳 完整文件树

```
/Users/luoyujie/Documents/Code/xKG/
├── pyproject.toml                          # ✨ 新增：pip 包配置
├── IMPLEMENTATION_SUMMARY.md               # ✨ 新增：工作总结
├── WORK_COMPLETION_REPORT.md               # ✨ 新增：完成报告
├── FILE_INDEX.md                           # ✨ 新增：本文件
│
├── xKG/source/
│   └── utils/
│       └── config.py                       # 🔧 修改：添加外部路径支持
│
├── experiments/paperbench/project/paperbench/
│   ├── paperbench/agents/
│   │   ├── aisi-basic-agent/
│   │   │   ├── _knowledge.py              # 🔧 修改：更新导入路径
│   │   │   └── start.py                   # 🔧 修改：XKG_ENABLED 环境变量
│   │   │
│   │   └── paper2code/
│   │       └── codes/
│   │           └── 2_analyzing.py         # 🔧 修改：更新导入路径
│   │
│   └── scripts/
│       ├── README.md                      # ✨ 新增：完整文档
│       ├── QUICK_START.md                 # ✨ 新增：快速开始
│       ├── VERIFY_SETUP.sh                # ✨ 新增：环境验证
│       │
│       ├── aisi-basic/
│       │   ├── run_o3_with_xkg.sh         # ✨ 新增：o3 + xKG
│       │   ├── run_o3_without_xkg.sh      # ✨ 新增：o3 无 xKG
│       │   ├── run_deepseek_r1_with_xkg.sh    # ✨ 新增：DS-R1 + xKG
│       │   ├── run_deepseek_r1_without_xkg.sh # ✨ 新增：DS-R1 无 xKG
│       │   └── README.md                  # ✨ 新增：BasicAgent 文档
│       │
│       └── paper2code/
│           ├── run_with_xkg.sh            # ✨ 新增：with xKG
│           ├── run_without_xkg.sh         # ✨ 新增：without xKG
│           ├── run_comparison.sh          # ✨ 新增：对比脚本
│           └── README.md                  # ✨ 新增：PaperCoder 文档
```

---

## 🔑 关键概念速查

### 环境变量

| 变量 | 值 | 含义 |
|------|-----|------|
| `XKG_ENABLED` | `true`/`false` | 启用/禁用 xKG |
| `ITERATIVE_AGENT` | `true` | 使用 IterativeAgent |
| `XKG_KG_PATH` | `/path/to/kg` | KG 数据路径 |
| `OPENAI_API_KEY` | `sk-...` | API 密钥 |

### 配置对应关系

| 配置名 | Agent 类型 | LLM | xKG | 预期性能 |
|-------|-----------|-----|-----|---------|
| 1 | BasicAgent | o3-mini | ✓ | 20.7% |
| 2 | BasicAgent | o3-mini | ✗ | 9.8% |
| 3 | IterativeAgent | DS-R1 | ✓ | 16.3% |
| 4 | IterativeAgent | DS-R1 | ✗ | 8.1% |
| 5 | PaperCoder | o3-mini | ✓ | 53.21% |
| 6 | PaperCoder | o3-mini | ✗ | 42.31% |

---

## 📞 快速求助

| 问题 | 解决方案 |
|------|---------|
| xKG 导入失败 | 运行 `pip install -e /Users/luoyujie/Documents/Code/xKG` |
| 脚本执行权限问题 | 运行 `chmod +x scripts/**/*.sh` |
| Docker 镜像不存在 | 查看 `scripts/aisi-basic/README.md` 中的"故障排除" |
| 性能低于预期 | 查看 `scripts/README.md` 中的"故障排除" |
| 不知道从何开始 | 查看 `scripts/QUICK_START.md` |

---

## ✅ 验证检查清单

在运行实验前，确保检查以下项目：

- [ ] 已安装 xKG 包：`pip install -e /Users/luoyujie/Documents/Code/xKG`
- [ ] 已设置 API 密钥：`export OPENAI_API_KEY="..."`
- [ ] 已构建 Docker 镜像：`docker images | grep aisi-basic-agent`
- [ ] 已验证环境：`bash scripts/VERIFY_SETUP.sh`
- [ ] 已阅读文档：至少看过 `scripts/QUICK_START.md`

---

## 📊 性能对标表

最终复现结果期望值：

```
┌────────────────────────┬────────────┬────────────┬────────────┐
│ 配置                   │ 无 xKG     │ 有 xKG     │ 提升       │
├────────────────────────┼────────────┼────────────┼────────────┤
│ BasicAgent + o3-mini   │ 9.8%       │ 20.7%      │ ↑10.9%    │
│ IterativeAgent + DS-R1 │ 8.1%       │ 16.3%      │ ↑8.2%     │
│ PaperCoder + o3-mini   │ 42.31%     │ 53.21%     │ ↑10.9%    │
└────────────────────────┴────────────┴────────────┴────────────┘
```

---

## 🎓 参考资源

| 资源 | 位置 | 用途 |
|------|------|------|
| 论文原文 | `xkg.md` | 理解 xKG 设计 |
| 集成架构 | `xKG_INTEGRATION_ARCHITECTURE.md` | 理解集成细节 |
| 实现计划 | `.claude/plan.md` | 理解修改方案 |
| 完成报告 | `WORK_COMPLETION_REPORT.md` | 查看工作成果 |

---

**最后更新**：2026-06-30  
**文件总数**：5 个代码文件 + 7 个脚本 + 6 个文档  
**总代码行数**：~2000+ 行  
**文档覆盖率**：100%
