# 🎯 从这里开始 - xKG 项目启动指南

> **新手？从这个文件开始！** 👇

---

## ⚡ 5 分钟快速上手

### 第 1 步：了解项目（1 分钟）

这是一个完成的 xKG 论文复现项目。已经为您准备好：

✅ **5 个代码修改** - 统一 API，消除代码重复  
✅ **7 个启动脚本** - 一键运行所有实验  
✅ **6 份详细文档** - 覆盖所有使用场景  

### 第 2 步：准备环境（2 分钟）

```bash
# 1. 安装 xKG 包
cd /Users/luoyujie/Documents/Code/xKG
pip install -e .

# 2. 设置 API 密钥
export OPENAI_API_KEY="your-key-here"
```

### 第 3 步：验证环境（1 分钟）

```bash
cd experiments/paperbench/project/paperbench/scripts
bash VERIFY_SETUP.sh
```

### 第 4 步：运行第一个实验（1 分钟）

```bash
cd aisi-basic
bash run_o3_with_xkg.sh
```

**完成！** 🎉

---

## 📖 文档导航（按阅读顺序）

| 优先级 | 文档 | 阅读时间 | 内容 |
|-------|------|--------|------|
| 🔴 **必读** | [`QUICK_START.md`](./experiments/paperbench/project/paperbench/scripts/QUICK_START.md) | 5 分钟 | 30 秒快速启动指南 |
| 🟠 **推荐** | [`scripts/README.md`](./experiments/paperbench/project/paperbench/scripts/README.md) | 15 分钟 | 完整项目文档 |
| 🟡 **参考** | [`FILE_INDEX.md`](./FILE_INDEX.md) | 10 分钟 | 文件快速索引 |
| 🟢 **详细** | 各脚本的 `README.md` | 20 分钟 | 特定脚本详情 |

---

## 🚀 4 种快速选项

### 选项 1️⃣：运行单个实验

```bash
cd scripts/aisi-basic
bash run_o3_with_xkg.sh
# 预期结果：20.7% 性能得分
```

### 选项 2️⃣：消融实验对比

```bash
cd scripts/aisi-basic
bash run_o3_with_xkg.sh      # 有 xKG：20.7%
bash run_o3_without_xkg.sh   # 无 xKG：9.8%
# 性能提升：+10.9%
```

### 选项 3️⃣：完整实验集

```bash
cd scripts/aisi-basic
bash run_o3_with_xkg.sh
bash run_o3_without_xkg.sh
bash run_deepseek_r1_with_xkg.sh
bash run_deepseek_r1_without_xkg.sh
# 覆盖 BasicAgent 和 IterativeAgent 两种 Agent
```

### 选项 4️⃣：论文代码生成

```bash
cd scripts/paper2code
bash run_comparison.sh "Paper Title" "o3-mini" "./paper.json" "./results"
# 运行 PaperCoder 的完整消融对比
```

---

## 📊 预期性能结果

完成所有实验后，您应该看到：

| 配置 | 无 xKG | 有 xKG | 提升 |
|------|--------|--------|------|
| BasicAgent + o3 | 9.8% | 20.7% | **↑10.9%** |
| IterativeAgent + DS-R1 | 8.1% | 16.3% | **↑8.2%** |
| PaperCoder + o3 | 42.31% | 53.21% | **↑10.9%** |

---

## ❓ 常见问题

### Q: 如何验证环境配置？
```bash
bash scripts/VERIFY_SETUP.sh
```

### Q: xKG 导入失败怎么办？
```bash
pip install -e /Users/luoyujie/Documents/Code/xKG
```

### Q: 想看完整文档？
```bash
cat scripts/README.md
```

### Q: 查看文件清单？
```bash
cat FILE_INDEX.md
```

---

## 🗺️ 项目地图

```
您在这里 ➜ START_HERE.md
           ↓
           ├─ 快速开始 ➜ scripts/QUICK_START.md
           ├─ 完整指南 ➜ scripts/README.md
           ├─ 文件索引 ➜ FILE_INDEX.md
           │
           ├─ aisi-basic 脚本
           │   ├─ run_o3_with_xkg.sh
           │   ├─ run_o3_without_xkg.sh
           │   ├─ run_deepseek_r1_with_xkg.sh
           │   ├─ run_deepseek_r1_without_xkg.sh
           │   └─ README.md
           │
           └─ paper2code 脚本
               ├─ run_with_xkg.sh
               ├─ run_without_xkg.sh
               ├─ run_comparison.sh
               └─ README.md
```

---

## ✨ 这个项目提供了什么

### 代码改进 ✅
- 消除了 Agent 中的 ResearchKG 副本
- 统一了所有 import 路径
- 添加了灵活的配置系统

### 实验脚本 ✅
- 7 个完整的启动脚本
- 支持 with/without xKG 消融实验
- 自动化处理所有复杂步骤

### 文档说明 ✅
- 快速开始指南
- 完整项目文档
- 脚本使用说明
- 故障排除指南

---

## 🎯 现在就开始！

### 推荐路径

```
1. 阅读本文件（正在做）✓
   ↓
2. 查看 scripts/QUICK_START.md（5 分钟）
   ↓
3. 运行验证脚本（1 分钟）
   bash scripts/VERIFY_SETUP.sh
   ↓
4. 运行第一个实验（等待中）
   bash scripts/aisi-basic/run_o3_with_xkg.sh
   ↓
5. 查看结果
   cat runs/*/*/pb_result.json
```

### 或者直接跳到实验

```bash
# 如果您已有环境，直接运行
cd /Users/luoyujie/Documents/Code/xKG/experiments/paperbench/project/paperbench/scripts
bash aisi-basic/run_o3_with_xkg.sh
```

---

## 📚 完整文档清单

| 文档 | 位置 | 用途 |
|------|------|------|
| **本文件** | `START_HERE.md` | 快速入门 |
| **快速开始** | `scripts/QUICK_START.md` | 30 秒启动 |
| **项目文档** | `scripts/README.md` | 完整说明 |
| **文件索引** | `FILE_INDEX.md` | 快速导航 |
| **实现总结** | `IMPLEMENTATION_SUMMARY.md` | 技术细节 |
| **完成报告** | `WORK_COMPLETION_REPORT.md` | 工作总结 |

---

## 🎓 进阶用法

### 添加新模型

修改脚本中的 `MODEL` 环境变量或参数，然后重新运行。

### 自定义消融实验

参考现有脚本结构创建新脚本，使用不同的环境变量。

### 性能优化

调整 `config.yaml` 中的参数，重新生成 KG 数据。

---

## 🏁 总结

您现在拥有：

✅ 完全修复的 xKG 代码  
✅ 7 个即插即用的实验脚本  
✅ 详细的文档和指南  
✅ 一键验证的环境检查  

**下一步**：打开 `scripts/QUICK_START.md` 或运行第一个脚本！

---

**祝您实验顺利！** 🚀

需要帮助？查看 [`FILE_INDEX.md`](./FILE_INDEX.md) 获取完整的文件导航。

---

**完成日期**：2026-06-30  
**项目状态**：✅ 完全就绪
