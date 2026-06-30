#!/bin/bash

# ===================================
# 配置区
# ===================================
TAR_FILE="$1"
MODEL_NAME="DMXAPI-HuoShan-DeepSeek-V3"
CLEAN_UP=false  # 是否清理解压内容（设为 true 可自动清理）

# ===================================
# 自动推导逻辑（关键：顺序不能错）
# ===================================

# 1. 检查 TAR_FILE 是否存在
if [ -z "$TAR_FILE" ]; then
    echo "❌ 错误：请传入 .tar.gz 文件路径"
    echo "用法: $0 <path/to/submission.tar.gz>"
    exit 1
fi

if [ ! -f "$TAR_FILE" ]; then
    echo "❌ 错误：文件不存在: $TAR_FILE"
    exit 1
fi

# 2. 提取 PARENT_DIR_NAME（即 mechanistic-understanding_xxx 这个目录名）
PARENT_DIR_NAME=$(basename "$(dirname "$TAR_FILE")")
echo "📂 压缩包父目录名: $PARENT_DIR_NAME"

# 3. 用正则提取 TASK_NAME 和 RUN_ID
if [[ $PARENT_DIR_NAME =~ ^([^_]+)_([a-f0-9\-]+)$ ]]; then
    TASK_NAME="${BASH_REMATCH[1]}"   # e.g., mechanistic-understanding
    RUN_ID="${BASH_REMATCH[2]}"
else
    echo "⚠️ 无法匹配 task_name 和 run_id，使用默认值"
    TASK_NAME="$PARENT_DIR_NAME"
    RUN_ID="unknown"
fi

echo "✅ Task Name: $TASK_NAME"
echo "✅ Run ID: $RUN_ID"

# 4. 提取时间戳目录名（去掉 .tar.gz）
TIMESTAMP_DIR=$(basename "$TAR_FILE" .tar.gz)
echo "⏰ 解压目录名: $TIMESTAMP_DIR"

# 5. 推导路径
CONTAINING_DIR=$(dirname "$(dirname "$TAR_FILE")")  # runs/group_dir/
EXTRACTED_DIR="$CONTAINING_DIR/$TIMESTAMP_DIR"

# ===================================
# 解压
# ===================================
echo "📁 正在解压到: $EXTRACTED_DIR"
mkdir -p "$EXTRACTED_DIR"
tar -xzf "$TAR_FILE" -C "$EXTRACTED_DIR"

if [ $? -ne 0 ]; then
    echo "❌ 解压失败，请检查压缩包是否损坏"
    exit 1
fi

# ===================================
# 自动查找 submission 目录（修复关键点！）
# ===================================
echo "🔍 正在搜索 'submission' 目录（支持嵌套结构）..."
SUBMISSION_DIR=$(find "$EXTRACTED_DIR" -type d -name "submission" | head -n1)

if [ -z "$SUBMISSION_DIR" ] || [ ! -d "$SUBMISSION_DIR" ]; then
    echo "❌ 错误：未找到 'submission' 目录"
    echo "📦 当前解压内容结构："
    find "$EXTRACTED_DIR" -type d | sed 's|[^/]*/|  |g'
    exit 1
fi

echo "✅ 找到 submission 目录: $SUBMISSION_DIR"
SAVE_PATH="$SUBMISSION_DIR/evaluation"

# ===================================
# 执行评估命令
# ===================================
echo "🚀 开始评估任务: $TASK_NAME"
echo "📝 输出目录: $SAVE_PATH"
mkdir -p "$SAVE_PATH"

python /disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/scripts/run_judge.py \
    --submission-path "$SUBMISSION_DIR" \
    --paper-id "$TASK_NAME" \
    --judge simple \
    --model "$MODEL_NAME" \
    --out-dir "$SAVE_PATH" \
    --reasoning-effort high \
    --code-only

if [ $? -eq 0 ]; then
    echo "✅ 评估成功！结果保存在: $SAVE_PATH"
else
    echo "❌ 评估失败！请检查模型连接或输入格式"
    exit 1
fi

# ===================================
# 拷贝 grader_output.json 到 run group 根目录
# ===================================
SOURCE_JSON="$SAVE_PATH/grader_output.json"

if [ ! -f "$SOURCE_JSON" ]; then
    echo "❌ 错误：评估未生成结果文件: $SOURCE_JSON"
    exit 1
fi

TARGET_JSON="$CONTAINING_DIR/${TASK_NAME}_grader_output.json"
echo "📎 拷贝评估结果到: $TARGET_JSON"
cp "$SOURCE_JSON" "$TARGET_JSON"

if [ $? -eq 0 ]; then
    echo "✅ 成功保存: $TARGET_JSON"
else
    echo "❌ 拷贝失败！"
    exit 1
fi

# ===================================
# 清理（可选）
# ===================================
if [ "$CLEAN_UP" = true ]; then
    echo "🧹 正在清理解压目录: $EXTRACTED_DIR"
    rm -rf "$EXTRACTED_DIR"
    echo "✅ 清理完成"
fi

echo "🎉 全部完成！评估结束。"
