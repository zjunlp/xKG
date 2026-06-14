#!/bin/bash

# 检查参数数量
if [ $# -ne 2 ]; then
    echo "Usage: $0 <submission_path> <paper_id>"
    echo "Example: $0 /path/to/submission all-in-one"
    exit 1
fi

# 接收命令行参数
SUBMISSION_PATH="$1"
PAPER_ID="$2"

# 设置输出目录
SAVE_PATH="${SUBMISSION_PATH}/evaluation"

# 执行 Python 脚本
python /disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/scripts/run_judge.py \
    --submission-path "$SUBMISSION_PATH" \
    --paper-id "$PAPER_ID" \
    --judge simple \
    --model o3-mini-2025-01-31 \
    --out-dir "$SAVE_PATH" \
    --reasoning-effort high \
    --code-only

# python /disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/scripts/run_judge.py \
#     --submission-path "$SUBMISSION_PATH" \
#     --paper-id "$PAPER_ID" \
#     --judge simple \
#     --model DMXAPI-HuoShan-DeepSeek-V3 \
#     --out-dir "$SAVE_PATH" \
#     --reasoning-effort high \
#     --code-only