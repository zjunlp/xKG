#!/bin/bash

# 脚本功能: 
# 1. 从命令行读取一个数字，指定每篇论文的运行次数。
# 2. 读取同目录下的 'split.txt' 文件，获取论文ID列表。
# 3. 对列表中的每篇论文，并行执行指定次数的复现流程（此版本使用 guide.json）。
# 4. 使用一个任务池来控制并发数量，防止系统过载。
# 5. 将每个任务的输出重定向到独立的日志文件，方便调试。

# --- 安全设置 ---
# -e: 如果任何命令失败，脚本立即退出
# -u: 如果使用未定义的变量，脚本报错并退出
# -o pipefail: 如果管道中的任何命令失败，整个管道的返回值为失败
set -euo pipefail

# --- 配置区域 ---

# 设置最大并行任务数。请根据你的CPU核心数、内存和API速率限制进行调整。
# 一个好的起点是你的CPU核心数的一半或全部。
MAX_PARALLEL_JOBS=5

# API和模型配置
export OPENAI_API_KEY="sk-CfrUJ26ECE2kMLsPl141hC0PB3RBxCtfX5kVY8gO2WZDAGV3"
export OPENAI_BASE_URL="https://vip.dmxapi.com/v1/"
GPT_VERSION="o3-mini-2025-01-31"
EVAL_MODEL="o3-mini-2025-01-31"

# --- 输入检查 ---
if [ "$#" -ne 1 ]; then
    echo "错误: 需要提供运行次数作为参数。"
    echo "用法: $0 <每篇论文的运行次数>"
    exit 1
fi

if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -lt 1 ]; then
    echo "错误: 运行次数必须是一个正整数。"
    exit 1
fi

NUM_RUNS_PER_PAPER=$1
SPLIT_FILE="split.txt"
LOG_DIR="parallel_logs" # 日志文件存放目录

if [ ! -f "$SPLIT_FILE" ]; then
    echo "错误: 'split.txt' 文件在当前目录未找到。"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"
echo "所有任务日志将保存在 '${LOG_DIR}' 目录下。"

# --- 核心处理函数 ---
# 将处理单篇论文的所有逻辑封装到此函数中
process_paper() {
    local PAPER_ID=$1
    local run_num=$2
    
    # --- 为本次运行设置变量 ---
    local RUN_ID=$(date +%Y%m%d_%H%M%S) # 使用 年月日_时分秒 作为唯一ID
    local REPO_INDEX="${PAPER_ID}-${RUN_ID}"
    # local REPO_INDEX="mechanistic-understanding-20250929_005617"

    echo "========================================================================"
    echo ">> 开始运行: 第 ${run_num}/${NUM_RUNS_PER_PAPER} 次"
    echo ">> 论文ID  : ${PAPER_ID}"
    echo ">> 运行ID  : ${RUN_ID}"
    echo ">> 仓库索引: ${REPO_INDEX}"
    echo "========================================================================"

    # --- 定义路径 ---
    local BASE_DATA_DIR="/disk/disk_20T/luoyujie/preparedness/project/paperbench/data/papers"
    local BASE_RUN_DIR="/disk/disk_20T/luoyujie/preparedness/project/paperbench/runs/DeepCode"
    
    local PDF_JSON_PATH="${BASE_DATA_DIR}/${PAPER_ID}/paper.json"
    local PDF_JSON_CLEANED_PATH="${BASE_DATA_DIR}/${PAPER_ID}/paper_cleaned.json"
    local GUIDE_PATH="${BASE_DATA_DIR}/${PAPER_ID}/guide.json" # Guide 路径
    local OUTPUT_DIR="${BASE_RUN_DIR}/${REPO_INDEX}"
    local OUTPUT_REPO_DIR="${OUTPUT_DIR}_repo"

    # --- 创建输出目录 ---
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_REPO_DIR"

    # --- 激活 Conda 环境并执行 ---
    !! 注意: conda activate 需要在函数内部，因为每个子shell需要独立激活环境
    echo "------- 切换到 'researchkg' 环境 -------"
    source activate /disk/disk_20T/luoyujie/anaconda3/envs/researchkg

    # echo "------- 1. 预处理 PDF JSON -------"
    python ../codes/0_pdf_process.py \
        --input_json_path "${PDF_JSON_PATH}" \
        --output_json_path "${PDF_JSON_CLEANED_PATH}"

    echo "------- 2. PaperCoder - 规划 (Planning with Guide) -------"
    python ../codes/1_planning_no_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}"
        # --guide_json_path "${GUIDE_PATH}" \


    echo "------- 3. PaperCoder - 提取配置 (Extract Config) -------"
    python ../codes/1.1_extract_config.py \
        --paper_name "$REPO_INDEX" \
        --output_dir "${OUTPUT_DIR}"

    # 复制配置文件
    cp -rp "${OUTPUT_DIR}/planning_config.yaml" "${OUTPUT_REPO_DIR}/config.yaml"

    echo "------- 4. PaperCoder - 分析 (Analyzing with Guide) -------"
    python ../codes/2_analyzing_no_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}"
        # --guide_json_path "${GUIDE_PATH}" \


    echo "------- 5. PaperCoder - 编码 (Coding) -------"
    python ../codes/3_coding_no_knowledge.py \
        --paper_name "$REPO_INDEX" \
        --gpt_version "${GPT_VERSION}" \
        --pdf_json_path "${PDF_JSON_CLEANED_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --output_repo_dir "${OUTPUT_REPO_DIR}" \
        # --guide_json_path "${GUIDE_PATH}" \

    # --- 切换环境进行评估 ---
    echo "------- 切换到 'paperbench' 环境进行评估 -------"
    source activate /disk/disk_20T/luoyujie/anaconda3/envs/paperbench
    
    local SUBMISSION_PATH="$OUTPUT_REPO_DIR"
    local SAVE_PATH="${SUBMISSION_PATH}/evaluation"
    
    
    echo "------- 6. 运行评估 -------"
    python /disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/scripts/run_judge.py \
         --submission-path "$SUBMISSION_PATH" \
         --paper-id "$PAPER_ID" \
         --judge simple \
         --model "$EVAL_MODEL" \
         --out-dir "$SAVE_PATH" \
         --reasoning-effort high \
         --code-only
         
    echo ">> 完成运行: ${REPO_INDEX}"
    echo "------------------------------------------------------------------------"
}

# --- 主循环和任务分发 ---
# 导出函数，以便子shell可以通过 `bash -c` 访问
export -f process_paper
# 导出需要的环境变量，确保子shell也能获取
export OPENAI_API_KEY
export OPENAI_BASE_URL
export GPT_VERSION
export EVAL_MODEL

# 外层循环: 控制每篇论文运行的次数
for (( run_num=1; run_num<=NUM_RUNS_PER_PAPER; run_num++ )); do
    # 内层循环: 读取 split.txt, 遍历每篇论文
    while IFS= read -r PAPER_ID || [ -n "$PAPER_ID" ]; do
        # 跳过空行
        if [ -z "$PAPER_ID" ]; then
            continue
        fi

        # 当活动的后台任务数量达到上限时，等待任何一个任务完成
        # `jobs -p` 列出所有后台任务的PID
        while (( $(jobs -p | wc -l) >= MAX_PARALLEL_JOBS )); do
            sleep 1
        done

        # 启动后台任务，并将标准输出和错误输出重定向到日志文件
        echo "正在为 ${PAPER_ID} (第 ${run_num} 次) 派发新任务..."
        # 使用 bash -c 是为了确保在新的子shell中执行，并且可以方便地传递参数和重定向
        bash -c "process_paper '$PAPER_ID' '$run_num'" > "${LOG_DIR}/${PAPER_ID}_run${run_num}_no_knowledge.log" 2>&1 &

    done < "$SPLIT_FILE"
done

# 等待所有剩余的后台任务完成
echo "所有任务已派发，正在等待剩余任务全部完成..."
wait
echo "!!!!!!!!!! 所有任务已全部完成 !!!!!!!!!! "
