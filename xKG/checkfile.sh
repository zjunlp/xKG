#!/bin/bash

# --- 配置区域 ---

# 1. 存放JSON文件的目录
SEARCH_DIR="/disk/disk_20T/luoyujie/ResearchKG/storage/kg_additional_another"

# 2. 论文标题列表
TITLES_TO_CHECK=(
    "Understanding and improving visual prompting: A label-mapping perspective"
    "Exploring Visual Prompts for Adapting Large-Scale Models"
    "Adversarial Reprogramming of Neural Networks"
    "Transfer learning without knowing: Reprogramming black-box ML models"
    "Deep Residual Learning for Image Recognition"
    "Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning"
    "Feature Pyramid Networks for Object Detection"
    "BlackVIP: Black-box Visual Prompting for Robust Transfer Learning"
    "An Automated Visual Prompting Framework and Benchmark"
)

# --- 脚本正文 (已修正) ---

# 检查目标目录是否存在
if [ ! -d "$SEARCH_DIR" ]; then
    echo "错误：目录 '$SEARCH_DIR' 不存在！"
    exit 1
fi

echo "正在检查目录: $SEARCH_DIR"
echo "============================================="
found_missing=false

# 遍历列表中的每一个标题
for title in "${TITLES_TO_CHECK[@]}"; do

    # --- 修正后的核心处理步骤 ---
    # 1. 只去除冒号 :
    # 2. 将所有空格替换为下划线 _
    prefix=$(echo "$title" | sed -e 's/://g' -e 's/ /_/g')

    # --- 使用 -iname 进行不区分大小写的查找 ---
    # 查找以处理后前缀开头的文件
    # -print -quit: 找到一个就立刻退出，效率更高
    found_file=$(find "$SEARCH_DIR" -maxdepth 1 -type f -iname "${prefix}*" -print -quit)

    # 如果 $found_file 变量是空的，说明没有找到任何匹配的文件
    if [[ -z "$found_file" ]]; then
        echo "未找到对应的文件: ${title}"
        found_missing=true
    fi
done

if [ "$found_missing" = false ]; then
    echo "所有标题均找到了对应的文件。"
fi
echo "============================================="
echo "检查完成。"
