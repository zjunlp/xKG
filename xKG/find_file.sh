#!/bin/bash

# --- 请在这里修改你要搜索的目录 ---
SEARCH_DIR="/disk/disk_20T/luoyujie/ResearchKG/storage/raw"

# 定义所有要搜索的模式串
PATTERNS=(
  "Adapt_in_the_Wild_Test-Time_Entropy_Minimization_with_Sharpness_and_Feature_Regularization"
  "Direct_Preference_Optimization_Your_Language_Model_is_Secretly_a_Reward_Model"
  "Interpretability_in_the_Wild_a_Circuit_for_Indirect_Object_Identification_in_GPT-2_small"
  "Plug_and_Play_Language_Models_A_Simple_Approach_to_Controlled_Text_Generation"
  "Progress_measures_for_grokking_via_mechanistic_interpretability"
  "RealToxicityPrompts_Evaluating_Neural_Toxic_Degeneration_in_Language_Models"
)

# 动态构建 find 命令的参数
FIND_ARGS=()
for pattern in "${PATTERNS[@]}"; do
  # 如果不是第一个参数，就先加上 -o
  if [ ${#FIND_ARGS[@]} -ne 0 ]; then
    FIND_ARGS+=( -o )
  fi
  FIND_ARGS+=( -name "*$pattern*" )
done

# 执行最终的 find 命令
echo "Searching in directory: $SEARCH_DIR"
find "$SEARCH_DIR" \( "${FIND_ARGS[@]}" \)
