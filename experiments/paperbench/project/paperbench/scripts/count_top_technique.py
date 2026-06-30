import os
import json
import sys

# --- 配置区 ---
# 您要遍历的 JSON 文件所在的目录
TARGET_DIRECTORY = "/disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/agents/aisi-basic-agent/ResearchKG/storage/kg"
# --- 配置区结束 ---

def count_files_with_single_technique(directory: str) -> tuple[int, int, list[str]]:
    """
    统计指定目录中，顶层 'techniques' 列表长度恰好为 1 的 JSON 文件数量。

    Args:
        directory: 目标目录的路径。

    Returns:
        一个元组，包含:
        - 满足条件的文件的数量。
        - 扫描到的 JSON 文件总数。
        - 满足条件的文件的名称列表。
    """
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在或不是一个目录。", file=sys.stderr)
        sys.exit(1)

    single_tech_count = 0
    total_json_files = 0
    matching_files = []

    print(f"正在扫描目录: {directory}\n")

    # 遍历目录中的所有文件
    file_list = sorted([f for f in os.listdir(directory) if f.endswith(".json")])
    if not file_list:
        print("目录中未找到任何 .json 文件。")
        return 0, 0, []

    for filename in file_list:
        total_json_files += 1
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 核心检查逻辑：'techniques' 存在，是列表，且长度为 1
            if 'techniques' in data and isinstance(data.get('techniques'), list) and len(data['techniques']) == 1:
                single_tech_count += 1
                matching_files.append(filename)

        except json.JSONDecodeError:
            print(f"警告：文件 '{filename}' 不是有效的 JSON，已跳过。", file=sys.stderr)
        except Exception as e:
            print(f"处理文件 '{filename}' 时发生未知错误: {e}", file=sys.stderr)

    return single_tech_count, total_json_files, matching_files


if __name__ == "__main__":
    # 执行统计
    count, total_scanned, matched_list = count_files_with_single_technique(TARGET_DIRECTORY)
    
    # 打印最终结果
    print("\n" + "="*30)
    print("      统 计 结 果")
    print("="*30)
    print(f"\n总共扫描的 JSON 文件数量: {total_scanned}")
    print("\n>>> 顶层 'techniques' 节点有且只有一个的论文文件数目为: " f"{count}")
    print("="*30)
    
    # (可选) 打印匹配的文件列表以供核对
    # print("\n--- 匹配的文件列表 ---")
    # if matched_list:
    #     for name in matched_list:
    #         print(name)
    # else:
    #     print("未找到任何匹配的文件。")

