# import os
# import json
# from collections import Counter
# import sys

# # --- 配置区 ---
# # 您要遍历的 JSON 文件所在的目录
# TARGET_DIRECTORY = "/disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/agents/aisi-basic-agent/ResearchKG/storage/kg"
# # --- 配置区结束 ---

# def count_valid_nodes_recursive(node: dict) -> int:
#     """
#     根据新的累加规则，递归地统计一个节点及其所有子孙节点中
#     包含有效代码的节点总数。

#     Args:
#         node: 一个技术或组件对象（字典）。

#     Returns:
#         该节点树中有效节点的总数。
#     """
#     if not isinstance(node, dict):
#         return 0

#     # 1. 检查当前节点自身是否有效
#     # 如果当前节点的 code 不为 null，则自身算 1 个有效节点
#     current_node_count = 1 if node.get("code") is not None else 0

#     # 2. 递归累加所有子节点的有效数量
#     components = node.get("components", [])
#     if isinstance(components, list):
#         for component in components:
#             current_node_count += count_valid_nodes_recursive(component)
    
#     return current_node_count


# def process_directory(directory: str) -> list[int]:
#     """
#     处理指定目录下的所有 JSON 文件，并返回每个文件有效技术的数量列表。
#     """
#     if not os.path.isdir(directory):
#         print(f"错误：目录 '{directory}' 不存在或不是一个目录。", file=sys.stderr)
#         return []

#     valid_counts_per_file = []
    
#     print(f"正在扫描目录: {directory}\n")
    
#     # 遍历目录中的所有文件
#     file_list = sorted([f for f in os.listdir(directory) if f.endswith(".json")])
#     if not file_list:
#         print("目录中未找到任何 .json 文件。")
#         return []
        
#     for filename in file_list:
#         file_path = os.path.join(directory, filename)
        
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
            
#             if "techniques" not in data or not isinstance(data["techniques"], list):
#                 continue
            
#             total_valid_nodes_in_file = 0
#             # 对文件中每一个顶层 technique 对象进行递归计数
#             for top_level_tech in data["techniques"]:
#                 total_valid_nodes_in_file += count_valid_nodes_recursive(top_level_tech)
            
#             valid_counts_per_file.append(total_valid_nodes_in_file)

#         except json.JSONDecodeError:
#             print(f"警告：文件 '{filename}' 不是有效的 JSON，已跳过。", file=sys.stderr)
#         except Exception as e:
#             print(f"处理文件 '{filename}' 时发生未知错误: {e}", file=sys.stderr)

#     return valid_counts_per_file


# if __name__ == "__main__":
#     all_counts = process_directory(TARGET_DIRECTORY)
    
#     if not all_counts:
#         print("\n未找到或未成功处理任何 JSON 文件。")
#     else:
#         count_summary = Counter(all_counts)
        
#         print("\n--- 统计结果 ---")
#         if not count_summary:
#             print("所有已处理文件中均未发现有效技术。")
#         else:
#             print("每个文件中 'code' 不为 null 的节点总数分布情况：\n")
#             for count, num_papers in sorted(count_summary.items()):
#                 print(f"包含 {count} 个代码块的论文有: {num_papers} 篇")
        
#         print("\n--- 所有文件有效代码块数量的完整列表 ---")
#         print(all_counts)
        
#         print(f"\n总计处理了 {len(all_counts)} 个 JSON 文件。")

import os
import json
from collections import Counter
import sys

# --- 配置区 ---
# 您要遍历的 JSON 文件所在的目录
TARGET_DIRECTORY = "/disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/agents/aisi-basic-agent/ResearchKG/storage/kg"
# --- 配置区结束 ---

def count_code_blocks_recursive(node: dict) -> int:
    """
    递归地统计一个节点及其所有子孙节点中，
    'code' 字段不为 null 的节点总数。

    Args:
        node: 一个技术或组件对象（字典）。

    Returns:
        该节点树中 'code' 不为 null 的节点总数。
    """
    if not isinstance(node, dict):
        return 0

    # 1. 检查当前节点自身的 'code' 字段
    count = 1 if node.get("code") is not None else 0

    # 2. 递归地将所有子节点的计数累加起来
    components = node.get("components", [])
    if isinstance(components, list):
        for component in components:
            count += count_code_blocks_recursive(component)
    
    return count


def process_directory(directory: str) -> list[int]:
    """
    处理指定目录下的所有 JSON 文件，并返回每个文件 'code' 不为 null 的节点总数列表。
    """
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在或不是一个目录。", file=sys.stderr)
        return []

    code_block_counts_per_file = []
    
    print(f"正在扫描目录: {directory}\n")
    
    file_list = sorted([f for f in os.listdir(directory) if f.endswith(".json")])
    if not file_list:
        print("目录中未找到任何 .json 文件。")
        return []
        
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "techniques" not in data or not isinstance(data["techniques"], list):
                continue
            
            total_code_blocks_in_file = 0
            # 对文件中每一个顶层 technique 对象进行递归计数
            for top_level_tech in data["techniques"]:
                total_code_blocks_in_file += count_code_blocks_recursive(top_level_tech)
            
            code_block_counts_per_file.append(total_code_blocks_in_file)

        except json.JSONDecodeError:
            print(f"警告：文件 '{filename}' 不是有效的 JSON，已跳过。", file=sys.stderr)
        except Exception as e:
            print(f"处理文件 '{filename}' 时发生未知错误: {e}", file=sys.stderr)

    return code_block_counts_per_file


if __name__ == "__main__":
    all_counts = process_directory(TARGET_DIRECTORY)
    
    if not all_counts:
        print("\n未找到或未成功处理任何 JSON 文件。")
    else:
        count_summary = Counter(all_counts)
        
        print("\n--- 统计结果 ---")
        if not any(count > 0 for count in all_counts):
            print("所有已处理文件中均未发现 'code' 不为 null 的节点。")
        else:
            print("每个文件中 'code' 不为 null 的节点总数分布情况：\n")
            # 按数量排序输出
            for count, num_papers in sorted(count_summary.items()):
                print(f"包含 {count} 个代码块的论文有: {num_papers} 篇")
        
        print("\n--- 所有文件有效代码块数量的完整列表 ---")
        print(all_counts)
        
        print(f"\n总计处理了 {len(all_counts)} 个 JSON 文件。")



