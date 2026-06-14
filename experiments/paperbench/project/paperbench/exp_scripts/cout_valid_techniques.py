import os
import json
from typing import Dict, Any

# --- 配置 ---
# 请确保这个路径是正确的
TARGET_DIRECTORY = "/disk/disk_20T/luoyujie/preparedness/project/paperbench/paperbench/agents/aisi-basic-agent/ResearchKG/storage/kg"

# --- 全局计数器 ---
valid_nodes_count = 0
invalid_nodes_count = 0
files_processed = 0
files_with_techniques = 0

def process_technique_node(node: Dict[str, Any]):
    """
    递归处理一个技术节点及其所有子节点，并更新全局计数器。
    """
    global valid_nodes_count, invalid_nodes_count

    # 检查当前节点是否有效
    # "code" 键存在且其值不为 None (对应 JSON 中的 null)
    if node.get("code") is not None:
        valid_nodes_count += 1
    else:
        invalid_nodes_count += 1
    
    # 递归处理子节点 (components)
    if "components" in node and isinstance(node["components"], list):
        for child_node in node["components"]:
            if isinstance(child_node, dict): # 确保子节点是对象
                process_technique_node(child_node)


def main():
    """
    主函数，遍历目录并处理所有 JSON 文件。
    """
    global files_processed, files_with_techniques

    print(f"开始扫描目录: {TARGET_DIRECTORY}\n")

    if not os.path.isdir(TARGET_DIRECTORY):
        print(f"错误：目录 '{TARGET_DIRECTORY}' 不存在。请检查路径。")
        return

    # 遍历目录中的所有文件
    for filename in os.listdir(TARGET_DIRECTORY):
        if filename.endswith(".json"):
            file_path = os.path.join(TARGET_DIRECTORY, filename)
            files_processed += 1
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查是否存在 "techniques" 键并且它是一个列表
                if "techniques" in data and isinstance(data["techniques"], list):
                    if data["techniques"]: # 确保列表不为空
                        files_with_techniques += 1
                        # 遍历根技术节点
                        for root_node in data["techniques"]:
                             if isinstance(root_node, dict):
                                process_technique_node(root_node)

            except json.JSONDecodeError:
                print(f"警告: 文件 '{filename}' 不是有效的JSON格式，已跳过。")
            except Exception as e:
                print(f"处理文件 '{filename}' 时发生未知错误: {e}")

    # --- 最终结果输出 ---
    total_nodes = valid_nodes_count + invalid_nodes_count
    
    if total_nodes > 0:
        valid_percentage = (valid_nodes_count / total_nodes) * 100
    else:
        valid_percentage = 0.0

    print("\n--- 统计报告 ---\n")
    print(f"总共扫描文件数: {files_processed}")
    print(f"包含 'techniques' 键的文件数: {files_with_techniques}")
    print("-" * 20)
    print(f"有效 (valid) 节点总数: {valid_nodes_count}")
    print(f"无效 (invalid) 节点总数: {invalid_nodes_count}")
    print(f"技术节点总数: {total_nodes}")
    print("-" * 20)
    print(f"有效节点占比: {valid_percentage:.2f}%")
    print("\n--- 统计完成 ---")


if __name__ == "__main__":
    main()
