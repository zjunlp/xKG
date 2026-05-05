import os
import sys

def count_string_in_files(directory, target_string):
    """
    递归遍历目录中的所有文件，统计包含特定字符串的行数总和。
    """
    total_count = 0
    
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在。")
        return -1

    print(f"正在目录 '{directory}' 中搜索...")
    print(f"统计包含 '{target_string}' 的总行数...")
    print("----------------------------------------")
    
    # os.walk 会递归地遍历目录树
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            try:
                # 以只读模式打开文件，忽略编码错误（以防遇到二进制文件）
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if target_string in line:
                            total_count += 1
            except (IOError, OSError):
                # 忽略因权限不足等原因无法读取的文件
                continue
                
    return total_count

if __name__ == "__main__":
    # 要搜索的目标字符串
    TARGET_STRING = '"code": {'
    
    # 如果命令行提供了目录，则使用它；否则，使用当前目录 (.)
    search_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    
    # 执行统计
    count = count_string_in_files(search_dir, TARGET_STRING)
    
    # 输出结果
    if count != -1:
        print(f"总计找到: {count} 行")

