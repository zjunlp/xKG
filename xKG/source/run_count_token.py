import os
import tiktoken

# 支持的文件类型（根据你的项目调整）
SUPPORTED_EXTS = {'.py', '.json', '.txt', '.md', '.js', '.java', '.cpp', '.h', '.sh', '.yaml', '.yml'}

def count_tokens_in_directory(root_dir, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    total_tokens = 0
    file_count = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in SUPPORTED_EXTS):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    tokens = len(enc.encode(content))
                    total_tokens += tokens
                    file_count += 1
                    print(f"{filepath}: {tokens} tokens")
                except Exception as e:
                    print(f"⚠️ 无法读取 {filepath}: {e}")

    return total_tokens, file_count

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("用法: python count_tokens.py <目录路径>")
        sys.exit(1)

    directory = sys.argv[1]
    total, count = count_tokens_in_directory(directory)
    print(f"\n✅ 共统计 {count} 个文件")
    print(f"📊 总 token 数: {total:,}")
