"""
极简版知识图谱构建脚本。
- 只处理指定的纯论文任务。
- 按顺序执行，无并行。
- 移除了所有与代码处理和复杂配对相关的逻辑。
"""

import os
import logging

# 确保这里的导入路径相对于您的项目结构是正确的
from .utils import get_config, initialize_app
from .tools import ArxivScraper, PaperParser, GraphHandler

# ==============================================================================
#  1. 定义要处理的论文标题 (仅此而已)
# ==============================================================================
PATH_TO_PROCESS = [
    "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/origin_processed.json",
]

# ==============================================================================
#  2. 主执行函数 (极简、顺序执行)
# ==============================================================================
def main():
    """
    顺序处理每一篇论文。
    """
    # --- 1. 初始化和配置 ---
    print("Initializing application...")
    initialize_app()
    config = get_config()
    
    paper_save_path = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper"
    kg_path = "/disk/disk_20T/luoyujie/ResearchKG/storage/target"
    
    os.makedirs(paper_save_path, exist_ok=True)
    os.makedirs(kg_path, exist_ok=True)

    # --- 2. 实例化所有需要的工具 ---
    # (在循环外实例化，避免重复加载模型)
    arxiv_scraper = ArxivScraper(model=config['model'])
    paper_parser = PaperParser(model=config['model'])
    graph_constructor = GraphHandler(model=config['model'])

    # --- 3. 顺序处理每一篇论文 ---
    for title in PATH_TO_PROCESS:
        print("\n" + "="*80)
        print(f"Processing paper: '{title}'")
        print("="*80)
        
        try:
            # --- 步骤 1: 下载论文 ---
            # print(f"[1/3] Downloading paper PDF...")
            # paper_path = arxiv_scraper.run(title, save_path=paper_save_path)
            # if not paper_path or not os.path.exists(paper_path):
            #     raise FileNotFoundError("ArxivScraper did not return a valid file path.")
            # print(f"      -> Success! Saved to: {paper_path}")

            # --- 步骤 2: 解析论文内容 ---
            print(f"[2/3] Parsing paper content...")
            paper = paper_parser.run(paper_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw/origin_processed.json", save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw", llm_only=True)
            if not paper:
                raise ValueError("PaperParser did not return a valid Paper object.")
            print(f"      -> Success! Parsed title: {paper.title}")

            # --- 步骤 3: 生成并保存知识图谱节点 (无代码) ---
            print(f"[3/3] Generating Knowledge Graph node...")
            # 关键：调用 generate_node 时，将 code 参数设为 None
            node = graph_constructor.generate_node(
                paper=paper, 
                code=None,  # 明确告知没有代码
                save_path=kg_path
            )
            if not node:
                 raise RuntimeError("GraphHandler failed to generate a Node object.")
            print(f"      -> Success! Node generated and saved.")

        except Exception as e:
            error_message = f"!!! FAILED to process paper '{title}'. Error: {e}"
            print(error_message)
            # 记录到日志，方便排查
            logging.error(error_message, exc_info=True)

    print("\n" + "="*80)
    print("All tasks completed.")
    print("="*80)


if __name__ == "__main__":
    main()
