# build_kg.py (最终完整版)

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import os
from typing import List, Tuple, Dict, Optional
import json

# 导入你的模块
# 请确保这里的导入路径是正确的
from .utils import get_config, initialize_app
from .tools import ArxivScraper, GithubScraper, PaperParser, CodeParser, GraphHandler

# ==============================================================================
#  1. 定义要处理的源任务
# ==============================================================================
PAPER_CODE_TO_EXTRACT: List[Tuple[str, str]] = [
    ("Understanding and improving visual prompting: A label-mapping perspective", "https://github.com/OPTML-Group/ILM-VP"),
    ("Exploring Visual Prompts for Adapting Large-Scale Models", "https://github.com/hjbahng/visual_prompting"),
    ("Adversarial Reprogramming of Neural Networks", "https://github.com/hjbahng/visual_prompting"),
    ("Transfer learning without knowing: Reprogramming black-box ML models", "https://github.com/IBM/blackbox-adversarial-reprogramming"),
    ("Deep Residual Learning for Image Recognition", "https://github.com/tanjeffreyz/deep-residual-learning"),
    ("Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning", "https://github.com/IBM/model-reprogramming"),
    ("Feature Pyramid Networks for Object Detection", "https://github.com/unsky/FPN"),
    ("BlackVIP: Black-box Visual Prompting for Robust Transfer Learning", "https://github.com/changdaeoh/BlackVIP"),
    ("An Automated Visual Prompting Framework and Benchmark","https://github.com/IBM/AutoVP")
]
# Understanding and improving visual prompting: A label-mapping perspective
# Exploring visual prompts for adapting large-scale models
# Adversarial reprogramming of neural networks
# Transfer learning without knowing: Reprogramming black-box ML models
# Deep Residual Learning for Image Recognition (ResNet) & An Image is Worth 16x16 Words (ViT)
# Model Reprogramming: Resource-Efficient Cross-Domain Machine Learning
# Feature Pyramid Networks for Object Detection 
# BlackVIP: Black-box Visual Prompting for Robust Transfer Learning
# An Automated Visual Prompting Framework and Benchmark

# ==============================================================================
#  2. 解耦的、高效的并行下载辅助函数
# ==============================================================================

def download_paper(title: str, arxiv_scraper: ArxivScraper, config: Dict) -> Optional[Tuple[str, str]]:
    """下载单篇论文，成功时返回 (标题, 路径)"""
    log_prefix = f"[PaperDownloader | {title[:30]}...]"
    try:
        print(f"{log_prefix} Starting download.")
        paper_path = arxiv_scraper.run(title, save_path=config['paper_save_path'])
        if not paper_path or not os.path.exists(paper_path):
            raise FileNotFoundError(f"Failed to download paper for title: {title}")
        print(f"{log_prefix} SUCCESS.")
        return title, paper_path
    except Exception as e:
        logging.error(f"{log_prefix} FAILED. Error: {e}", exc_info=True)
        print(f"{log_prefix} FAILED. Reason: {e}")
        return None

def download_code(title: str, url: str, github_scraper: GithubScraper, config: Dict) -> Optional[Tuple[str, str]]:
    """下载单个代码库，成功时返回 (标题, 路径)"""
    clean_url = url.strip()
    log_prefix = f"[CodeDownloader | {title[:30]}...]"
    try:
        print(f"{log_prefix} Starting clone.")
        code_path = github_scraper.run(clean_url, save_path=config['code_save_path'])
        if not code_path or not os.path.exists(code_path):
            raise FileNotFoundError(f"Failed to clone repository: {clean_url}")
        print(f"{log_prefix} SUCCESS.")
        return title, code_path
    except Exception as e:
        logging.error(f"{log_prefix} FAILED. Error: {e}", exc_info=True)
        print(f"{log_prefix} FAILED. Reason: {e}")
        return None

# ==============================================================================
#  3. 核心处理函数 (带有调试日志的简洁版)
# ==============================================================================
def process_single_pair(paper_path: str, code_path: str, config: Dict) -> Dict:
    """
    处理单个 (论文, 代码) 对，生成知识图谱节点。
    """
    pid = os.getpid()
    log_prefix = f"[Processor {pid} | {os.path.basename(paper_path)}]"
    print(f"{log_prefix} Starting processing.")

    result = {
        "paper_path": paper_path, "code_path": code_path, "status": "FAILURE",
        "paper_title": None, "node_path": None, "error": None
    }

    try:
        # 在每个子进程中独立初始化工具实例
        paper_parser = PaperParser(model=config['model'])
        code_parser = CodeParser(model=config['model'])
        graph_constructor = GraphHandler(model=config['model'])
        
        # 增加详细的日志，用于调试卡住问题
        print(f"{log_prefix} Parsing paper...")
        paper = paper_parser.run(paper_path=paper_path)
        print(f"{log_prefix} Finished parsing paper. Parsing code...")
        
        code = code_parser.run(code_path=code_path)
        print(f"{log_prefix} Finished parsing code. Generating graph node...")
        
        result["paper_title"] = paper.title

        node = graph_constructor.generate_node(paper=paper, code=code, save_path=config['kg_path'], llm_rerank=True)
        print(f"{log_prefix} Finished generating graph node.")

        if node is None:
            raise RuntimeError(f"Failed to generate graph node for paper: {paper_path}")

        result["status"] = "SUCCESS"
        result["node_path"] = os.path.join(config['kg_path'], f"{paper.title}.json")
        print(f"{log_prefix} SUCCESS! Node saved to {result['node_path']}")

    except Exception as e:
        logging.error(f"{log_prefix} FAILURE", exc_info=True)
        result["error"] = str(e)

    return result

# ==============================================================================
#  4. 主执行逻辑
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--profile", type=str, default="basic-deepseek-v3", help="Configuration profile to use.")
    args = parser.parse_args()

    initialize_app()
    config = get_config()
    MAX_WORKERS = config.get('max_workers', 15)
    current_model_name = config.get('model', 'N/A')

    # --- 配置数据存储路径 ---
    config['paper_save_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper"
    config['code_save_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code"
    config['kg_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/kg_additional_another"
    log_file_path = os.path.join(".", 'build_kg_results_another.jsonl')

    for path in [config['paper_save_path'], config['code_save_path'], config['kg_path']]:
        os.makedirs(path, exist_ok=True)

    # ==========================================================================
    #  阶段 1: 高效并行下载所有资源
    # ==========================================================================
    print("\n" + "="*20 + " STAGE 1: FULLY PARALLEL DOWNLOADING " + "="*20)

    arxiv_scraper = ArxivScraper(model=config['model'])
    github_scraper = GithubScraper(model=config['model'])

    downloaded_papers = {}
    downloaded_codes = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 1. 一次性提交所有下载任务（论文和代码）
        futures = []
        for title, url in PAPER_CODE_TO_EXTRACT:
            futures.append(executor.submit(download_paper, title, arxiv_scraper, config))
            futures.append(executor.submit(download_code, title, url, github_scraper, config))

        # 2. 异步收集所有已完成的结果
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                # 使用标题作为键，来区分是哪个任务的结果
                key_title, path = result
                # 判断是论文还是代码的路径，并存入对应的字典
                if "paper" in config['paper_save_path'] and config['paper_save_path'] in path:
                    downloaded_papers[key_title] = path
                elif "code" in config['code_save_path'] and config['code_save_path'] in path:
                    downloaded_codes[key_title] = path

    # 3. 组合成功下载的对
    PATH_PAIRS_TO_PROCESS = []
    print("\n" + "="*20 + " PAIRING DOWNLOADED ITEMS " + "="*20)
    for title, _ in PAPER_CODE_TO_EXTRACT:
        if title in downloaded_papers and title in downloaded_codes:
            paper_path = downloaded_papers[title]
            code_path = downloaded_codes[title]
            PATH_PAIRS_TO_PROCESS.append((paper_path, code_path))
            print(f"[Pairing] SUCCESS: Paired '{title}'")
        else:
            print(f"[Pairing] FAILED: Missing paper or code for '{title}'")
            
    if not PATH_PAIRS_TO_PROCESS:
        print("\nNo complete pairs were successfully downloaded. Exiting.")
        return

    # ==========================================================================
    #  阶段 2: 并行处理已下载的本地路径对
    # ==========================================================================
    print("\n" + "="*20 + f" STAGE 2: PROCESSING {len(PATH_PAIRS_TO_PROCESS)} DOWNLOADED PAIRS " + "="*20)
    print(f"Using profile: '{args.profile}' with model '{current_model_name}' and up to {MAX_WORKERS} workers.")
    print(f"Results will be logged to: {log_file_path}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(log_file_path, 'w') as log_file:

        future_to_pair = {
            executor.submit(process_single_pair, p_path, c_path, config): (p_path, c_path)
            for p_path, c_path in PATH_PAIRS_TO_PROCESS
        }

        total_success = 0
        total_failure = 0

        for future in concurrent.futures.as_completed(future_to_pair):
            try:
                result = future.result()
                log_file.write(json.dumps(result) + '\n')
                log_file.flush()
                if result["status"] == "SUCCESS":
                    total_success += 1
                else:
                    total_failure += 1
            except Exception as e:
                pair_info = future_to_pair[future]
                error_result = {
                    "paper_path": pair_info[0], "code_path": pair_info[1],
                    "status": "CRITICAL_FAILURE", "error": f"Process-level exception: {str(e)}"
                }
                log_file.write(json.dumps(error_result) + '\n')
                log_file.flush()
                total_failure += 1
                logging.error(f"Critical failure for pair {pair_info[0]}: {e}", exc_info=True)

    # --- 最终统计 ---
    print("\n" + "="*20 + " OVERALL SUMMARY " + "="*20)
    print(f"Total pairs attempted: {len(PAPER_CODE_TO_EXTRACT)}")
    print(f"Successfully formed pairs for processing: {len(PATH_PAIRS_TO_PROCESS)}")
    print("---------------------------------------------------------")
    print(f"Total pairs processed: {total_success + total_failure}")
    print(f"  - Succeeded: {total_success}")
    print(f"  - Failed:    {total_failure}")
    print(f"\nDetailed processing results have been logged to: {log_file_path}")
    print("="*50)

if __name__ == "__main__":
    main()
