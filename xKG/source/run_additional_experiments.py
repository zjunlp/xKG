# build_kg.py (仅优化下载并行逻辑，保留统计功能)

import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import os
from typing import List, Tuple, Dict, Optional
import json
from multiprocessing import Manager  # 保留Manager

# 导入你的模块
from .utils import get_config, initialize_app
from .tools import ArxivScraper, GithubScraper, PaperParser, CodeParser, GraphHandler
from .llm import llm_def # 保留llm_def的导入

# ==============================================================================
#  1. 定义要处理的源任务
# ==============================================================================
PAPER_CODE_TO_EXTRACT: List[Tuple[str, str]] = [
    # ("Denoising Diffusion Probabilistic Models", " https://github.com/hojonathanho/diffusion"),
    # ("High-Resolution Image Synthesis with Latent Diffusion Models", "https://github.com/CompVis/latent-diffusion"),
    # ("Image Generation From Small Datasets via Batch Statistics Adaptation", "https://github.com/nogu-atsu/small-dataset-image-generation"),
    # ("Diffusion Guided Domain Adaptation of Image Generators", "https://github.com/KunpengSong/styleganfusion"),
    # ("Back to the Source: Diffusion-Driven Test-Time Adaptation", "https://github.com/shiyegao/DDA"),
    # ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics", "https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models"),
    # ("Denoising Diffusion Implicit Models", "https://github.com/ermongroup/ddim"),
    # ("Diffusion Models Beat GANs on Image Synthesis", "https://github.com/openai/guided-diffusion"),
    ("Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction", None)
]
# Diffusion Models Beat GANs on Image Synthesis
# Parameter-Efficient Transfer Learning for NLP
# Image Generation from Small Datasets via Batch Statistics Adaptation
# Graph Transfer Learning via Adversarial Domain Adaptation with Graph Convolution
# High-Resolution Image Synthesis with Latent Diffusion Models
# ==============================================================================
#  2. 新的、高效的并行下载辅助函数
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
#  3. 核心处理函数 (保持不变，保留统计逻辑)
# ==============================================================================
def process_single_pair(
    paper_path: str,
    code_path: str,
    config: Dict,
    shared_stats: Dict  # 保留此参数
) -> Dict:
    """
    通过猴子补丁来拦截LLM调用并计数，处理单个(论文,代码)对。
    (此函数完全保持不变)
    """
    pid = os.getpid()
    log_prefix = f"[Processor {pid} | {os.path.basename(paper_path)}]"
    print(f"{log_prefix} Starting processing.")

    result = {
        "paper_path": paper_path, "code_path": code_path, "status": "FAILURE",
        "paper_title": None, "node_path": None, "error": None
    }

    original_query = llm_def.OpenAIBackend.query
    def token_counting_wrapper(self, *args, **kwargs):
        output, req_time, in_tokens, out_tokens, info = original_query(self, *args, **kwargs)
        shared_stats["in_tokens"] += in_tokens
        shared_stats["out_tokens"] += out_tokens
        return output, req_time, in_tokens, out_tokens, info

    try:
        llm_def.OpenAIBackend.query = token_counting_wrapper
        paper_parser = PaperParser(model=config['model'])
        code_parser = CodeParser(model=config['model'])
        graph_constructor = GraphHandler(model=config['model'])

        if not os.path.exists(paper_path): raise FileNotFoundError(f"Paper path does not exist: {paper_path}")
        if not os.path.exists(code_path): raise FileNotFoundError(f"Code path does not exist: {code_path}")
        
        print(f"{log_prefix} Parsing paper...")
        paper = paper_parser.run(paper_path=paper_path)
        print(f"{log_prefix} Finished parsing paper. Parsing code...")
        code = code_parser.run(code_path=code_path)
        print(f"{log_prefix} Finished parsing code. Generating graph node...")
        
        result["paper_title"] = paper.title
        node = graph_constructor.generate_node(paper=paper, code=code, save_path=config['kg_path'], llm_rerank=True)
        print(f"{log_prefix} Finished generating graph node.")
        
        if node is None: raise RuntimeError(f"Failed to generate graph node for paper: {paper_path}")

        result["status"] = "SUCCESS"
        result["node_path"] = os.path.join(config['kg_path'], f"{paper.title}.json")
        print(f"{log_prefix} SUCCESS! Node saved to {result['node_path']}")
    except Exception as e:
        logging.error(f"{log_prefix} FAILURE", exc_info=True)
        result["error"] = str(e)
    finally:
        llm_def.OpenAIBackend.query = original_query
    return result

# ==============================================================================
#  4. 主执行逻辑 (仅修改阶段1)
# ==============================================================================

# 价格表和成本计算函数 (保持不变)
MODEL_PRICE_TABLE = {
    "DMXAPI-DeepSeek-R1": {"in": 0.6320, "out": 2.5280, "currency": "$"}, 
    "gpt-4.1-mini-2025-04-14": {"in": 0.4000, "out": 1.6000, "currency": "$"}, 
    "gpt-4o-mini": {"in": 0.1500, "out": 0.6000, "currency": "$"}, 
    "o3-mini-2025-01-31": {"in": 1.1000, "out": 4.4000, "currency": "$"}, 
    "o4-mini-2025-04-16": {"in": 1.1000, "out": 4.4000, "currency": "$"}, 
    "claude-sonnet-4-20250514": {"in": 3.0000, "out": 15.0000, "currency": "$"}, 
    "gemini-2.5-pro-preview-06-05": {"in": 1.2500, "out": 10.0000, "currency": "$"}, 
    "DMXAPI-HuoShan-DeepSeek-V3": {"in": 0.3160, "out": 1.2640, "currency": "$"}, 
    "deepseek-v3.1-nothinking": {"in": 0.4000, "out": 1.6000, "currency": "$"}, 
    "text-embedding-3-small": {"in": 0.0200, "out": 0.0200, "currency": "$"},
}
def calculate_cost(total_in_tokens: int, total_out_tokens: int, in_cost_per_million: float, out_cost_per_million: float, currency_symbol: str = '$') -> str:
    in_cost = (total_in_tokens / 1_000_000) * in_cost_per_million
    out_cost = (total_out_tokens / 1_000_000) * out_cost_per_million
    total_cost = in_cost + out_cost
    return f"{currency_symbol}{total_cost:.4f}"

def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph Builder")
    parser.add_argument("--profile", type=str, default="basic-deepseek-v3", help="Configuration profile to use.")
    args = parser.parse_args()

    initialize_app()
    config = get_config()
    MAX_WORKERS = config.get('max_workers', 15)
    current_model_name = config.get('model')

    # --- 配置数据存储路径 (保持不变) ---
    config['paper_save_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper"
    config['code_save_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code"
    config['kg_path'] = "/disk/disk_20T/luoyujie/ResearchKG/storage/kg_additional"
    log_file_path = os.path.join(".", 'build_kg_results.jsonl')
    for path in [config['paper_save_path'], config['code_save_path'], config['kg_path']]:
        os.makedirs(path, exist_ok=True)

    # ==========================================================================
    #  阶段 1: 高效并行下载所有资源 (修改部分)
    # ==========================================================================
    print("\n" + "="*20 + " STAGE 1: FULLY PARALLEL DOWNLOADING " + "="*20)

    arxiv_scraper = ArxivScraper(model=config['model'])
    github_scraper = GithubScraper(model=config['model'])

    downloaded_papers = {}
    downloaded_codes = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for title, url in PAPER_CODE_TO_EXTRACT:
            futures.append(executor.submit(download_paper, title, arxiv_scraper, config))
            futures.append(executor.submit(download_code, title, url, github_scraper, config))

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                key_title, path = result
                # 使用路径中的特定目录名来区分是论文还是代码
                if os.path.basename(config['paper_save_path']) in path:
                    downloaded_papers[key_title] = path
                elif os.path.basename(config['code_save_path']) in path:
                    downloaded_codes[key_title] = path

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
    #  阶段 2: 并行处理 (此部分完全保持不变，包括Manager和统计)
    # ==========================================================================
    with Manager() as manager:
        shared_stats = manager.dict({"in_tokens": 0, "out_tokens": 0})

        print("\n" + "="*20 + f" STAGE 2: PROCESSING {len(PATH_PAIRS_TO_PROCESS)} DOWNLOADED PAIRS " + "="*20)
        print(f"Using profile: '{args.profile}' with model '{current_model_name}' and up to {MAX_WORKERS} workers.")
        print(f"Results will be logged to: {log_file_path}")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor, \
             open(log_file_path, 'w') as log_file:

            future_to_pair = {
                executor.submit(process_single_pair, p_path, c_path, config, shared_stats): (p_path, c_path)
                for p_path, c_path in PATH_PAIRS_TO_PROCESS
            }

            total_success, total_failure = 0, 0
            for future in concurrent.futures.as_completed(future_to_pair):
                try:
                    result = future.result()
                    log_file.write(json.dumps(result) + '\n')
                    log_file.flush()
                    if result["status"] == "SUCCESS": total_success += 1
                    else: total_failure += 1
                except Exception as e:
                    pair_info = future_to_pair[future]
                    error_result = {"paper_path": pair_info[0], "code_path": pair_info[1], "status": "CRITICAL_FAILURE", "error": f"Process-level exception: {str(e)}"}
                    log_file.write(json.dumps(error_result) + '\n')
                    log_file.flush()
                    total_failure += 1
                    logging.error(f"Critical failure for pair {pair_info[0]}: {e}", exc_info=True)

        # --- 最终统计 (保持不变) ---
        print("\n" + "="*20 + " OVERALL SUMMARY " + "="*20)
        print(f"Total pairs attempted: {len(PAPER_CODE_TO_EXTRACT)}")
        print(f"Successfully formed pairs for processing: {len(PATH_PAIRS_TO_PROCESS)}")
        print("---------------------------------------------------------")
        print(f"Total pairs processed: {total_success + total_failure}")
        print(f"  - Succeeded: {total_success}")
        print(f"  - Failed:    {total_failure}")
        
        print("\n" + "-"*20 + " TOKEN & COST SUMMARY " + "-"*20)
        overall_in_tokens = shared_stats["in_tokens"]
        overall_out_tokens = shared_stats["out_tokens"]
        print(f"Total Input Tokens:  {overall_in_tokens:,}")
        print(f"Total Output Tokens: {overall_out_tokens:,}")
        
        count_cost_model_name = "o4-mini-2025-04-16"
        if count_cost_model_name and count_cost_model_name in MODEL_PRICE_TABLE:
            prices = MODEL_PRICE_TABLE[count_cost_model_name]
            total_cost_str = calculate_cost(overall_in_tokens, overall_out_tokens, prices["in"], prices["out"], prices["currency"])
            print(f"Estimated Total Cost for model '{count_cost_model_name}': {total_cost_str}")
        else:
            print(f"Could not find price information for model '{count_cost_model_name}'.")

        print(f"\nDetailed processing results have been logged to: {log_file_path}")
        print("="*50)

if __name__ == "__main__":
    main()
