# build_kg.py

import argparse
import concurrent.futures
import logging
import os
from typing import List, Tuple, Dict
import json # 用于记录结果

from .utils import get_config, initialize_app
# ArxivScraper 和 GithubScraper 不再需要，但保留以防未来使用
from .tools import ArxivScraper, GithubScraper, PaperParser, CodeParser, GraphHandler
from .schema import Paper, Code

# ==============================================================================
#  1. 定义要处理的任务 (现在直接使用本地路径)
# ==============================================================================
# PAIRS_TO_PROCESS 不再使用
# PAIRS_TO_PROCESS: List[Tuple[str, str]] = [ ... ]
PAPER_CODE_TO_EXTRACT: List[Tuple[str, str]] = [
    ("Denoising Diffusion Probabilistic Models", " https://github.com/hojonathanho/diffusion"),
    ("High-Resolution Image Synthesis with Latent Diffusion Models", "https://github.com/CompVis/latent-diffusion"),
    ("Image Generation From Small Datasets via Batch Statistics Adaptation", "https://github.com/nogu-atsu/small-dataset-image-generation"),
    ("Diffusion Guided Domain Adaptation of Image Generators", "https://github.com/KunpengSong/styleganfusion"),
    ("Back to the Source: Diffusion-Driven Test-Time Adaptation", "https://github.com/shiyegao/DDA"),
    ("Diffusion-based Probabilistic Uncertainty Estimation for Active Domain Adaptation", "https://github.com/TL-UESTC/DAPM"),
    ("Deep Unsupervised Learning using Nonequilibrium Thermodynamics", "https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models"),
    ("Denoising Diffusion Implicit Models", "https://github.com/ermongroup/ddim")
]
# Denoising diffusion probabilistic models 
# High-Resolution Image Synthesis with Latent Diffusion Models
# Image Generation from Small Datasets via Batch Statistics Adaptation
# Domain-Adversarial Training of Neural Networks pass
# StyleGAN-Fusion: Diffusion Guided Domain Adaptation of Image Generators
# Back to the Source: Diffusion-Driven Adaptation to Test-Time Corruption
# Diffusion-based Probabilistic Uncertainty Estimation for Active Domain Adaptation
# Deep Unsupervised Learning using Nonequilibrium Thermodynamics
# Auto-Encoding Variational Bayes
# Denoising Diffusion Implicit Models

arxiv_scraper = ArxivScraper(model=config['model'])
paper_path = arxiv_scraper.run("xxx", save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper", paper_format="latex")


github_scraper = GithubScraper(model=config['model'])
code_path = github_scraper.run("xxx", save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code")


PATH_PAIRS_TO_PROCESS: List[Tuple[str, str]] = [
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2006.10726v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/DequanWang_tent"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2302.12400v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mr-eggplant_SAR"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2203.13591v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/qinenergy_cotta"), 
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2310.03335v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/gengchenmai_OFTTA"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2508.12939v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/taolinzhang_BoostAdapter"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/1909.13231v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/yueatsprograms_ttt_imagenet_release"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/1911.08731v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/kohpangwei_group_DRO"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2312.01648v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/RandallBalestriero_SplineLLM"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2012.14913v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mega002_ff-layers"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2203.14680v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/aviclu_ffn-values"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2411.06424v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/Yushi-Y_dpo-toxic-neurons"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2503.11667v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/zhenyu-02_LogitLens4LLMs"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2301.05217v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/progress-measures-paper"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2302.03025", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mr-eggplant_SAR"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2305.18290v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/eric-mitchell_direct-preference-optimization"),
    ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2211.00593v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/Easy-Transformer")
]

# ==============================================================================
#  2. 简化后的处理函数 (直接读取本地路径)
# ==============================================================================
def process_single_pair(paper_path: str, code_path: str, config: Dict) -> Dict:
    """
    完整处理单个 (论文路径, 代码路径) 对：验证路径 -> 解析 -> 构建图节点
    返回一个包含处理结果的字典。
    """
    pid = os.getpid()
    # 使用论文目录名作为日志标识
    log_prefix = f"[Worker {pid} | {os.path.basename(paper_path)}]"
    print(f"{log_prefix} Starting.")
    
    # 结果记录字典
    result = {
        "paper_path": paper_path,
        "code_path": code_path,
        "status": "FAILURE",
        "paper_title": None, # 标题将在解析后获得
        "node_path": None,
        "error": None
    }

    try:
        # --- 初始化工具 (不再需要下载器) ---
        paper_parser = PaperParser(model=config['model'])
        code_parser = CodeParser(model=config['model'])
        graph_constructor = GraphHandler(model=config['model'])

        # --- a. 验证路径 (替换了下载步骤) ---
        print(f"{log_prefix} Verifying paths...")
        if not os.path.exists(paper_path):
            raise FileNotFoundError(f"Paper path does not exist: {paper_path}")
        if not os.path.exists(code_path):
            raise FileNotFoundError(f"Code path does not exist: {code_path}")
        
        print(f"{log_prefix} Paths verified. Starting analysis...")

        # --- b. 解析 ---
        paper = paper_parser.run(paper_path=paper_path, paper_format="latex")
        code = code_parser.run(code_path=code_path)
        
        # 解析后，更新结果中的标题
        result["paper_title"] = paper.title

        # --- c. 构建图节点 ---
        node = graph_constructor.generate_node(paper=paper, code=code, save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/kg_raw", llm_rerank=False)
        
        if node is None:
            raise RuntimeError(f"Failed to generate graph node for paper: {paper_path}")

        result["status"] = "SUCCESS"
        result["node_path"] = os.path.join(config['kg_path'], f"{paper.title}.json")
        print(f"{log_prefix} SUCCESS! Node saved to {result['node_path']}")

    except Exception as e:
        # 记录任何步骤中发生的错误
        logging.error(f"{log_prefix} FAILURE", exc_info=True)
        result["error"] = str(e)
    
    return result

# ==============================================================================
#  3. 主执行逻辑 (已更新以使用路径对)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="KG Builder")
    parser.add_argument("--profile", type=str, default="basic-deepseek-v3")
    args = parser.parse_args()

    initialize_app()
    config = get_config()
    MAX_WORKERS = config.get('max_workers', 15)
    
    # 定义日志文件路径
    log_file_path = os.path.join(config.get('log_path', '.'), 'build_kg_results.jsonl')
    
    print("\n" + "="*20 + " STARTING KNOWLEDGE GRAPH CONSTRUCTION " + "="*20)
    print(f"Processing {len(PATH_PAIRS_TO_PROCESS)} pairs from local paths using up to {MAX_WORKERS} workers.")
    print(f"Results will be saved to: {log_file_path}")

    # 使用 ProcessPoolExecutor 来实现真正的并行计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor, \
         open(log_file_path, 'w') as log_file:
        
        # 提交所有任务，现在迭代路径对
        future_to_pair = {
            executor.submit(process_single_pair, p_path, c_path, config): (p_path, c_path)
            for p_path, c_path in PATH_PAIRS_TO_PROCESS
        }
        
        total_success = 0
        total_failure = 0
        
        # 任务完成后立即处理结果并写入文件
        for future in concurrent.futures.as_completed(future_to_pair):
            try:
                result = future.result()
                
                # 实时写入JSON Lines格式的日志文件
                log_file.write(json.dumps(result) + '\n')
                log_file.flush() # 确保立即写入磁盘

                if result["status"] == "SUCCESS":
                    total_success += 1
                else:
                    total_failure += 1
            except Exception as e:
                # 捕获进程级别的严重错误
                pair_info = future_to_pair[future]
                error_result = {
                    "paper_path": pair_info[0], # 记录原始路径
                    "code_path": pair_info[1],  # 记录原始路径
                    "status": "CRITICAL_FAILURE",
                    "error": f"Process-level exception: {str(e)}"
                }
                log_file.write(json.dumps(error_result) + '\n')
                log_file.flush()
                total_failure += 1
                logging.error(f"Critical failure for pair {pair_info[0]}: {e}", exc_info=True)

    print("\n==================== OVERALL SUMMARY ====================")
    print(f"Total pairs to process: {len(PATH_PAIRS_TO_PROCESS)}")
    print(f"  - Succeeded: {total_success}")
    print(f"  - Failed:    {total_failure}")
    print(f"\nDetailed results have been logged to: {log_file_path}")
    print("=========================================================")

if __name__ == "__main__":
    main()
