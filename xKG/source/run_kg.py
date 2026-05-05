import argparse
import json
import os
import traceback
import concurrent.futures
import logging  # <--- ADDED

from .utils import *
from .llm import *
from .tools import *
from .schema import *

# ==============================================================================
#  Worker Function: Processes an entire "bundle" of tasks for a single category.
#  This function will be executed in a separate process.
# ==============================================================================
def process_category_bundle(category_name, pairs_in_bundle, config):
    """
    Processes a list of paper-code pairs belonging to a single category.
    """
    pid = os.getpid()
    log_prefix = f"[Process {pid} | Category: {category_name}]"
    logger = logging.getLogger(__name__)  # <--- ADDED: Get a logger instance

    print(f"{log_prefix} Started. Processing {len(pairs_in_bundle)} pairs.")
    
    bundle_success_count = 0
    bundle_failure_count = 0

    # 在每个进程中独立初始化，避免状态共享
    paper_parser = PaperParser(model=config['model'])
    code_parser = CodeParser(model=config['model'])
    graph_constructor = GraphHandler(model=config['model'])

    for i, (paper_dir, code_dir) in enumerate(pairs_in_bundle):
        pair_log_prefix = f"{log_prefix} Pair {i+1}/{len(pairs_in_bundle)} ({os.path.basename(paper_dir)})"
        print(f"{pair_log_prefix}: Starting...")
        
        try:
            # --- 1. 解析论文和代码 ---
            print(f"{pair_log_prefix}: Parsing paper...")
            paper = paper_parser.run(paper_path=paper_dir, paper_format="latex", save_path=paper_dir)
            if not paper:
                raise ValueError("Paper parsing returned None.")

            print(f"{pair_log_prefix}: Parsing code...")
            code = code_parser.run(code_path=code_dir, save_path=code_dir)
            if not code:
                raise ValueError("Code parsing returned None.")

            # <--- MODIFIED: Capture the return value of generate_node ---
            node = graph_constructor.generate_node(
                paper=paper,
                code=code,
                save_path=config['kg_path']
            )
            
            # <--- ADDED: Check if node generation failed (returned None) ---
            if node is None:
                # Log the specific error as requested
                logger.error(f"Failed to generate node for paper: {paper_dir}")
                bundle_failure_count += 1
            else:
                # Success case
                print(f"{pair_log_prefix}: SUCCESS - Node for '{paper.paper_title}' saved.")
                bundle_success_count += 1

        except Exception as e:
            # <--- MODIFIED: Use logger for exceptions ---
            error_message = f"An error occurred: {e}"
            logger.error(f"{pair_log_prefix}: FAILURE - {error_message}", exc_info=True) # exc_info=True adds traceback
            bundle_failure_count += 1
            # 继续处理捆绑包中的下一个任务

    print(f"{log_prefix} Finished. Success: {bundle_success_count}, Failure: {bundle_failure_count}.")
    return (category_name, bundle_success_count, bundle_failure_count)


# ==============================================================================
#  Main Function: Orchestrates the parallel execution of category bundles.
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="KG Builder")
    parser.add_argument(
        "--profile",
        type=str,
        default="basic-deepseek-v3",
        choices=["basic-deepseek-v3", "gpt-5-2025-08-07", "claude-sonnet-4-20250514", "deepseek-r1", "gemini-2-5-pro", "o1"],
        help="Configuration profile (e.g., basic-deepseek-v3)",
    )
    args = parser.parse_args()

    # 加载配置并初始化日志
    initialize_app()
    config = get_config()

    # 按类别捆绑的任务字典
    tasks_by_category = {
        # "fre": [
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2507.10485v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/pettza_EWC"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/1611.07725v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/srebuffi_iCaRL"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/1810.12894v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/openai_random-network-distillation"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2206.01626v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/google-research_reincarnating_rl"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2210.13846v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/zhaoyi11_adaptive_bc"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2010.05595v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/hastings24_rethinking_er"),
        # ],
        # "test-time-model-adaptation": [
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2006.10726v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/DequanWang_tent"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2302.12400v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mr-eggplant_SAR"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2203.13591v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/qinenergy_cotta"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2310.03335v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/gengchenmai_OFTTA"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2508.12939v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/taolinzhang_BoostAdapter"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/1909.13231v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/yueatsprograms_ttt_imagenet_release"),
        # ],
        # "mechanistic-understanding": [
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2312.01648v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/RandallBalestriero_SplineLLM"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2012.14913v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mega002_ff-layers"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2203.14680v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/aviclu_ffn-values"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2411.06424v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/Yushi-Y_dpo-toxic-neurons"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2503.11667v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/zhenyu-02_LogitLens4LLMs"),
        # ],
        "all-in-one": [
            # ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2508.12939v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/sbi-dev_sbi"),
            # ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2207.05636v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/kmzzhang_nbi"),
            ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2505.22573v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mackelab_fnope"),
            # ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2506.06087v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/yugahikida_multilevel-sbi"),
            # ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2308.01054v3", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/dirmeier_ssnl"),
            # ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2203.06481v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/mackelab_gatsbi"),
        ],
        # "stay-on-topic-with-classifier-free-guidance": [
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2207.12598v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/lucidrains_classifier-free-guidance-pytorch"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2406.08070v2", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/CFGpp-diffusion_CFGpp"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2502.10574v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/gmum_beta-CFG"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2409.03755v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/wl-zhao_DC-Solver"),
        #     ("/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2404.05384v1", "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/code/shiimizu_ComfyUI-semantic-aware-guidance"),
        # ]
    }
    
    MAX_WORKERS = 5
    print(f"Starting parallel processing with {MAX_WORKERS} workers, one for each of the {len(tasks_by_category)} categories.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_category = {
            executor.submit(process_category_bundle, category, pairs, config): category
            for category, pairs in tasks_by_category.items()
        }
        
        total_success = 0
        total_failure = 0
        
        for future in concurrent.futures.as_completed(future_to_category):
            category = future_to_category[future]
            try:
                cat, successes, failures = future.result()
                print(f"--- Bundle '{cat}' Completed: {successes} succeeded, {failures} failed. ---")
                total_success += successes
                total_failure += failures
            except Exception as exc:
                logging.error(f"--- [CRITICAL FAILURE] Bundle for category '{category}' terminated with an exception: {exc} ---", exc_info=True)
                total_failure += len(tasks_by_category[category])

    total_tasks = sum(len(pairs) for pairs in tasks_by_category.values())
    print("\n==================== OVERALL SUMMARY ====================")
    print(f"Total tasks across all categories: {total_tasks}")
    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failure}")
    print("=========================================================")

if __name__ == "__main__":
    main()
