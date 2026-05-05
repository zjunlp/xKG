"""
极简版知识图谱构建脚本（带统计功能）。
- 只处理指定的纯论文任务。
- 按顺序执行，无并行。
- 移除了所有与代码处理和复杂配对相关的逻辑。
- 新增：LLM调用次数、Token消耗和成本估算统计。
"""

import os
import logging

# 确保这里的导入路径相对于您的项目结构是正确的
from .utils import get_config, initialize_app
from .components import ArxivScraper, PaperParser, GraphHandler
from .llm import llm_def  # <-- 新增：导入llm_def以进行猴子补丁

# ==============================================================================
#  1. 价格表和成本计算函数 (从第二个脚本移植)
# ==============================================================================
MODEL_PRICE_TABLE = {
    # (价格表保持不变，只显示部分)
    "gpt-4o-mini": {"in": 0.1500, "out": 0.6000, "currency": "$"},
    "deepseek-v3.1-nothinking": {"in": 0.4000, "out": 1.6000, "currency": "$"},
    "text-embedding-3-small": {"in": 0.0200, "out": 0.0200, "currency": "$"},
    # ... 您可以根据需要添加更多模型
}

def calculate_cost(
    total_in_tokens: int,
    total_out_tokens: int,
    in_cost_per_million: float,
    out_cost_per_million: float,
    currency_symbol: str = '$'
) -> str:
    """根据Token数量和单价计算总成本。"""
    in_cost = (total_in_tokens / 1_000_000) * in_cost_per_million
    out_cost = (total_out_tokens / 1_000_000) * out_cost_per_million
    total_cost = in_cost + out_cost
    return f"{currency_symbol}{total_cost:.4f}"

# ==============================================================================
#  2. 定义要处理的论文标题
# ==============================================================================
PAPERS_TO_PROCESS = [
    "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction" # 用于测试错误处理
]

# ==============================================================================
#  3. 主执行函数 (集成统计功能)
# ==============================================================================
def main():
    """
    顺序处理每一篇论文，并在结束后统计LLM调用成本。
    """
    # --- 1. 初始化和配置 ---
    print("Initializing application...")
    initialize_app()
    config = get_config()
    current_model_name = config.get('model')
    
    paper_save_path = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper"
    kg_path = "/disk/disk_20T/luoyujie/ResearchKG/storage/target"
    
    os.makedirs(paper_save_path, exist_ok=True)
    os.makedirs(kg_path, exist_ok=True)

    # --- 2. 准备统计和猴子补丁 ---
    # 因为是单线程，普通的字典就足够了，不需要Manager
    stats = {"in_tokens": 0, "out_tokens": 0, "api_calls": 0}

    # 保存原始的LLM查询方法
    original_query = llm_def.OpenAIBackend.query

    # 定义一个包装函数，用于拦截调用、计数，然后执行原始调用
    def token_counting_wrapper(self, *args, **kwargs):
        # 执行原始的查询方法
        output, req_time, in_tokens, out_tokens, info = original_query(self, *args, **kwargs)
        
        # 更新统计数据
        stats["in_tokens"] += in_tokens
        stats["out_tokens"] += out_tokens
        stats["api_calls"] += 1
        
        print(f"      [LLM Call] Tokens In: {in_tokens}, Out: {out_tokens}. Total calls: {stats['api_calls']}")
        
        # 返回原始结果，不改变任何行为
        return output, req_time, in_tokens, out_tokens, info

    # 应用猴子补丁：用我们的包装函数替换原始方法
    llm_def.OpenAIBackend.query = token_counting_wrapper
    print(f"Applied monkey patch to '{llm_def.OpenAIBackend.__name__}.query' for statistics.")

    try:
        # --- 3. 实例化所有需要的工具 ---
        # (在循环外实例化，避免重复加载模型)
        arxiv_scraper = ArxivScraper(model=config['model'])
        paper_parser = PaperParser(model=config['model'])
        graph_constructor = GraphHandler(model=config['model'])

        # --- 4. 顺序处理每一篇论文 ---
        total_success, total_failure = 0, 0
        for title in PAPERS_TO_PROCESS:
            print("\n" + "="*80)
            print(f"Processing paper: '{title}'")
            print("="*80)
            
            try:
                # --- 步骤 1: 下载论文 ---
                print(f"[1/3] Downloading paper PDF...")
                paper_path = arxiv_scraper.run(title, save_path=paper_save_path)
                if not paper_path or not os.path.exists(paper_path):
                    raise FileNotFoundError("ArxivScraper did not return a valid file path.")
                print(f"      -> Success! Saved to: {paper_path}")

                # --- 步骤 2: 解析论文内容 ---
                print(f"[2/3] Parsing paper content...")
                paper = paper_parser.run(paper_path=paper_path, save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/target")
                if not paper:
                    raise ValueError("PaperParser did not return a valid Paper object.")
                print(f"      -> Success! Parsed title: {paper.title}")

                # --- 步骤 3: 生成并保存知识图谱节点 (无代码) ---
                print(f"[3/3] Generating Knowledge Graph node...")
                node = graph_constructor.generate_node(
                    paper=paper, 
                    code=None,  # 明确告知没有代码
                    save_path=kg_path
                )
                if not node:
                    raise RuntimeError("GraphHandler failed to generate a Node object.")
                print(f"      -> Success! Node generated and saved.")
                total_success += 1

            except Exception as e:
                total_failure += 1
                error_message = f"!!! FAILED to process paper '{title}'. Error: {e}"
                print(error_message)
                logging.error(error_message, exc_info=True)

    finally:
        # --- 5. 无论成功与否，最后都恢复原始方法并打印统计摘要 ---
        # 这是一个好习惯，确保程序状态被清理
        llm_def.OpenAIBackend.query = original_query
        print("\nRestored original LLM query method.")

        print("\n" + "="*30 + " OVERALL SUMMARY " + "="*30)
        print(f"Total papers attempted: {len(PAPERS_TO_PROCESS)}")
        print(f"  - Succeeded: {total_success}")
        print(f"  - Failed:    {total_failure}")
        
        print("\n" + "-"*28 + " TOKEN & COST SUMMARY " + "-"*28)
        print(f"Model used: '{current_model_name}'")
        print(f"Total API Calls:     {stats['api_calls']:,}")
        print(f"Total Input Tokens:  {stats['in_tokens']:,}")
        print(f"Total Output Tokens: {stats['out_tokens']:,}")
        
        if current_model_name and current_model_name in MODEL_PRICE_TABLE:
            prices = MODEL_PRICE_TABLE[current_model_name]
            total_cost_str = calculate_cost(
                stats["in_tokens"], 
                stats["out_tokens"], 
                prices["in"], 
                prices["out"], 
                prices["currency"]
            )
            print(f"Estimated Total Cost: {total_cost_str}")
        else:
            print(f"Could not find price information for model '{current_model_name}'. Cost calculation skipped.")

        print("="*80)
        print("All tasks completed.")
        print("="*80)


if __name__ == "__main__":
    main()
