import argparse
import concurrent.futures
import json
from .utils import get_config, initialize_app
from .components import GraphHandler, PaperParser
from .schema import Paper, Code

# 将单个文件的处理逻辑封装成一个函数，供线程池调用
def process_paper(paper_path: str, graph_handler: GraphHandler, save_dir: str, paper_parser: PaperParser, config: dict) -> str:
    """
    加载并处理单个论文JSON文件，生成图节点。
    """
    try:
        # with open(paper_path, 'r', encoding='utf-8') as f:
        #     paper_data = json.load(f)
        
        # paper = Paper.from_dict(paper_data)
        paper = paper_parser.run(paper_path=paper_path, save_path=config['raw_data_path'] + "/paper", paper_format="latex")
        # 假设 code 始终为 None，根据您的原代码
        code = None
        
        # 调用核心处理方法
        graph_handler.generate_node(paper=paper, code=code, save_path=save_dir)
        
        print(f"[SUCCESS] 已处理: {paper.title}")
        return paper_path
    except Exception as e:
        print(f"[ERROR] 处理失败: {paper_path}, 原因: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="KG Builder - Multi-threaded")
    parser.add_argument("--profile", type=str, default="basic-deepseek-v3")
    args = parser.parse_args()

    initialize_app()
    config = get_config()

    # paper_path_list = [
    #     # "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2404.09636v3/All-in-one_simulation-based_inference_llm.json",
    #     # "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2401.01967v1/A_Mechanistic_Understanding_of_Alignment_Algorithms_A_Case_Study_on_DPO_and_Toxicity_llm.json"
    #     # "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2306.17806v1/Stay_on_topic_with_Classifier-Free_Guidance_llm.json",
    #     # "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2404.01650v2/Test-Time_Model_Adaptation_with_Only_Forward_Passes_llm.json",
    #     "/disk/disk_20T/luoyujie/ResearchKG/storage/raw/paper/2402.17135v1/Unsupervised_Zero-Shot_Reinforcement_Learning_via_Functional_Reward_Encodings_llm.json"
    # ]
    raw_paper_path = [
        # "./storage/raw/paper/2306.17806v1/",
        "./storage/raw/paper/2404.01650v2/",
        "./storage/raw/paper/2402.17135v1/",
        "./storage/raw/paper/2404.09636v3/",
        "./storage/raw/paper/2401.01967v1/"
    ]
    save_directory = "/disk/disk_20T/luoyujie/ResearchKG/storage/raw"
    
    # 只需要创建一个 GraphHandler 实例，在所有线程中共享
    # 确保 GraphHandler 的 generate_node 方法是线程安全的（大部分API调用都是）
    graph_constructor = GraphHandler(model=config['model'])
    paper_parser = PaperParser(model=config['model'])   
    print(f"开始并行处理 {len(raw_paper_path)} 个文件，最多使用 5 个线程...")

    # 使用 ThreadPoolExecutor 进行多线程处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # 提交所有任务到线程池
        future_to_path = {
            executor.submit(process_paper, path, graph_constructor, save_directory, paper_parser, config): path
            for path in raw_paper_path
        }

        # 等待任务完成并打印结果
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                future.result()  # 获取任务结果，如果任务中出现异常，这里会重新抛出
            except Exception as exc:
                print(f"[FATAL] 任务 {path} 产生了一个未捕获的异常: {exc}")

    print("所有任务处理完成。")

if __name__ == "__main__":
    main()
