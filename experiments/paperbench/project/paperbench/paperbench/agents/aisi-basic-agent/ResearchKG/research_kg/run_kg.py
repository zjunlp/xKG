import argparse
from utils import *
from llm import *
from tools import *
from schema import *
import pathlib
def main():
    parser = argparse.ArgumentParser(description="KG Builder")
    parser.add_argument(
        "--profile",
        type=str,
        default="basic-deepseek-v3",
        choices=["basic-deepseek-v3",
                "gpt-5-2025-08-07", 
                "claude-sonnet-4-20250514",
                "deepseek-r1",
                "gemini-2-5-pro",
                "o1"],  
        help="Configuration profile (e.g., basic-deepseek-v3)",
    )
    args = parser.parse_args()

    # 加载配置并初始化日志
    initialize_app()
    config = get_config()
    # llm =  get_llm_backend()
    
    web_scraper = ArxivFetcher(model=config['model'])
    paper_parser = PaperParser(model=config['model'])
    github_scraper = GithubFetcher(model=config['model'])
    code_parser = CodeParser(model=config['model'])
    graph_constructor = GraphHandler(model=config['model'])

    paper_code_pair = [
        # ("/data/lyj/ResearchKG/storage/raw/paper/2211.00593v1", "/data/lyj/ResearchKG/storage/raw/code/Easy-Transformer"),
        ("/data/yzy/ResearchKG/storage/raw_rag/paper/2009.11462v2/RealToxicityPrompts_Evaluating_Neural_Toxic_Degeneration_in_Language_Models_llm.json", "/data/yzy/ResearchKG/storage/raw_rag/code/allenai_real-toxicity-prompts/allenai_real-toxicity-prompts_meta.json"),
        # ("/data/lyj/ResearchKG/storage/raw/paper/2302.03025", "/data/lyj/ResearchKG/storage/raw/code/rep-theory-mech-interp")
    ]
    for paper_path, code_path in paper_code_pair:
        with open(paper_path, "r", encoding="utf-8") as f:
            paper_dict = json.load(f)
        paper = Paper.from_dict(paper_dict)
        with open(code_path, "r", encoding="utf-8") as f:
            code_dict = json.load(f)
        code = Code.from_dict(code_dict)
        graph_constructor.load_kg(config['kg_path'])
        graph_constructor.build_index()
        node = graph_constructor.generate_node(paper=paper, code=code, save_path=config['kg_path'])
    # paper_meta_path = "/data/lyj/ResearchKG/storage/raw/paper/2211.00593v1/Interpretability_in_the_Wild_a_Circuit_for_Indirect_Object_Identification_in_GPT-2_small_llm.json"
    # with open(paper_meta_path, "r", encoding="utf-8") as f:
    #     paper_dict = json.load(f)
    # paper = Paper.from_dict(paper_dict)
    # graph_constructor.load_kg(config['kg_path'])
    # graph_constructor.build_index()
    # node = graph_constructor.generate_node(paper=paper, code=None)
    # node_path = "/data/lyj/ResearchKG/storage/kg/A_Mechanistic_Understanding_of_Alignment_Algorithms_A_Case_Study_on_DPO_and_Toxicity_graph.json"
    # with open(node_path, "r", encoding="utf-8") as f:
    #     node_dict = json.load(f)
    # node = Node.from_dict(node_dict)
    # graph_constructor.load_kg(config['kg_path'])
    # graph_constructor.build_index()
    # # res = graph_constructor.retrieve_paper_by_paper(query_node=node, top_k=5)
    # # for result in res:
    # #     print(f"Title: {result[0].paper_title}, Abstract: {result[0].paper_abstract}\nSimilarity: {result[1]}\n\n")
    # query = "toxic vector detection"
    # result = graph_constructor.retrieve_technique_by_query(query=query, top_k=5)
    # for res in result:
    #     print(f"Technique: {res[0].name}, {res[0].description} \nSimilarity: {res[1]}\n\n")
        
    # node_path = "/data/lyj/ResearchKG/research_kg/A_Mechanistic_Understanding_of_Alignment_Algorithms_A_Case_Study_on_DPO_and_Toxicity_graph.json"
    # with open(node_path, "r", encoding="utf-8") as f:
    #     node_dict = json.load(f)
    # node = Node.from_dict(node_dict)
    # graph_constructor.load_kg(config['kg_path'])
    # graph_constructor.build_index()
    # res = graph_constructor.retrieve_paper_by_paper(query_node=node, top_k=3)
    # for result in res:
    #     print(f"Title: {result[0].paper_title}, Abstract: {result[0].paper_abstract}\nSimilarity: {result[1]}\n\n")
    # web_scraper.run(query="Direct Preference Optimization: Your Language Model is Secretly a Reward Model", save_path=config['raw_data_path'])
    
    # paper_parser = PaperParser(model=config['model'])
    # paper = paper_parser.run(paper_path="/data/lyj/ResearchKG/storage/raw/paper/2401.01967v1", paper_format="latex", save_path="/data/lyj/ResearchKG/storage/raw/paper/2401.01967v1")

    # web_scraper = ArxivFetcher(model=config['model'])
    # path = web_scraper.run(query="A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity", save_path=config['raw_data_path'])  
    # paper_parser = PaperParser(model=config['model'])
    # paper = paper_parser.run(paper_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw/2305.18290v3", paper_format="latex", save_path="/disk/disk_20T/luoyujie/ResearchKG/storage/raw/2305.18290v3")
    
    # web_scraper = GithubFetcher()
    # web_scraper.run(repo_url="https://github.com/ajyl/dpo_toxic", save_path=os.path.join(config['raw_data_path'], "code"))
    # file_path = "/data/lyj/ResearchKG/storage/raw/paper/2401.01967v1/A_Mechanistic_Understanding_of_Alignment_Algorithms_A_Case_Study_on_DPO_and_Toxicity_llm.json"
    # code_path = "/data/lyj/ResearchKG/storage/raw/code/ajyl_dpo_toxic/ajyl_dpo_toxic_meta.json"
    # with open(file_path, "r", encoding="utf-8") as f:
    #     paper_dict = json.load(f)
    # paper = Paper.from_dict(paper_dict)
    # with open(code_path, "r", encoding="utf-8") as f:
    #     code_dict = json.load(f)
    # code = Code.from_dict(code_dict)
        
    # save_path =  config['kg_path']
    # coderag = CodeRAG()
    # db_path = "/data/lyj/ResearchKG/storage/raw/code/direct_preference_optimization/db.pkl"
    # coderag.prepare_retriever(code = code, db_path=db_path)
    # query = "Direct Preference Optimization (DPO): DPO is a novel, end-to-end procedure for training language models directly from human preference data without resorting to separate reward-model learning or reinforcement learning. It leverages a reparameterization of the latent reward function—rewriting the reward as β times the logarithm of the ratio between the model’s probability and a fixed reference probability—to derive an optimal policy in closed‐form. The method uses a simple binary cross-entropy loss computed over preference pairs (preferred vs. dispreferred completions) under the Bradley-Terry model. This approach simplifies the RLHF pipeline by directly optimizing the language model with standard supervised objectives."
    # context_text = coderag(query)
    # print(context_text)
    
    # query = "Reward Reparameterization Technique: This technique reparameterizes the latent reward function r(x, y) into the form r(x, y) = β log (π(y | x) / π_ref(y | x)). By expressing rewards in this way, any additive function of the prompt cancels out in pairwise preference comparisons (as in the Bradley-Terry model), and the corresponding optimal policy is obtained in closed form. It removes the need to learn an explicit reward model and allows the language model itself to implicitly represent rewards."
    # context_text = coderag(query)
    # print(context_text)
    
    # query = "Binary Cross-Entropy Preference Loss Computation: This technique directly defines a loss function over human preference pairs by leveraging the reparameterized reward. For a pair of completions (y_w, y_l) given prompt x, the method computes the loss as the negative log of the sigmoid function applied to β times the difference of the log-probability ratios (i.e. β log(π(y_w|x)/π_ref(y_w|x)) minus β log(π(y_l|x)/π_ref(y_l|x))). Minimizing this loss increases the probability of preferred completions and decreases that of dispreferred ones, thus aligning the policy with human preferences."
    # context_text = coderag(query)
    # print(context_text)
    
    
    # graph_constructor = GraphConstructor(model=config['model'])
    # graph_constructor.load_kg(get_config()['kg_path'])
    # graph_constructor.build_index()
    # # node = graph_constructor.generate_node(paper=paper, code=code, save_path=save_path)
    # technique1 = Technique(
    #     name="Toxicity Attribution Scoring",
    #     description="A technique that measures each token's contribution to the overall toxicity prediction by computing the difference in model outputs when masking individual tokens. The attribution score is calculated as s(t) = P(toxic|full_text) - P(toxic|text_without_t) for each token t, enabling fine-grained analysis of which parts of the text most influence toxicity predictions."
    # )
    # technique2 = Technique(
    #     name="Reference Model KL Minimization",
    #     description="This technique minimizes the KL divergence between policy and reference model distributions during DPO training. It adds a KL penalty term β * KL(π||π_ref) to the objective, preventing the policy from deviating too far from the reference model while optimizing for preferences, where β controls the strength of the KL constraint."
    # )
    # technique3 = Technique(
    #     name="Attention Pattern Analysis",
    #     description="A technique for analyzing internal model behavior by examining attention weight matrices across layers and heads. It quantifies how different tokens attend to each other, revealing hierarchical patterns in how the model processes information. The analysis involves computing attention statistics like entropy, sparsity, and cross-layer correlation to characterize the model's internal representations."
    # )
    # technique4 = Technique(
    #     name="Neuron Activation Clustering",
    #     description="This technique clusters neurons based on their activation patterns across a large corpus of inputs. By applying k-means clustering to neuron activation vectors, it identifies groups of neurons that consistently activate together, revealing functional modules within the model's hidden layers that may correspond to specific linguistic or semantic features."
    # )
    # technique5 = Technique(
    #     name="Progressive Token Masking",
    #     description="A technique for toxicity analysis that progressively masks tokens in order of their estimated toxicity contribution. It iteratively removes tokens and measures the change in predicted toxicity, generating a trajectory that reveals how the text's toxicity evolves as potentially problematic tokens are eliminated."
    # )
    # technique_list = [technique1, technique2, technique3, technique4, technique5]
    # for technique in technique_list:
    #     result = graph_constructor.retrieve_technique_by_technique(technique=technique, top_k=3)
    #     print(f"\n\n\nSimarity Results of {technique}:")
    #     for res in result:
    #         print(f"Technique: {res[0].name}, {res[0].description} \nSimilarity: {res[1]}\n\n")
    
    
    
if __name__ == "__main__":
    main()
