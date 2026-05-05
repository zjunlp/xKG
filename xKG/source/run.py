import argparse
from .utils import *
from .llm import *
from .tools import *
from .schema import *
from .interface import *
initialize_kg()
# res = find_node_by_paper_title("Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization")
# res = find_similar_techniques(name="Toxic Vector Extraction", top_k=1)
des = """
While alignment algorithms are commonly used to tune pre-trained language models towards user preferences, we lack explanations for the underlying mechanisms in which models become \"aligned\", thus making it difficult to explain phenomena like jailbreaks. In this work we study a popular algorithm, direct preference optimization (DPO), and the mechanisms by which it reduces toxicity. Namely, we first study how toxicity is represented and elicited in pre-trained language models (GPT2-medium, Llama2-7b). We then apply DPO with a carefully crafted pairwise dataset to reduce toxicity. We examine how the resulting models avert toxic outputs, and find that capabilities learned from pre-training are not removed, but rather bypassed. We use this insight to demonstrate a simple method to un-align the models, reverting them back to their toxic behavior.
"""
# res = find_similar_techniques("PPLM-based Toxicity Controlled Generation", des, top_k=3, llm_rerank=True)
# for technique in res:
#     print(technique.name)
#     print(technique.description)
#     print("\n")

res = decompose_and_find_techniques(description=des, top_k=5, llm_rerank=True)
for technique in res:
    print("Here is the retrieved techniques:")
    print(technique.name)
    print(technique.description)
    print("\n")