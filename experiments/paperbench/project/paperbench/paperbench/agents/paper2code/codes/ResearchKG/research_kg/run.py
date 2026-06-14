import argparse
from utils import *
from llm import *
from tools import *
from schema import *
from interface import *
initialize_kg()
# res = find_node_by_paper_title("Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization")
# res = find_similar_techniques(name="Toxic Vector Extraction", top_k=1)
res = find_similar_papers("Adapt in the Wild: Test-Time Entropy Minimization with Sharpness and Feature Regularization", top_k=2)
print(res[0][0].paper_title)
print(res[0][1])

print(res[1][0].paper_title)
print(res[1][1])
