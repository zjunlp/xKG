You should download ImageNet-1K using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```

- The implementation details of PTQ4ViT for model quantization are partially missing from the main text. We refer to the original PTQ4ViT paper for the complete quantization process details.

- The CMA-ES used in the paper was a Python library, for which an implementation is in https://github.com/CyberAgentAILab/cmaes

Baselines should be imported from the links provided in the paper.

These experiments are not required for replication:
- Run-Time Memory Usage (in Section 4.4)
- Computational Complexity Analyses (in Section 4.4)
- Comparison w.r.t. In-Distribution Performance (in Section 4.4)

For memory usage measurements reported in the paper, these represent both the runtime and peak GPU memory usage during processing. Higher GPU memory usage is observed when one uses commands, such as nvidia-smi, to check memory usage. This is because some GPU memory, unused but cached by pytorch's allocator for acceleration, is included in the memory usage. Please refer to here if you are interested in more information.

To accurately estimate the GPU memory usage, use the following code from pytorch (or equivalent):
```python
torch.cuda.reset_peak_memory_stats()
output = model(images)
print(f'memory usage: {torch.cuda.max_memory_allocated()/(1024*1024):.3f}MB')
```

- The arrangement of input sequence elements is [CLS token, learnable prompts, patch embeddings] in that specific order.

- Table 7 mentions "FOA-I V1/V2" which refer to two different implementation strategies for interval-based FOA: V1 stores features between updates while V2 stores images between updates.

- For the activation shifting moving average (Equation 9), $\mÎ¼_N(0)$ should be initialized using the statistics of the first batch $\mu_N(X_1)$.