For half-precision attacks, 16-bit ints needs to be used, and for single-precision attacks, 32-bit ints need to be used. This is not explicitly mentioned in the paper.

The LLaVA model used by the paper is taken from [this repository](https://github.com/haotian-liu/LLaVA/tree/main). The implementations for the POPE benchmark and the SQA-I benchmarks are also taken from the repository. There are some differences in the LLaVA model used in the paper. LLaVA-1.5 7B is set up to use the OpenAI CLIP ViT-L/14@224 vision encoder (rather than the default ViT-L/14@336). Additionally, the code has been modified as needed to allow LLaVA to work with OpenCLIP CLIP implementation instead of the Huggingface implementation

The OpenFlamingo model used by the paper is taken from [this repository](https://github.com/mlfoundations/open_flamingo/tree/main).

The APGD algorithm is taken from [here](https://github.com/fra31/robust-finetuning).

The jailbreaking implementation is taken from The codebase for the paper from Qi et al. (2023)](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/llava_llama_2_utils/visual_attacker.py). Note, they implement it for minigpt and LLaVA-LLaMA-2 (13B), so this needs to be adapted to work on LlaVA-1.5 (7B). For the attacks, 5000 iterations and alpha = 1/255 is used, as Qi et al. In the jailbreaking attacks, [a single image](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/adversarial_images/clean.jpeg) is used for the attacks. The PGD in the attacks doesn't use momentum. The adversarial attacks are universal targeted attacks based on [these harmful target strings](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/harmful_corpus/derogatory_corpus.csv). [These harmful prompts](https://github.com/Unispac/Visual-Adversarial-Examples-Jailbreak-Large-Language-Models/blob/main/harmful_corpus/manual_harmful_instructions.csv) are used for evaluation. Harmfulness is determined by a human. Criterium is that the output actually contains something harmful. A response that is affirmative but does not actually contain harmful output is not considered harmful.

For visual-question answering, low-precision attacks are performed on the top 5 frequent ground truth, and then a high-precision attack is done on the ground truth that led to the lowest score for each sample. Then, we do a targeted attack on the most frequent ground truth with "maybe" on it, using a clean perturbation initialization. Then, we do a targeted attack attack with "Word" on it, with another clean perturbation initalization -- note: the attack with "Word" is not done on TextVQA. It's important that the target "maybe" is not capitalized, but "Word" is.

Details about prompts for LLaVA and OpenFlamingo — when grading, the rubric will check for implementations of things asked for from the code below.

The images for the sixth caption is taken from stock photo sites that have permissive licenses. Details can be found in the PR for the Robust CLIP rubric. The reproducer is not required to reproduce these.

The PGD implementation includes: gradient normalization with elementwise sign for l_infinity, momentum factor of 0.9, initialization with uniform random perturbation, and computation of l_infinity ball around non-normalized inputs

For computation of the CIDEr scores, they compute the CIDEr scores after every attack, so that they can take the worst case score for each sample, and remember the best ground-truth and perturbation for the single-precision attack.

You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```
