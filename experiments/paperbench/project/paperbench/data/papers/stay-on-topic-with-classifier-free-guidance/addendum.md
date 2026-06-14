## Useful details

Unless otherwise specified, models are sampled with the default hyperparameters in Eleuther's LLM evaluation harness. https://github.com/EleutherAI/lm-evaluation-harness

In Section 3.2 the paper states "instead of upweighting just $w_{p}$, we might upweight $w_{p}, w_{cot}$". Here $w_{p}$ is the prompt and so is used as $c$ in the equation for CFG.

Note that Figure 2's caption labels 'top' and 'bottom' figures, but the figures are actually positioned left and right.

In Section 3.3 the paper states "As shown in Table 3, the number of tasks where CFG outperforms...". This should be Figure 3, not Table 3.

In the experiments in Section 4.1 FLOPs are measured using this formula: https://github.com/google-research/electra/blob/master/flops_computation.py

At the start of Section 5 the paper states "we use the Falcon-7b-Base model (Almazrouei et al. 2023) and, when applicable, compare against the Falcon-7b-Instruct version. We run these models on a sample dataset of 32,902 datapoints from P3 (Sanh et al., 2021). We replicate our findings on the Open-Assistant Dataset (Köpf et al., 2023) and Redpajama-3b model family". To clarify: all of the reported results in the paper are from Falcon and P3. You may ignore the combinations of Falcon with Open-Assistant, and Redpajama with both P3 and Open-Assistant.

As for _how_ the 32,902 datapoints are sampled from P3: ~50 samples are randomly sampled from each of the 660 datasets in P3. Some datasets have less than 50 samples in which case you just take the entire dataset. When sampling, the authors filtered out samples that had an input length longer than 200 tokens, but this is just an optimization for inference and should not have an impact on the results.

In Section 5.1, the definition of entropy used is $H(p)=-\sum_k p_k \log p_k$ and the implementation used was from https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html

In Section 5.1 the mean entropy is averaged over each token. i.e. if $H\left(p\left(x_i \mid x_{<i}\right)\right)$ is the entropy of the vocabulary distribution produced by the LLM, $p\left(x_i \mid x_{<i}\right)$, then the "mean entropy" is $\frac{1}{n} \sum_{i=1}^n H\left(p\left(x_i \mid x_{<i}\right)\right)$ over $n$ completion tokens.

In Section 5.1 the paper states "This restricts the number of tokens in the top-p $=90 \%$ of the vocabulary distribution". To clarify, a lower entropy means that more probability mass is on fewer tokens. In the context of sampling the next token, lower entropy means one samples from fewer possible tokens because fewer tokens have probability mass. The mention of "top-p" in the paper is somewhat irrelevant; one would see the # of tokens be smaller if one considers sampling from the full $p\left(x_i \mid x_{<i}\right)$ or the top 90% of that distribution.

In Section 5.2 the paper states "There are cases where the two are similar. As shown in Table 5". This should be Figure 5, not Table 5.

In Section 5.2 perplexity is computed on the continuation, and so ignores ang loglikelihood on the prompt.

In Section 5.3 the paper states "In Figure 3, we prompt a model...". This should be Table 3, not Figure 3.

In Appendix C.2 the paper states that the ANCOVA results are in Table 9. This is an error, it should be Table 6.

### Dataset links

Below we list official links to the datasets used in the paper:

- [ARC-c](https://huggingface.co/datasets/allenai/ai2_arc)
- [ARC-e](https://huggingface.co/datasets/allenai/ai2_arc)
- [BoolQ](https://huggingface.co/datasets/google/boolq)
- [HellaSwag](https://rowanzellers.com/hellaswag/)
- [PiQA](https://github.com/ybisk/ybisk.github.io/tree/master/piqa)
- [SciQ](https://huggingface.co/datasets/allenai/sciq)
- [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)
- [WinoGrande](https://huggingface.co/datasets/allenai/winogrande)
- [LAMBADA](https://huggingface.co/datasets/EleutherAI/lambada_openai)
- [GSM8k](https://huggingface.co/datasets/openai/gsm8k)
- [AqUA](https://huggingface.co/datasets/nguyen-brat/aqua)
- [HumanEval](https://github.com/openai/human-eval)
- [P3](https://huggingface.co/datasets/bigscience/P3)

- For accessing the Jigsaw toxic comment classification dataset, reproductions should use <https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge> rather than the Kaggle hosted dataset.

## Clarifications on reproduction scope

Evaluating on LLaMA models is out of scope for reproduction as acquiring access to these models requires permission.

The experiments described in Appendix D.1 with GPT-J and CodeGen which are mentioned at the start of Section 3.3.1 are out of scope for reproduction and may be ignored.

All of Section 3.4 is out of scope for reproduction as it requires a human preference study.

The VRAM and memory usage analysis mentioned in Section 4 is out of scope for reproduction because it is analytical work that doesn't involve any new empirical work.

The results specific to the FLOPs experiment in Section 4.1, on the other hand, are IN SCOPE for the reproduction.

The experiments in Appendix D.1 that are mentioned at the start of Section 3.3.1 are out of scope for reproduction.

Figures 6,7,9,11,12,13 do not need to be reproduced as part of this reproduction.

The results of Table 5 are IN SCOPE for the reproduction.

The results of Table 3 are IN SCOPE for the reproduction.

The results of Table 1 are NOT IN SCOPE for the reproduction.
