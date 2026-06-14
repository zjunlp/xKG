# Additional information

The definition of information content stated in "Hierarchical classification at multiple operating points" should be used when implementating information content:

> Two popular choices are the depth of the node d(y) and its information content I(y) = - log p(y) = log |L| - log |L(y)| (assuming a uniform distribution over the leaf nodes).

All vision-only models should be accessed via the [torchvision](https://github.com/pytorch/vision) module.

All vision-language models should be accessed via the [OpenCLIP](https://github.com/mlfoundations/open_clip) and CLIP (https://github.com/openai/CLIP) modules.

## ImageNet datasets

You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```

The other in- and out-of-distribution ImageNet datasets were downloaded from:

- ImageNet-v2: https://imagenetv2.org/
- ImageNet-S: https://huggingface.co/datasets/songweig/imagenet_sketch
- ImageNet-R: https://github.com/hendrycks/imagenet-r
- ImageNet-A: https://github.com/hendrycks/natural-adv-examples
- ObjectNet: https://objectnet.dev/

Note:

- ImageNet-v2 contains three variants; only the `MatchedFrequency` split was used in the paper. The paper retrieves the `MatchedFrequency` split from commit d626240 of https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main.

## WordNet dataset

The WordNet dataset was downloaded from https://github.com/jvlmdr/hiercls/blob/main/resources/hierarchy/imagenet_fiveai.csv.

## Reproducing Table 3

The data from Section 4.1 is used to fit a linear regression line to see how well in-distribution LCA can predict out-of-distribution performance.

## Data features

The data features $M(X)$ are taken from the last hidden layer before the linear classifier (FC layer) is applied.

## Calculating the LCA distance from $k$-means clusters

Regarding the clustering process outlined in Appendix E.1, the cluster level at which a pair of classes first share a cluster is the pairwise LCA height.

## Aline-S and Aline-D implementations

The implementations of Aline-S and Aline-D were copied from https://github.com/kebaek/Agreement-on-the-line/blob/main/agreement_trajectory.ipynb.

## Constructing the LCA distance matrix using latent hierarchies

In Appendix E.2, the $n \times n$ LCA distance matrix $M$ is defined as $M_{\mathrm{LCA}}=\operatorname{MinMax}\left(M^T\right)$. However, when constructing the matrix using latent hierarchies, the matrix is inverted, like so:

```python
def process_lca_matrix(lca_matrix_raw,tree_prefix,temperature=1.0):
    if lca_matrix_raw is None:
        return None
    if tree_prefix!='WordNet':
        result_matrix=np.max(lca_matrix_raw)-lca_matrix_raw  # The LCA distance matrix is inverted when using latent hierarchies
    else:
        result_matrix=lca_matrix_raw
    result_matrix=result_matrix**temperature

    scaler = MinMaxScaler()
    result_matrix=scaler.fit_transform(result_matrix)
    print(result_matrix)
    return torch.from_numpy(result_matrix)
```

## Sanity checking the LCA distance matrix

When computing the LCA distance matrix, keep in mind that it should store the pair-wise distance between classes -- _not_ similarity. This means that the LCA distance matrix should have a diagonal of zeros, since the distance between a class and itself is zero. It follows that the inverted LCA distance matrix, as described in Line 2 of Algorithm 1, should have a diagonal of ones. It's highly recommended that you write sanity checks to check that both the LCA and inverted LCA distance matrices have these properties.
