## Out of scope for reproduction

- The experiments (and their results) from Appendix E.5 and E.6 do not need to
  be reproduced. They are out of scope.
- Experiments involving the ImageNet-1k dataset (so, all of section 5.4) do not
  need to be reproduced. They are out of scope

## Useful details

### General useful details

- Submissions should NOT rely on Kaggle for accessing any of the datasets. They
  should use other routes (e.g.
  [PyTorch](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)) that
  do not rely on API keys or credentials.

### Useful details for Figure 1

- In the experiment for Figure 1, a value of lambda of 0.5 is used for equation
  (4).
- In the experiment for Figure 1, an arbitrarily random subset of MNIST is used.
  You may for instance use MNIST-S.
- For the CNN used, the authors use the `ConvNet` class from Zhou et al, 2022.
  (https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection/blob/master/models.py)
- In the experiment for Figure 1, when running bilevel coreset selection, 1000
  iterations of the outer loop are run (i.e. hparam T = 1000).

### Useful details for Section 5.1

- The CNN described is exactly the same CNN as the one used in Figure 1, i.e.
  the `ConvNet` class from Zhou et al, 2022.
  (https://github.com/x-zho14/Probabilistic-Bilevel-Coreset-Selection/blob/master/models.py)

### Useful details for Section 5.2

- The authors say "For fair comparisons, we reproduce the baselines based on
  their code repositories." -- This means that the authors reimplemented the
  baselines themselves, by reading the code of the respective papers.
  Reproductions should follow this same procedure.
- The ResNet18 is trained using an SGD optimizer with a learning rate of 0.1,
  momentum of 0.9, and a cosine scheduler.

### Useful details for Section 5.3

- In section 5.3, the same proxy and target models are used as in section 5.2
  for F-MNIST, i.e. a LeNet for both the proxy and target model.
- For creating the class-imbalanced version of F-MNIST, the authors leveraged a
  class-imbalanced sampler, adjusting the code from
  [this file](https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py)
  to work with F-MNIST.
  - Note that the imbalance is just injected into the training set, which does
    not include the test set.

### Useful details for section 5.4

In section 5.4, the author say "Partial results are from previous work". These
are the results for Uniform, EL2N, GraNd, Influential and Moderate. The authors
implemented and produced the results for CCS and Probabilistic themselves,
referring to
[this code base](https://github.com/rgeirhos/dataset-pruning-metrics).
