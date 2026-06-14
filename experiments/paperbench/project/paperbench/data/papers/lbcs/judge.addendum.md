## Useful details for Judge

### Bilevel Coreset Selection -- Generic Algorithm

The following is pseudocode describing the generic algorithm for bilevel coreset
selection

```pseudocode
Given a dataset D = {x_1, x_2, ..., x_n}
and a model M(w) with parameters w
Define an outer objective, e.g. f_1 (performance) + f_2 (coreset size)

Initialize coreset C, or coreset-mask parameters (like s_i)

For t in 1, 2, ... T (outer loop iterations):
    # Inner loop: train model on the current coreset
    for i in 1, 2, ... I (inner loop iterations):
        train M on C (or on samples drawn from C)

    # Outer step: update coreset based on performance
    # e.g. equation (3) or algorithm 2 or other methods
    C = update_coreset(C, M)  # Minimizing the outer objective

# After T iterations, C is the final refined coreset
# Optionally, re-train M on C or use M directly
```

### CNN architecture from Zhou et al, 2022

The CNN architecture used for the experiment in Figure 1 and in section 5.1 is
the `ConvNet` class from Zhou et al, 2022, implemented as such:

```python
class ConvNet(nn.Module):
    def __init__(self, output_dim, maxpool=True, base_hid=32):
        super(ConvNet, self).__init__()
        self.base_hid = base_hid
        self.conv1 = nn.Conv2d(1, base_hid, 5, 1)
        self.dp1 = torch.nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(base_hid, base_hid*2, 5, 1)
        self.dp2 = torch.nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 4 * base_hid*2, base_hid*4)
        self.dp3 = torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(base_hid*4, output_dim)
        self.maxpool = maxpool

    def forward(self, x, return_feat=False):
        x = self.embed(x)
        out = self.fc2(x)
        if return_feat:
            return out, x.detach()
        return out

    def embed(self, x):
        x = F.relu(self.dp1(self.conv1(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.dp2(self.conv2(x)))
        if self.maxpool:
            x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 2*self.base_hid)
        x = F.relu(self.dp3(self.fc1(x)))
        return x
```

### Process for forming class-imbalanced F-MNIST (figure 2b, section 5.3)

The authors leveraged a class-imbalanced sampler, adjusting the code from
[this file](https://github.com/YyzHarry/imbalanced-semi-self/blob/master/dataset/imbalance_cifar.py)
to work with F-MNIST, which has the same number of classes as CIFAR10. For
reference, this is the code snippet that was adjusted:

```python
class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None, download=False):
        super(ImbalanceCIFAR10, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(img_num_list)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
```

### More details on the baselines

#### Uniform sampling

- Decide on a target subset size $k$.
- Sample $k$ points uniformly at random from the full dataset (without
  replacement).
- The resulting coreset is simply those selected points.

#### EL2N

- Train a “proxy” model (e.g., a small CNN) on the entire dataset for some
  epochs.
- After or during training, compute for each sample $(x, y)$:
  - The predicted probability vector $p(x)$.
  - The “error vector” $e = p(x) - \text{one\_hot\_label}(y)$.
  - The EL2N score $= \| e \|_2$ (the L2-norm of that error vector).
- Rank all samples by their EL2N score, typically descending (largest scores =
  “hardest”).
- Select the top $k$ samples as the coreset.

#### GraNd

- Train a “proxy” model on the entire dataset for some epochs.
- For each sample $(x, y)$, compute the gradient of the loss w.r.t. the model
  parameters: $\nabla_{\theta} \,\ell(h(x;\,\theta),\,y).$
- Let the GraNd score $= \|\nabla_{\theta}\,\ell\|_2,$ i.e., the L2-norm of that
  gradient.
  - Often you can average these gradient norms over a few training steps or
    epochs.
- Rank all samples by their GraNd score (descending).
- Select the top $k$ samples to form the coreset.

#### Influential coreset

- Train a “proxy” model on the entire dataset to convergence, obtaining
  parameters $\hat{\theta}$.
- Compute the Hessian
  $$
    H_{\hat{\theta}}
    \;\approx\;
    \frac{1}{n}\,\sum_{(x_i,y_i)\,\in\,D}\,\nabla_{\theta}^2\,\ell\bigl(h(x_i;\theta),\,y_i\bigr)\biggr\rvert_{\theta=\hat{\theta}},
  $$
  often approximated or diagonalized for efficiency.
- For each training sample $z$, compute its parameter‐influence vector
  $I_{\text{param}}(z)$ according to Eq. (2) of (Yang et al., 2023):
  $$
    I_{\text{param}}(z)
    \;=\;
    -\,H_{\hat{\theta}}^{-1}\,\nabla_{\theta}\,\ell\bigl(h(z;\theta),\,y_z\bigr)\biggr\rvert_{\theta=\hat{\theta}}.
  $$
  This approximates how removing or reweighting $z$ shifts the final trained
  parameters.
- **Formulate and solve the selection problem** (an optimization over a 0–1
  indicator vector $W$) as in Eq. (4) of (Yang et al., 2023). For instance, the
  “cardinality‐guaranteed pruning” version is:
  $$
    \min_{W \in \{0,1\}^n}\,\bigl\|W^\top S\bigr\|_2
    \quad
    \text{subject to}
    \quad
    \sum_{i=1}^n W_i = m,
  $$
  where $S$ is the collection of all influence vectors $I_{\text{param}}(z)$,
  and $W_i = 1$ means sample $i$ is selected into the coreset.
- Once $W$ is solved, select the corresponding subset from $D$ to form the
  coreset.
- Finally, train the _desired_ (final) model on this chosen coreset.

#### Moderate coreset

- Train a “proxy” model on the entire dataset for some epochs to extract
  embeddings.
- For each sample $(x, y)$, obtain its embedding $\phi(x).$
- For each class $c,$ compute the class centroid $\mu_c$ as the average
  embedding of all samples of class $c.$
- For each sample, compute the distance $d = \|\phi(x) - \mu_{y}\|_2,$ i.e.,
  distance to its class centroid.
- Sort samples within each class by distance and pick those near the median or
  “moderate” region (i.e., not the very smallest distances, not the largest).
  - If a fixed size $k$ is required, use per-class quotas or another scheme to
    get exactly $k$ total.
- The selected subset is the coreset.

#### CCS (Coverage-Centric Coreset Selection)

- Either train a “proxy” model or use a fixed feature extractor to get
  embeddings $\phi(x)$ for each sample $(x, y).$
- Define a coverage-based objective, e.g., $k$-center or facility-location
  style:
  - You want to pick a subset such that all points in the full dataset are
    “close” to at least one coreset point.
- Solve that coverage objective (e.g., via a greedy selection) to get a subset
  of size $k.$
  - Some implementations also account for class balance or constraints in the
    same procedure.
- The resulting subset is the coreset.

#### Probabilistic coreset

- Initialize a probability vector $s,$ where each $s_i \in [0, 1]$ encodes how
  likely it is to include sample $i.$
- **Inner loop**: sample a binary mask $m$ from the Bernoulli($s$) distribution,
  train a model on just those selected points, measure performance on the full
  dataset or a validation set.
- **Outer loop**: update $s$ with gradient-based or policy-gradient methods to
  improve performance while aiming for a smaller expected coreset size
  ($\sum_i s_i$).
- Repeat the inner+outer process until convergence.
- Finally, produce a definite coreset by thresholding $s$ (e.g., pick $i$ where
  $s_i > 0.5$) or by taking the top $k$ $s_i.$
