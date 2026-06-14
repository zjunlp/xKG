## Clarifications from author

- The binary model for extracting the probe vector W_toxic is defined as 'W_toxic \mathbf{x}', which means that W_toxic is actually a matrix. In fact, in the probe model, W_toxic is a matrix of shape [d_model, 2], where W_toxic[:, 0] is for non-toxic and W_toxic[:, 1] is for toxic. When the authors say "cosine similarity with all value vectors with highest cosine similarity with W_toxic ", they mean with W_toxic[:, 1]

- In table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector

- In table 1, on the MLP value vectors, superscript refers to layer number and subscript to index number in the parameter matrix.

- In Section 3.1, when the authors describe decomposing toxic vectors with SVD, they mention performing SVD on an $N \times d$ matrix. This is a mistake, and in fact what the authors meant is performing SVD on the _transpose_ of the $N \times d$ matrix, i.e. on a $d \times N$ matrix, hence obtaining $d$-dimensional singular value vectors from the $U$ matrix.

- When training DPO, the training data was split in 90:10, and approximately 6k toxic/non-toxic pairs were needed for convergence.

- In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.

- Clarification on what "Idx" means in Figure 2: For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector -- ie, "Idx" refers to the index of a value vector in the MLP weights.

- In Figure 5, the plots are each measuring the cosine similarity between the value vectors in a particular layer vs. the delta of layer 19.

- Figure 4 is a PCA plot, showing the mean element-wise difference of vectors. Section 5.2 (page 6, left column) has more details. The colors indicate how strongly the residual stream activates a toxic value vector. The shapes indicate whether the residual stream is taken from GPT2 or GPT2_DPO.


## Scope of reproduction

- Results related to Llama2 have been deemed out of scope because access to get the llama2 model depends on getting permission. 
- The creation of the 24,576 pairs of toxic and nontoxic continuations is WITHIN the scope of reproduction. A reproduction of this dataset is necessary to fully reproduce the paper.

## Useful details

- For measuring toxicity, reproductions should use https://huggingface.co/unitary/unbiased-toxic-roberta instead of Perspective API.
- For accessing the Jigsaw toxic comment classification dataset, reproductions should use https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge rather than the Kaggle hosted dataset.
