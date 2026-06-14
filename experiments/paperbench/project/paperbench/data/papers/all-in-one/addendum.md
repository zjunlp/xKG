# Training
The sbi python library must be used to implement NPE, NRE, and NLE. Default parameters from the library are used, aside from using a more expressive neural spine flow for NPE and NLE.

C2ST is implemented using a random forest classifier with 100 trees.

In Section 4.2, infer the posterior distributions on a uniform grid between t=0 and t=15.

Diffusion time is embedded as a random Gaussian Fourier embedding, and a linear projection is added to the output of each feed-forward block in the transformer

During training, for each element in a batch, the condition mask $M_C$ is sampled uniformly at random from the following options:
- joint mask (all False)
- posterior mask (all "parameter" variables are False, all "data" variables are True)
- likelihood mask (all "data" variables are False, all "parameter" variables are True)
- rand_mask1 ~ Ber(0.3) (resampled for every element)
- rand_mask2 ~ Ber(0.7) (resampled for every element)

## Graph Inversion
Algorithm Graph Inversion by Webb et al. 2018:
1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
3. Set all vertices of $J$ to be unmarked.
4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
5. $S \leftarrow$ all latent variables without latent parent in $G$.
6. while $S \neq \emptyset$ do
7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
8.   Add edges in $J$ between unmarked neighbours of $v$.
9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
10.   Mark $v$ and remove from $S$.
11.   for unmarked child latents $u$ of $v$ in $G$ do
12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
13.   end for
14. end while
15. return $H$.

To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

## Tasks
The summary statistics computed for the Hodgkin-Huxley task are:
1. Spike count
2. Mean of the resting potential
3. Standard deviation of the resting potential
4. Mean potential in the spiking domain
5. Second central moment (variance) of the voltage during the spiking domain
6. Third central moment of the voltage during the spiking domain
7. Fourth central moment of the voltage during the spiking domain

In the Hodgkin-Huxley task, the energy consumption is computed based on sodium charge using the following formula:
```python
def convert_charge_to_energy(E):
    E = E # Energy is negative
    E = jnp.diff(E) # Non cumulative energy
    E = jnp.convolve(E, 1/5*jnp.ones(5), mode="same") # Smooth due to numerical differentiation error
    return convert_total_energy(E)

def convert_total_energy(E):
    E = -E # Energy is negative
    E = E/1000 # mS to S
    E = E / 1000 # mV to V
    E = E * 0.628e-3 # Area of the membrane
    e = 1.602176634e-19 # Elementary charge
    N_Na = E / e # Number of elementary charges
    valence_Na = 1 # Valence of sodium
    number_of_transports = 3 # Number of Na out per ATP 
    ATP_Na = N_Na / (valence_Na * number_of_transports) # Number of ATP
    ATP_energy = 10e-19 # Energy by ATP hydrolysis
    E = ATP_Na * ATP_energy # Energy in Joules
    E = E / 0.2 # Energy in J/s
    return E * 1e+6 # Energy in uJ/s
```

## Task Dependencies
For each task considered the attention mask representing a directed graphical model for the joint distributions is given. The undirected mask is obtained by making it undirected. Assuming the variables are in order $\theta_1, \theta_2,…, x_1, x_2,…$, these can be constructed as follows for the directed case:

```python
import numpy as np
from scipy.linalg import block_diag
# Gaussian Linear Task: Represents that data depends on parameters but is factorized across dimensions.
M_θθ = np.eye(10)
M_xx = np.eye(10)
zeros = np.zeros((10, 10))
M_θx = np.eye(10)
M_E_gaussian = np.block([[M_θθ, zeros], [M_θx, M_xx]])
# Two moons and Gaussian Mixture: Represents that each data variable depends on all parameters and the other data variables
M_θθ = np.eye(2)
M_xx = np.tril(np.ones((10, 10)))
zeros = np.zeros((2, 10))
M_θx = np.ones((10, 2))
M_E_two_moons = np.block([[M_θθ, zeros], [M_θx, M_xx]])
# SLCP: Dense parameter-data dependence
M_θθ = np.eye(4)
M_xx = block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])
zeros = np.zeros((4, 8))
M_θx = np.ones((8, 4))
M_E_slcp = np.block([[M_θθ, zeros], [M_θx, M_xx]])
# Tree Structure: Diagonal is always true, follows tree dependencies
M_E_tree = np.eye(10)
M_E_tree[0, 1:3] = True  # θ₁, θ₂ depend on θ₀
M_E_tree[1, 3:5] = True  # x₁, x₂ depend on θ₁
M_E_tree[2, 5:7] = True  # x₃, x₄ depend on θ₂
# HMM (Hidden Markov Model): Markov chain for parameters and factorized data
M_θθ = np.eye(10) + np.diag(np.ones(9), k=-1)
M_xx = np.eye(10)
zeros = np.zeros((10, 10))
M_θx = np.eye(10)
M_E_hmm = np.block([[M_θθ, zeros], [M_θx, M_xx]])
```
The mask for the Lotka Volterra task is metadata dependent, i.e., it depends on which T time points the predator or prey variables are observed. The prior hence M_θθ = I, the first two parameters are for the prey population and the last two for the predator M_θx = np.array([[1]*T + [0]*T, [1]*T + [0]*T, [0]*T + [1]*T, [0]*T + [1]*T]). The simulation is Markovian so M_x1x1 = M_x2x2 = np.eye(T) + np.diag(np.ones(T-1), k=-1). For the cross-data dependence a causal dependency is used, i.e., each prey variable depends additionally on all the past predator variables.

# Tokenization
To correctly compute the embedding for values, the embedding for a single value is created by repeating the scalar value to match the desired dimensionality. For example, to embed the value 1 to desired dimensionality N, we would have a vector [1, 1, ...., 1] of length N.

To embed the condition mask, the tokenizer uses learnable vector embeddings to embed each value - "True" values are projected using the learnable vector embeddings (of whatever dimensionality), whereas "False" values are projected to zeros (of the desired dimensionality).

For each input, the tokenizer concatenates the embeddings of the identifier, value, metadata (if used), and condition state in that order

# Experiments

In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.

In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.

In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.
