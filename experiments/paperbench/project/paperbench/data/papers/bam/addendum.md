## Useful additional details and clarifications

- In sections 5.2 and 5.3, like in 5.1, a grid search was used to determine the
  best learning rate for the gradient-based methods.
- In section 5.1, the paper writes "In Appendix E.2, we present wallclock
  timings for the methods, which show that the gradient evaluations dominate the
  computational cost in lower-dimensional settings." The correct statement
  should say "higher-dimensional" settings, not "lower-dimensional".
- For the experiments relevant for Figure E.1, the batch size was set to 4 for
  all methods (with the exception of $D=4$, where it was set to 3 in order to
  run the low-rank BaM solver that requires $B < D$).
- For computing the gradient of the log density functions for the PosteriorDB
  models, the authors used the bridgestan library
  (https://roualdes.github.io/bridgestan/latest/)

### Additional Details on the VAE neural network

- **Optimizer**: Adam
- **Learning Rate**:
  - **Initial Value**: 0.0
  - **Peak Value**: 1e-4
  - **Warmup Steps**: 100
  - **Warmup function**: linear
  - **Decay Steps**: 500 (number of training batches)
  - **End Value**: 1e-5
- **Activation Functions**:
  - GELU in all hidden layers
  - tanh in final decoder layer
- **Pooling / Dropout / Normalization**:
  - No explicit pooling (downsampling via stride=2 conv)
  - No dropout
  - No batch/layer normalization
- **Encoder Architecture**:
  1. `Conv (in_channels=3, out_channels=c_hid, kernel_size=3, stride=2) -> 16×16`
  2. `Conv (in_channels=c_hid, out_channels=c_hid, kernel_size=3, stride=1) -> 16×16`
  3. `Conv (in_channels=c_hid, out_channels=2×c_hid, kernel_size=3, stride=2) -> 8×8`
  4. `Conv (in_channels=2×c_hid, out_channels=2×c_hid, kernel_size=3, stride=1) -> 8×8`
  5. `Conv (in_channels=2×c_hid, out_channels=2×c_hid, kernel_size=3, stride=2) -> 4×4`
  6. `Flatten → Dense (output = latent_dim)`
- **Decoder Architecture**:
  1. `Dense → reshape to [batch, 4, 4, 2×c_hid]`
  2. `ConvTranspose (in_channels=2×c_hid, out_channels=2×c_hid, kernel=3, stride=2) -> 8×8`
  3. `Conv (in_channels=2×c_hid, out_channels=2×c_hid, kernel=3, stride=1) -> 8×8`
  4. `ConvTranspose (in_channels=2×c_hid, out_channels=c_hid, kernel=3, stride=2) -> 16×16`
  5. `Conv (in_channels=c_hid, out_channels=c_hid, kernel=3, stride=1) -> 16×16`
  6. `ConvTranspose (in_channels=c_hid, out_channels=3, kernel=3, stride=2) -> 32×32`
  7. `Final activation is tanh → outputs in [-1, 1]`
- **Other Key Details**:
  - Latent dimension (`latent_dim`) is set to 256.
  - Negative ELBO is estimated with 1 Monte Carlo sample (`mc_sim=1`) by
    default.

## Out of scope for reproduction

- For Section 5.1, for the Gaussian targets:
  - you do not need to run the experiment where the authors "evaluated BaM with
    a number of different schedules for the learning rates" as they describe in
    Appendix E.3. It is sufficient to run the experiment with the learning rate
    reported in the main text, i.e. $\lambda=BD$.
  - As such, Figure E.2 is out of scope for this reproduction and may be
    ignored.
  - The grid searches performed for gradient-based methods are however in scope.
- For section 5.1, for the non-Gaussian targets:
  - You do not need to run the experiments exploring different learning rates
    for BaM, as described in Appendix E.4. It is sufficient to run the
    experiment with the learning rate reported in the main text, i.e.
    $\lambda_t=BD/(t+1)$.
  - As such, Figure E.5 is out of scope for this reproduction and may be
    ignored.
  - the grid searches for the gradient-based methods are in scope.
- Because "wallclock timings [...] are heavily dependent on implementation and
  JIT-compilation details and hardware", results dependent on wallclock timings
  are out of scope for this reproduction. This includes, for example, figure
  E.1.
- The results of Figure E.6 are in scope for the reproduction.
