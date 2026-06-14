## Out of scope for reproduction

- The results implied by Figure 3, discussed in section 5.1, are out of scope
  for reproduction since the discussion is qualitative and therefore cannot be
  straightforwardly judged.

## Additional Useful Information

### Miscellaneous details

- Figure 5 shows the evaluation tasks on antmaze.
- Figure 5, the returns are normalized by dividing by the maximum return that
  any agent scores on that task set. Thus there are four columns that have a
  normalized return of 1 (one for each task set).
- Table 1 calculates uncertainty as the standard deviation over 5 seeds (with 20
  rollouts each, averaged).
- For the ExORL environments, the training is done on the RND dataset for each
  domain.

### Some notes on the evaluation environments

- The D4RL repo has been updated since the release of the paper. For
  reproducibility, it may be useful to use commits of the D4RL repo from before
  the paper publication date, i.e. before June 2024

### Additional Details on the FRE architecture

- the scalar reward is discretized into 32 bins by rescaling the reward to [0,
  1] and then multiplying by 32 and flooring to the nearest integer
- The discretized reward is mapped to a continuous vector representation using a
  learned embedding table.
- the environment state projected into an embedding using a learned linear
  transformation
- for each state, the reward embedding is concatenated to end of the state
  embedding before the set of reward-labeled states is passed through the
  encoder
- The Appendix lists the "Reward Embedding Dim" as 128. This in incorrect. In
  reality the state embedding is 64-dimensional and the reward embedding is
  64-dimensional, and, concatenated together give a 128-dimensional embedding
  vector.
- The latent embedding (z) is 128-dimensional
- The "Encoder Layers" list in the appendix refers to the MLP dimensions in the
  transformer. (i.e. the residual/attention activations are all 128-dimensional,
  and the MLP block expands to 256, then back to 128.
- There is no embedding step for the observation state passed to the decoder.
  The raw state and the z-vector are concatenated directly.
- For conditioning the RL components (value, critic, etc.) of the FRE-agent with
  the latent embedding z, the latent embedding is simply concatenated to the
  observation state that is fed into the RL components.

### Details on the evaluation tasks

#### Ant Maze evaluation tasks

The online evaluation is performed with a _maximum_ length of 2000 steps _per
trajectory_.

- `ant-goal-reaching`: This is the average performance on a set of 5 goal
  reaching tasks, consisting in 5 hand-crafted fixed reward functions which
  respectively reward the agent for reaching a goal location. The reward is set
  to -1 for every timestep that the goal is not achieved. The following 5
  locations on an (X,Y) grid with the origin at the bottom left are used for the
  5 hand-crafted evaluation reward functions:
  - `goal-bottom` at `(28, 0)`
  - `goal-left` at `(0, 15)`
  - `goal-top` at `(35, 24)`
  - `goal-center` at `(12, 24)`
  - `goal-right` at `(33, 16)`
- `ant-directional`: This is the average performance on 5 directional tasks.
  - Each evaluation task specifies a **target velocity** in the (X,Y) plane
    (e.g., left = (-1,0), up = (0,1), etc.).
  - The reward function checks the agent’s actual velocity and grants higher
    reward the closer it is to the target velocity, using a simple dot product.
  - The four specific directions tested are:
    - `vel_left` (target velocity = (-1, 0))
    - `vel_up` (target velocity = (0, 1))
    - `vel_down` (target velocity = (0, -1))
    - `vel_right`(target velocity = (1, 0))
  - The final performance reported is averaged across these four directional
    tasks.
- `ant-random-simplex`: This is the average performance on 5 seeded tasks.
  - Each evaluation task is defined by a **random 2D noise** “height map” plus
    velocity preferences in the (X,Y) grid of the AntMaze generated via
    opensimplex
  - There are **five** fixed seeds (1 to 5), each yielding a different noise
    pattern.
  - The agent gets baseline negative reward (-1) at each step, a bonus if it
    stands in higher “height” regions, and an additional bonus for moving in the
    local “preferred” velocity direction indicated by the noise field.
  - The final performance reported is the average of these five seeds.
- The `ant-path-center`, `ant-path-loop` and `ant-path-edges` are simply reward
  functions that reward the agent for moving along hand-crafted corridors placed
  in the center of the grid, for moving in a hand-crafted loop around the grid,
  and for moving along the edges of the grid, respectively.

#### ExORL evaluation tasks

The online evaluation is performed with a _maximum_ length of 1000 steps _per
trajectory_.

- `exorl-cheetah-velocity`: Specific to the `cheetah` domain, this is the
  average performance on 4 custom tasks:
  - `cheetah-run`: At each step, the agent is assigned a reward if the agent's
    horizontal forward velocity is at least 10. The reward is 1 if the velocity
    is 10 and linearly decays to 0 for values below 10. If the agent's
    horizontal velocity is in the opposite direction of the target velocity, the
    reward is 0.
  - `cheetah-run-backwards`: Same as `cheetah-run`, but rather than forward
    horizontal velocity, the agent is rewarded for backward horizontal velocity.
  - `cheetah-walk`: Same as `cheetah-run`, but the agent is rewarded for a
    velocity of at least 1.
  - `cheetah-walk-backwards`: Same as `cheetah-run-backwards`, but the agent is
    rewarded for a velocity of at least 1.
- `exorl-cheetah-goals`: Specific to the `cheetah` domain, this is the average
  performance on 5 goal reaching tasks.
  - Specifically, 5 random states are selected from the offline dataset and used
    as goal states, and kept fixed throughout the online evaluation.
  - The agent is assigned a reward of -1 at each step unless it is within a
    threshold distance of 0.1 of the goal state, in which case it is assigned a
    reward of 0
  - The distance is the euclidean distance between the agent's current state and
    the goal state.
- `exorl-walker-velocity`: Specific to the `walker` domain, this is the average
  performance on 4 custom tasks, which are referred to as `walker-run`, and
  `walker-walk` in the paper (despite being 4 tasks in total). velocity is at
  least some threshold value.
  - The reward is 1 if the velocity is at least the threshold value and linearly
    decays to 0 for values below the threshold value.
  - If the agent's horizontal velocity is in the opposite direction of the
    target velocity, the reward is 0.
  - The 4 tasks use values of 0.1, 1, 4, and 8 respectively.
- `exorl-walker-goals`: Specific to the `walker` domain, this is the average
  performance on 5 goal reaching tasks.
  - Specifically, 5 random states are selected from the offline dataset and used
    as goal states, and kept fixed throughout the online evaluation.
  - The agent is assigned a reward of -1 at each step unless it is within a
    threshold distance of 0.1 of the goal state, in which case it is assigned a
    reward of 0
  - The distance is the euclidean distance between the agent's current state and
    the goal state.

Finally, the information about the physics used for training that is mentioned
in Appendix C.2 is also used during evaluation.

### Clarifications on FRE Prior Reward Distributions

- `FRE-all`:
  - refers to the vanilla prior reward distribution, including an equal split of
    - singleton goal-reaching reward functions
    - random linear functions
    - random MLP functions
  - It is used in sections 5.1 and 5.2, where it just referred to as `FRE`, in
    section 5.3 where it is referred to as `FRE-all`, and in section 5.4 where
    it is referred to as `FRE`. For clarity it will be canonically referred to
    as `FRE-all`.
- `FRE-hint`:
  - Uses a prior reward distribution that is a superset of the evaluation tasks.
    For ant-directional, the prior rewards are all reward corresponding to
    movement in a unit (x,y) direction. For Cheetah-velocity and
    walker-velocity, the rewards are for moving at a specific velocity
- `FRE-goals`:
  - This is a prior reward distribution consisting exclusively of singleton
    goal-reaching reward functions.
- `FRE-lin`:
  - This is a prior reward distribution consisting exclusively of random linear
    reward functions.
- `FRE-mlp`:
  - This is a prior reward distribution consisting exclusively of random MLP
    reward functions.
- `FRE-lin-mlp`:
  - This is a prior reward distribution consisting of an equal split of random
    linear and random MLP reward functions.
- `FRE-goal-mlp`:
  - This is a prior reward distribution consisting of an equal split of
    singleton goal-reaching and random MLP reward functions.
- `FRE-goal-lin`:
  - This is a prior reward distribution consisting of an equal split of
    singleton goal-reaching and random linear reward functions.

### Additional Details on GC-BC

- Network Architecture
  - A multi-layer perceptron (MLP)with three hidden layers of size 512
  - ReLU activationsare applied between each hidden layer.
  - Layer normalization is applied before each activation to stabilize training.
  - The output layer predicts a Gaussian distribution over actions, i.e. two
    outputs:
    - The **mean action** is a linear output.
    - The **log of the standard deviation** is clamped with a lower bound of
      \(-5.0\) to prevent excessive variance.
- Loss Function:
  - The policy is optimized using maximum likelihood estimation (MLE), where the
    objective is to maximize the log-probability of dataset actions under the
    predicted action distribution.
  - The loss function is given by:
    $$
        L_{\pi} = -\mathbb{E}_{(s, g, a) \sim D} \log \pi(a | s, g)
    $$
- Training: Hindsight relabeling is used during training where the goal is
  sampled from the dataset. Specifically for GC-BC, only use geometric sampling
  is used to sample goals from future states in the trajectory (i.e. no random
  goals, or goals which are the current state).
- Evaluation: the goal-conditioned agent is given the ground-truth goal that the
  specific evaluation task contains, to condition on.

### Additional Details on GC-IQL

GC-IQL is just IQL with the additional goal state.

Specifically, given a random state, a random goal state is sampled from 1)
future states in the trajectory using a geometric distributions, 2) a random
goal in the dataset, or 3) the current state is the goal, in which case the
reward is 0 and the mask/terminal flag is True. In other cases, reward is -1.
The ratio is p_randomgoal=0.3, p_geometric_goal=0.5, p_current_goal=0.2.

To give the information to the neural network, goals and observations are
concatenated.

No environment rewards are used to train GC-IQL. The reward passed to IQL are
the goal-conditioned rewards generated by the preprocessor as described above,
which is 0 if the state==goal and -1 otherwise.

Evaluation: the goal-conditioned agent is given the ground-truth goal that the
specific evaluation task contains, to condition on.

### Additional Details on SF and FB Baselines

- Both the SF and FB baselines are trained and evaluated using the following
  codebase: <https://github.com/facebookresearch/controllable_agent>
- As such, reproductions should also use this codebase for training and
  evaluating these baselines. Failure to do so will result in missing partial
  credit assignment.
- some further details:
  - All SF/FB ExoRL experiments use the RND dataset.
  - ICM features are used for SF.
  - Training the FB/SF policies did not require any changes to the
    `facebookresearch/controllable_agent` codebase.
  - For SF/FB evaluation, the set of evaluation tasks considered in the paper
    were re-implemented. To do this, the authors introduced a custom reward
    function into the pre-existing environments (e.g. antmaze, walker, cheetah,
    kitchen) that replaced the default reward with their custom rewards.
  - To run the FB baseline, the authors largely followed the instructions on
    `facebookresearch/controllable_agent` repo:
    - First, they download the offline RND dataset.
    - Then, they construct the replay buffer using the code from the repo
      README, and run the training command.
    - Evaluation numbers are logged during the training run.

### Additional Details on OPAL

- No manually designed rewards are used in OPAL.
- For the OPAL encoder, the same transformer architecture is used as in FRE.
- For the privileged execution evaluation described in the paper:
  - OPAL's task policy is not used
  - 10 random skills are sampled from a unit Gaussian,
  - for each skill $z$, the policy is conditioned on it and evaluated for the
    entire episode,
  - and the best performing rollout is taken.
