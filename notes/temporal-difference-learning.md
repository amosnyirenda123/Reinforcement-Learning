# Temporal Difference Learning

- Combines the advantages of both the dynamic programming (DP) method and the Monte Carlo (MC) method.
- In DP, we take advantage of a technique known as **bootstrapping** to compute the value of a state without waiting for the end of the episode. However, we need to know the model dynamics of the environment to use this method.
- In the Monte Carlo method, we do not need to know the model dynamics of the environment. However, we do need to wait until the end of the episode to estimate the state value or Q value. We also cannot apply MC methods to continuous tasks (non-episodic tasks).
- In **TD learning**, we perform bootstrapping so that we do not have to wait for the end of the episode to compute the state value or Q value.
- TD is also a **model-free** method — that is, we do not need to know the model dynamics of the environment.

## TD Prediction

- The policy is given as input, and we try to estimate the value function using that policy.
- In MC, we estimate the value of a state across $N$ episodes by:

$$
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} R_i(s)
$$

- The disadvantage is that we have to wait until the end of the episode to compute the value of a state, which can take time.
- So, in TD learning, we make use of bootstrapping and estimate the value of a state as:

$$
V(s) \approx r + \gamma V(s')
$$

- A single estimate $r + \gamma V(s')$ cannot approximate the value of a state perfectly, so we instead take a running (incremental) mean of these estimates over time.
- In the MC method, the incremental mean update for the value function is given by:

$$
V(s) = V(s) + \alpha\left(R - V(s)\right)
$$

- In TD learning, the incremental mean update is given by:

$$
V(s) = V(s) + \alpha \left(r + \gamma V(s') - V(s)\right)
$$

- This is known as the **TD learning update rule**.
- We do not use the full return $R$ here — instead, we use the bootstrap estimate $r + \gamma V(s')$, so that we don't have to wait until the end of the episode to compute the value of the state.
- The difference between the bootstrap estimate and $V(s)$, i.e. $\left(r + \gamma V(s') - V(s)\right)$, is known as the **TD error**.
- $\alpha$ is the learning rate, or step size.

### The TD Prediction Algorithm

1. Given the policy, initialize the values of all states with random values.
2. For each step in the episode:
   1. Perform an action in the current state $s$ according to the given policy; observe the reward $r$ and the next state $s'$.
   2. Update the value of state $s$ using the TD update rule.
   3. Set $s \leftarrow s'$.
   4. If $s$ is not a terminal state, repeat from step 2.1.

## On-Policy TD Control — SARSA

- The goal is to find the optimal policy.
- We extract the policy from the Q function.
- The update rule in terms of the Q function is given by:

$$
Q(s, a) = Q(s, a) + \alpha\left(r + \gamma Q(s', a') - Q(s, a)\right)
$$

- We first initialize the Q function with random values (or zeros), then extract a policy from this randomly initialized Q function.
- The Q function is updated after every step, and the policy is re-extracted from the updated Q function.
- We select actions using the epsilon-greedy policy: with probability $1-\epsilon$ we select the best (greedy) action, and with probability $\epsilon$ we select a random action.
- These steps are repeated over many episodes to converge on the optimal policy.

### The SARSA Algorithm

1. Initialize the Q function $Q(s,a)$ with random values.
2. Extract a policy from $Q(s,a)$ and select an action $a$ to perform in the starting state $s$ (e.g. via the epsilon-greedy policy).
3. For each step in the episode:
   1. Perform action $a$, move to the next state $s'$, and observe the reward $r$.
   2. In state $s'$, select the next action $a'$ using the epsilon-greedy policy.
   3. Update the Q value of $(s,a)$ using the SARSA update rule above.
   4. Set $s \leftarrow s'$ and $a \leftarrow a'$.
   5. Repeat this process if $s$ is not a terminal state.

SARSA gets its name from the quintuple used in each update: **S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction.

## Off-Policy TD Control — Q-Learning

- We use two different policies: an epsilon-greedy policy and a greedy policy.
- To select an action *in the environment*, we use the epsilon-greedy policy. To update the Q value of the *next* state-action pair, we use the greedy policy instead.
- Since we use the greedy policy to select the action for the next state, the Q value update rule becomes:

$$
Q(s, a) = Q(s, a) + \alpha\left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)
$$

- So in Q-learning, we select an action in the environment using the epsilon-greedy policy, but when computing the value of the next state-action pair, we use the greedy (max) policy instead — this mismatch between the acting policy and the updating policy is what makes it **off-policy**.

### The Q-Learning Algorithm

1. Initialize the Q function $Q(s,a)$ with random values (or zeros) for all state-action pairs.
2. For each episode:
   1. Initialize the starting state $s$.
   2. For each step in the episode:
      1. Select action $a$ in state $s$ using the epsilon-greedy policy (derived from $Q$).
      2. Perform action $a$, observe the reward $r$ and the next state $s'$.
      3. Update the Q value using the Q-learning update rule:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha\left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)
$$

      4. Set $ s \leftarrow s' $.
      5. Repeat from step 2.2.1 if $s$ is not a terminal state.
3. Repeat for many episodes until the Q function converges to $Q^{*}$, from which the optimal policy $\pi^{*}(s) = \operatorname*{arg\,max}_{a} Q^{*}(s,a)$ can be extracted.

## SARSA vs Q-Learning

- **SARSA** is an **on-policy** algorithm: we use a single epsilon-greedy policy both to select an action in the environment *and* to compute the Q value of the next state-action pair.

$$
Q(s, a) = Q(s, a) + \alpha\left(r + \gamma Q(s', a') - Q(s, a)\right)
$$

- **Q-learning** is an **off-policy** algorithm: we use the epsilon-greedy policy to select an action in the environment, but we use the **greedy** policy (not epsilon-greedy) to compute the value of the next state-action pair.

$$
Q(s, a) = Q(s, a) + \alpha\left(r + \gamma \max_{a'} Q(s', a') - Q(s, a)\right)
$$