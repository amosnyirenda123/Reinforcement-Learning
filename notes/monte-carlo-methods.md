# Monte Carlo Methods

- A statistical technique used to find an approximate solution through sampling.
- The greater the sample size, the better the approximation.
- A **model-free** method — that is, it does not require the model dynamics of the environment to compute the value and Q functions in order to find the optimal policy.
- Given the expected value of a random variable $X$:

$$
\mathbb{E}\left(X\right) = \sum_{i=1}^{N} x_i \, p(x_i)
$$

- We can estimate the expected value of $X$ by simply sampling the values of $X$ some $N$ times:

$$
\mathbb{E}_{x \sim p(x)}\left[X\right] \approx \frac{1}{N}\sum_{i=1}^{N} x_i
$$

## Prediction and Control Tasks

### Prediction Task
- The policy $\pi$ is given as input, and we try to predict the value function or Q function under that policy.
- The goal is to **evaluate** the given policy.
- The policy is considered good when the agent obtains a good return.
- We compute the value function or Q function using the given policy.
- By computing the value function under the given policy, we can understand the expected return the agent would obtain from each state.
- In a prediction task, we do not make any change to the input policy.

### Control Task
- No policy is given as input.
- The goal is to find the **optimal** policy.
- We initialize a random policy and iteratively improve it to find the optimal policy.

## Monte Carlo Prediction

- **Recall:** the value function is given by:

$$
V^{\pi}(s) = \mathbb{E}_{\tau \sim \pi} \left[ R(\tau) \mid s_0 = s \right]
$$

- Using the Monte Carlo method, the value of state $s$ can be approximated by computing the average return obtained from state $s$ across $N$ episodes:

$$
V(s) \approx \frac{1}{N} \sum_{i=1}^{N} R_i(s)
$$

### The Monte Carlo Prediction Algorithm

Let **total_return(s)** be the sum of returns obtained from state $s$ across several episodes, and **N(s)** be the number of times state $s$ has been visited across those episodes.

1. Initialize `total_return(s)` and `N(s)` to zero for all states.

   For example (where $s_2$ is the final state):

   | State | total_return(s) | N(s) |
   |---|---|---|
   | $s_0$ | 0 | 0 |
   | $s_1$ | 0 | 0 |

2. Generate an episode using the policy $\pi$. For example:

$$
s_0 \xrightarrow{+1} s_1 \xrightarrow{+1} s_2
$$

   (where the $+1$ on each arrow is the reward received for that transition).

3. Store all the rewards from the episode in a list, e.g. `rewards = [1, 1]`.

4. For each time step $t$ in the episode:
   - Compute the return from state $s_t$ onward:

$$
R(s_t) = \sum_{k=t}^{T} \text{rewards}[k]
$$

   - Update the total return for that state: `total_return`$(s_t) \mathrel{+}= R(s_t)$
   - Update the visit counter: `N`$(s_t) \mathrel{+}= 1$

5. Compute the value of each state by averaging:

$$
V(s) = \frac{\text{total\_return}(s)}{N(s)}
$$

### First-Visit Monte Carlo
- If the same state is visited more than once within the same episode, we only compute (and count) its return the **first** time it is visited.

### Every-Visit Monte Carlo
- We compute (and count) the return **every** time a state is visited within an episode, even if it's visited multiple times.

## Monte Carlo Control

- The goal is to find the optimal policy.
- Since no policy is given as input, we begin by initializing a random policy.
- We then try to find the optimal policy iteratively.
- The policy is computed from the Q function as follows:

$$
\pi(s) = \operatorname*{arg\,max}_{a} Q(s,a)
$$

- So to compute the policy, we first need to compute the Q function.
- Using the Monte Carlo method, we compute the Q function as the average return across all visits to the state-action pair $(s,a)$:

$$
Q(s, a) = \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} R_i(s,a)
$$

- Since the policy was initialized randomly, the policy extracted from the Q function will not be optimal on the first iteration.
- In the next iteration, we use this new policy to generate episodes, recompute the Q function, and extract an even better policy.
- We repeat these steps iteratively until we converge on the optimal policy:

$$
\pi_{0} \rightarrow Q^{\pi_{0}} \rightarrow \pi_{1} \rightarrow Q^{\pi_{1}} \rightarrow \dots \rightarrow \pi^{*} \rightarrow Q^{\pi^{*}}
$$

- This process is referred to as **policy evaluation and improvement**.
- The policy is selected in a greedy manner, since we always take the action with the maximum Q value.

## On-Policy Control and Off-Policy Control

### On-Policy Control
- The agent behaves using one policy and also tries to improve that *same* policy.
- Episodes are generated using one policy, and that same policy is iteratively improved to find the optimal policy.
- On-policy Monte Carlo control methods fall into two categories: **Monte Carlo exploring starts** and **Monte Carlo with an epsilon-greedy policy**.

#### Monte Carlo Exploring Starts
- In any state there can be several possible actions, some optimal and some not.
- The agent has to explore an action to know whether it's a good one — without exploring it, the agent can never find out.
- In MC exploring starts, every state-action pair is given a non-zero probability of being chosen as the initial state-action pair.
- Before generating an episode, we choose the initial state-action pair at random, then generate the rest of the episode from that pair by following the policy $\pi$.
- The policy is then updated greedily at each iteration.
- This approach is not applicable to all environments. For instance, in a car-racing environment we want to always start from a particular position, not a random one.

#### Monte Carlo with Epsilon-Greedy Policy
- **Recall:** a greedy policy always selects the best action available at the moment.
- The problem with a purely greedy policy is that it always exploits the current best-known action — there could be a better alternative that hasn't been explored yet. This tension is known as the **exploration-exploitation dilemma**.

### Off-Policy Control
- The agent behaves using one policy but tries to improve a *different* policy.
- Episodes are generated using one policy, while a different policy is iteratively improved to find the optimal policy.
- We distinguish between the **behavior policy** and the **target policy**:
  - Episodes are generated using the **behavior policy** $b$.
  - We iteratively improve a different policy, the **target policy** $\pi$.
- For each step in the episode, we compute the return of the state-action pair and update the Q function as an average return.
- From the Q function, we extract a new target policy.
- We repeat these steps iteratively.
- The behavior policy is usually set to an epsilon-greedy policy (to keep exploring).
- The target policy is set to be the fully greedy policy (to keep exploiting what's been learned).
- Because the two policies differ, we use a technique called **importance sampling** to estimate the values of one distribution using samples drawn from another.

#### Importance Sampling

- To compute the expectation of a function $f(x)$ where $x$ is sampled from distribution $p(x)$, i.e. $x \sim p(x)$, we write:

$$
\mathbb{E}\left[f(x)\right] = \int_{x} p(x)\,f(x)\,dx
$$

- In importance sampling, $x$ is instead drawn from a different distribution $q(x)$, so:

$$
\mathbb{E}\left[f(x)\right] \approx \int_{x} f(x) \frac{p(x)}{q(x)} \, q(x) \, dx
$$

- Approximated with $N$ samples drawn from $q(x)$, this becomes:

$$
\mathbb{E}\left[f(x)\right] \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i) \frac{p(x_i)}{q(x_i)}
$$

- There are two types of importance sampling: **ordinary importance sampling** and **weighted importance sampling**.

- In **ordinary importance sampling**, the importance sampling ratio is the ratio of the target policy's probability to the behavior policy's probability:

$$
\frac{\pi(a \mid s)}{b(a \mid s)}
$$

- In **weighted importance sampling**, the ratio also carries a cumulative weight $W$:

$$
W\frac{\pi(a \mid s)}{b(a \mid s)}
$$

- **Recall:** the incremental method for computing the Q function is:

$$
Q(s_t, a_t) = Q(s_t, a_t) + \alpha \left( R_t - Q(s_t, a_t) \right)
$$

- Letting $W$ be the weight and $C(s_t, a_t)$ be the cumulative sum of weights across all episodes for that state-action pair, the incremental Q function update becomes:

$$
Q(s_t, a_t) = Q(s_t, a_t) + \frac{W}{C(s_t, a_t)} \left( R_t - Q(s_t, a_t) \right)
$$

### The Off-Policy Monte Carlo Control Algorithm

Using weighted importance sampling, the full off-policy Monte Carlo control algorithm is:

1. Initialize $Q(s,a)$ arbitrarily for all state-action pairs, and set $C(s,a) = 0$ for all $s,a$.
2. Initialize the target policy $\pi(s) = \operatorname*{arg\,max}_{a} Q(s,a)$ (greedy with respect to $Q$).
3. Repeat for each episode:
   1. Choose the behavior policy $b$ (e.g. epsilon-greedy) and generate a full episode with it: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$.
   2. Initialize the return $G = 0$ and the weight $W = 1$.
   3. Loop over the episode backward, for $t = T-1, T-2, \dots, 0$:
      - Update the return: $G \leftarrow \gamma G + R_{t+1}$
      - Update the cumulative weight: $C(S_t, A_t) \leftarrow C(S_t, A_t) + W$
      - Update the Q function: $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \dfrac{W}{C(S_t, A_t)} \left( G - Q(S_t, A_t) \right)$
      - Update the target policy for this state: $\pi(S_t) \leftarrow \operatorname*{arg\,max}_{a} Q(S_t, a)$
      - If $A_t \neq \pi(S_t)$, the behavior policy has diverged from the target policy for the rest of this episode — exit the inner loop and move to the next episode.
      - Otherwise, update the weight: $W \leftarrow W \dfrac{1}{b(A_t \mid S_t)}$
4. The target policy $\pi$ converges to the optimal policy $\pi^{*}$ as the number of episodes increases.