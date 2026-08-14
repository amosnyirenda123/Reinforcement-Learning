# Dynamic Programming

- Involves breaking a complex problem into sub-problems, then computing and storing the solution to each sub-problem.
- It is a **model-based** method — i.e., it helps find the optimal policy only when the model dynamics (transition probabilities) of the environment are known.

## Value Iteration

- **Recall:** the optimal policy tells the agent which action to perform in each state.
- To compute the optimal policy, we first compute the optimal value function, which then helps us derive the optimal policy.
- To compute the optimal value function, we use the Bellman optimality equation:

$$
V^{*}(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{*}(s') \right]
$$

- Recall the relationship between the value and Q functions:

$$
Q^{*}(s, a) = \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{*}(s') \right]
$$

- From this, we deduce that:

$$
V^{*}(s) = \max_{a} Q^{*}(s,a)
$$

- To compute the optimal value of a state, we compute the Q value of every state-action pair in that state and take the maximum. For example:

| State | Action | Q-value |
|---|---|---|
| $s_0$ | 0 | 2.7 |
| $s_0$ | 1 | 3 |
| $s_1$ | 0 | 4 |
| $s_1$ | 1 | 2 |

  The optimal values of states $s_0$ and $s_1$ are **3** and **4** respectively — the maximum Q value among the actions available in each state.

### The Value Iteration Algorithm

1. Compute the optimal value function by taking the maximum over the Q function:

$$
V^{*}(s) = \max_{a} Q^{*}(s,a)
$$

2. Extract the optimal policy from the computed optimal value function:

$$
\pi^{*}(s) = \operatorname*{arg\,max}_{a} Q^{*}(s, a)
$$

## Policy Iteration

The steps of the policy iteration algorithm are as follows:

1. Initialize a random policy $\pi$.
2. **Policy evaluation** — compute the value function for the current policy $\pi$ using the Bellman expectation equation:

$$
V^{\pi}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{\pi}(s') \right]
$$

   In practice, this is done iteratively: starting from an arbitrary $V_0$, we repeatedly apply the update

$$
V_{k+1}(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V_k(s') \right]
$$

   for every state $s$, until $V_k$ converges (i.e., $V_{k+1} \approx V_k$).

3. **Policy improvement** — extract a new, improved policy $\pi'$ by acting greedily with respect to the value function obtained in step 2:

$$
\pi'(s) = \operatorname*{arg\,max}_{a} \sum_{s'} P(s' \mid s, a) \left[ R(s, a, s') + \gamma V^{\pi}(s') \right]
$$

   which is equivalent to choosing the action with the highest Q value under $\pi$:

$$
\pi'(s) = \operatorname*{arg\,max}_{a} Q^{\pi}(s, a)
$$

4. If the extracted policy $\pi'$ is the same as the policy $\pi$ used in step 2, stop — $\pi$ is optimal. Otherwise, set $\pi \leftarrow \pi'$ and repeat steps 2–4.