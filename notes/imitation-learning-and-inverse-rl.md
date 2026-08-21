# Imitation Learning and Inverse RL

- Supervised imitation learning
- DAgger
- Deep Q-learning from demonstrations
- Inverse reinforcement learning
- Maximum entropy inverse reinforcement learning
- Generative adversarial imitation learning

## Supervised Imitation Learning

- In the imitation learning setting, our goal is to mimic an expert.
- Agents learn from **expert demonstrations**.
- Expert demonstrations are a set of trajectories consisting of state-action pairs, where each action was performed by the expert.
- We can view expert demonstrations as training data used to train our agent.
- In supervised imitation learning, we perform the following steps:
  - Collect a set of expert demonstrations.
  - Initialize a policy $\pi_{\theta}$.
  - Learn the policy by minimizing the loss $\mathcal{L}(a^{*}, \pi_{\theta}(s))$, where $a^{*}$ is the expert's action and $\pi_{\theta}(s)$ is the action produced by the agent — this is just ordinary supervised learning (e.g. classification, if actions are discrete), with expert actions as labels.

### Drawbacks

- The agent's knowledge is limited entirely to the expert demonstrations (its training data) — it has no way to learn about states the expert never visited.
- The agent's accuracy is highly dependent on the quality of the expert's demonstrations: if the expert's demonstrations aren't optimal, the agent has no mechanism to learn better-than-expert (or even correctly optimal) actions — it can only ever imitate what it was shown, mistakes included.

## DAgger (Dataset Aggregation)

- One of the most widely used imitation learning algorithms.

### How DAgger Works

Suppose we're training an agent to drive a car. We start off with an empty dataset $\mathcal{D}$.

**Iteration 1:**
- We start with some initial policy $\pi_1$ and generate a trajectory $\tau$ by running it.
- We create a new dataset $\mathcal{D}_1$ consisting only of the *states visited by our policy*, with the expert providing the correct action label for each of those states.
- We aggregate this into our running dataset:

$$
\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_1
$$

- We train a classifier on the updated dataset $\mathcal{D}$, producing a new policy $\pi_2$.

**Iteration 2:**
- We run the newly trained policy $\pi_2$ in the environment to generate a fresh trajectory, visiting states that $\pi_2$ — not $\pi_1$ — actually encounters (which may well include mistakes or edge cases $\pi_1$ never ran into).
- We create $\mathcal{D}_2$, again consisting of these newly visited states, labeled with the expert's action at each: $\mathcal{D}_2 = \{(s, \pi_E(s)) : s \text{ visited by } \pi_2\}$.
- We aggregate again: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_2$.
- We retrain the classifier on the full (now larger) aggregated dataset $\mathcal{D}$, producing $\pi_3$.

**Iteration 3, 4, …, N:** we simply repeat this same pattern — run the latest policy $\pi_i$ to collect newly visited states, query the expert for correct actions at those states to form $\mathcal{D}_i$, aggregate it into $\mathcal{D}$, and retrain on the full dataset to get $\pi_{i+1}$. The key idea driving all of this: each new policy is trained not just on the *expert's own* trajectories, but increasingly on the states the *agent itself* tends to wander into (including its own mistakes) — this is exactly what addresses supervised imitation learning's core drawback above, since the training data now grows to cover the agent's actual, learned behavior, not just the expert's.

### Formalization

Suppose we have a human expert, whose policy we denote $\pi_E$. We initialize an empty dataset $\mathcal{D}$, and a novice policy $\hat\pi_1$.

**Iteration 1:**
- We create a policy $\pi_1$ by mixing some amount of the expert policy $\pi_E$ with some amount of the novice policy $\hat\pi_1$:

$$
\pi_1 = \beta_1 \pi_E + (1 - \beta_1)\hat{\pi}_1
$$

- How much of each we take is controlled by the parameter $\beta$, given by:

$$
\beta_i = p^{\,i - 1}, \qquad 0.1 \leq p \leq 0.9
$$

- Since we're in the first iteration, $i=1$, so $\beta_1 = p^{0} = 1$, which leaves us with:

$$
\pi_1 = \pi_E
$$

  So in the very first iteration, the mixed policy $\pi_1$ is simply the expert policy outright — makes sense, since we have no novice policy worth trusting yet.
- We create a new dataset $\mathcal{D}_1$ consisting of the states visited by $\pi_1$, with the expert providing the action at each:

$$
\mathcal{D}_1 = \left\{(s, \pi_E(s))\right\}
$$

- We aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_1$.
- We train a novice policy $\hat\pi_2$ on this dataset via supervised learning.

**Iteration 2:**
- We now form $\pi_2 = \beta_2 \pi_E + (1-\beta_2)\hat\pi_2$, where $\beta_2 = p^{\,2-1} = p$. Since $p < 1$, this means $\pi_2$ already leans more heavily on our newly trained novice policy $\hat\pi_2$ than on the expert, compared to iteration 1.
- We run this mixed policy $\pi_2$ to collect newly visited states, and again query the expert for the correct action at each: $\mathcal{D}_2 = \{(s, \pi_E(s)) : s \text{ visited by } \pi_2\}$.
- We aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_2$.
- We retrain on the full aggregated dataset $\mathcal{D}$ to get a new novice policy $\hat\pi_3$.
- As $i$ increases, $\beta_i = p^{\,i-1} \to 0$ (since $p<1$), meaning later iterations rely almost entirely on the learned novice policy $\hat\pi_i$ to decide which states to visit, while the expert is only ever consulted to *label* those states (never to *choose* which states get visited). This gradual handover is exactly what lets DAgger's final policy generalize to states the pure expert policy might never have wandered into on its own.

### The DAgger Algorithm

1. Initialize the aggregated dataset $\mathcal{D} \leftarrow \emptyset$.
2. Initialize a novice policy $\hat\pi_1$ (e.g. randomly, or via ordinary supervised imitation learning on an initial expert-only dataset).
3. For $i = 1, 2, \dots, N$:
   1. Form the mixed policy $\pi_i = \beta_i \pi_E + (1-\beta_i)\hat\pi_i$, with $\beta_i = p^{\,i-1}$.
   2. Run $\pi_i$ in the environment to generate one or more trajectories, recording the states visited.
   3. Query the expert for the correct action at each visited state, forming $\mathcal{D}_i = \{(s, \pi_E(s))\}$.
   4. Aggregate: $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_i$.
   5. Train a new novice policy $\hat\pi_{i+1}$ on the full aggregated dataset $\mathcal{D}$, via supervised learning (minimizing the loss between predicted and expert actions).
4. Return the best-performing policy $\hat\pi_i$ found across all iterations (typically evaluated on a held-out validation set or via direct rollout performance).

## Deep Q-Learning from Demonstrations (DQfD)

- We fill the replay buffer with expert demonstrations and **pre-train** the agent on them before it ever interacts with the environment itself.
- Once pre-trained, the agent interacts with the environment, gathers its own experience, and continues learning from a mix of both expert and self-generated data.
- DQfD therefore consists of two phases: **pre-training** and **training**.
- We use a **prioritized** experience replay buffer, and give expert demonstrations higher priority than self-generated data (see [DQN and Its Variants](./deep-q-networks-and-variants.md) for prioritized experience replay).

### Loss Function of DQfD

The full loss is the sum of four separate loss terms:

- **Double DQN loss**, $\mathcal{L}_{DQ}(Q)$ — the ordinary 1-step Double DQN loss (see [DQN and Its Variants](./deep-q-networks-and-variants.md)).
- **N-step Double DQN loss**, $\mathcal{L}_n(Q)$ — the same idea, but bootstrapped over $n$ steps instead of just 1, which helps propagate reward information (and especially the expert's demonstrated behavior) further back through the value function more quickly.
- **Supervised classification loss** (large margin loss), $\mathcal{L}_E(Q)$:

$$
\mathcal{L}_E(Q) = \max_{a \in A}\big[Q(s, a) + l(a_E, a)\big] - Q(s, a_E)
$$

  **Explaining each term:**
  - $a_E$ is the action the expert actually took in state $s$ (the label).
  - $Q(s,a)$ is the network's current Q value for action $a$ in state $s$.
  - $l(a_E, a)$ is a **margin function**: it equals $0$ when $a = a_E$, and some fixed positive constant (e.g. $0.8$) whenever $a \neq a_E$. Its purpose is to require the expert's action to have a Q value that beats every other action by *at least* this margin — not just be marginally higher.
  - $\max_{a \in A}[Q(s,a) + l(a_E,a)]$ finds whichever action currently looks best *after* adding this margin bonus to every non-expert action — effectively asking "even giving every other action a head start, which one still looks most attractive right now?"
  - Subtracting $Q(s,a_E)$ measures how much this margin-boosted competitor exceeds the expert action's value. If $Q(s,a_E)$ already exceeds every other action's (margin-boosted) value, the loss is $0$ — meaning greedy action selection would already correctly reproduce the expert's choice, with a comfortable safety margin. Otherwise, the loss is positive, and gradient descent pushes $Q(s,a_E)$ up (or the competing actions' values down) until that margin is achieved.
- **L2 regularization loss**, $\mathcal{L}_{L2}(Q)$ — penalizes large network weights, helping prevent the agent from overfitting to the (comparatively small) set of expert demonstration data.

The final combined loss is:

$$
\mathcal{L}(Q) = \mathcal{L}_{DQ}(Q) + \lambda_1\mathcal{L}_n(Q) + \lambda_2 \mathcal{L}_E(Q) + \lambda_3 \mathcal{L}_{L2}(Q)
$$

- Each $\lambda$ acts as a weighting factor, controlling how much influence its corresponding loss term has on the overall update.

### The DQfD Algorithm

**Phase 1 — Pre-training:**
1. Fill the replay buffer $\mathcal{D}$ entirely with expert demonstration transitions $(s, a_E, r, s')$.
2. Initialize the main network $Q_\theta$ and target network $Q_{\theta'}$ (with $\theta' \leftarrow \theta$).
3. For a fixed number of pre-training steps:
   1. Sample a prioritized minibatch from $\mathcal{D}$ (entirely expert data at this stage).
   2. Compute the full combined loss $\mathcal{L}(Q) = \mathcal{L}_{DQ}(Q) + \lambda_1\mathcal{L}_n(Q) + \lambda_2\mathcal{L}_E(Q) + \lambda_3\mathcal{L}_{L2}(Q)$.
   3. Update $\theta$ via gradient descent: $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(Q)$.
   4. Periodically update the target network: $\theta' \leftarrow \theta$.

**Phase 2 — Training with self-generated experience:**
4. Continue running the now pre-trained policy in the actual environment.
5. For each step:
   1. Select an action $a$ via the epsilon-greedy policy with respect to $Q_\theta$.
   2. Execute $a$; observe reward $r$ and next state $s'$.
   3. Store the self-generated transition $(s,a,r,s')$ into the same replay buffer $\mathcal{D}$ (tagged as self-generated, distinct from expert transitions).
   4. Sample a prioritized minibatch from $\mathcal{D}$, drawing from both expert and self-generated transitions, with expert transitions weighted more heavily.
   5. Compute the combined loss as before — **note:** the supervised classification loss $\mathcal{L}_E(Q)$ is only meaningful for expert transitions (since only those have a known "correct" expert action $a_E$); it's typically omitted (or zeroed out) for self-generated transitions.
   6. Update $\theta$ via gradient descent, and periodically sync $\theta' \leftarrow \theta$.
6. Repeat until the agent's performance converges.

## Inverse Reinforcement Learning (IRL)

- If we have expert demonstrations, we can use them to **learn the reward function** the expert appears to be optimizing.
- IRL is, in a sense, the inverse of ordinary reinforcement learning.
- In RL, we try to find the optimal policy *given* a reward function.
- In IRL, we try to learn the reward function *given* expert demonstrations of (presumably close to optimal) behavior.
- Once we've recovered a reward function this way, we can plug it into any ordinary RL algorithm to train an agent to find the optimal policy under it.

### Maximum Entropy IRL

#### Key Terms

- **Feature vector:** we represent a state $s$ using a feature vector $f_s$ (e.g., hand-crafted or learned features summarizing relevant properties of that state).
- **Feature count:** the sum of the feature vectors of every state in a trajectory:

$$
f_{\tau} = \sum_{s \in \tau} f_s
$$

- The reward function is defined as a **linear combination** of features, with weights $\theta$:

$$
R_{\theta}(\tau) = \theta_1 f_{s_1} + \theta_2 f_{s_2} + \cdots + \theta_T f_{s_T} = \sum_{s \in \tau} \theta^{T} f_s = \theta^{T} f_{\tau}
$$

#### How Maximum Entropy IRL Works

- Suppose we have expert demonstrations $\mathcal{D}$. Our goal is to learn a reward function consistent with them.
- We sample a trajectory from $\mathcal{D}$, and try to find the reward function by finding the optimal parameter $\theta$.
- Trajectories that achieve higher reward under $R_\theta$ are assumed to be *more likely* to have been the ones the expert actually demonstrated.
- The probability of a trajectory being "selected" is modeled with a softmax (Boltzmann) distribution over trajectory rewards:

$$
p(\tau) = \frac{\exp(R_{\theta}(\tau))}{\sum_{\tau'}\exp(R_{\theta}(\tau'))}, \qquad 0 \leq p(\tau) \leq 1
$$

- Our objective is therefore to maximize $p(\tau)$ for the trajectories we actually observed the expert demonstrating.
- We define our objective function using the **log**-likelihood:

$$
J(\theta) = \frac{1}{M} \log p(\tau) = \frac{1}{M} \left(R_\theta(\tau) - \log \sum_{\tau'}\exp(R_{\theta}(\tau'))\right) = \frac{1}{M} \left(\theta^{T}f_{\tau} - \log \sum_{\tau'}\exp(\theta^{T}f_{\tau'})\right)
$$

  where $M$ is the number of demonstrations.

- **Why move to the log form?** Maximizing $\log p(\tau)$ is mathematically equivalent to maximizing $p(\tau)$ itself, since $\log$ is a strictly increasing (monotonic) function — it never changes *where* the maximum occurs, only the scale on which we measure it. But computationally, working with $p(\tau)$ directly is dangerous: it involves a ratio of exponentials, $\exp(R_\theta(\tau))$, summed over potentially enormous numbers of trajectories — and $R_\theta(\tau)$ can be a large positive or negative number, meaning $\exp(R_\theta(\tau))$ can easily overflow (become numerically infinite) or underflow (round to exactly zero) in floating-point arithmetic. Taking the log turns this ratio-of-exponentials into a difference involving a $\log$-$\sum$-$\exp$ term, which is far more numerically stable to compute (typically using the standard "log-sum-exp trick," which factors out the largest exponent before summing, to avoid overflow). This is an extremely common pattern anywhere a softmax-style probability shows up in machine learning — e.g. cross-entropy loss in classification, log-likelihoods in language models — precisely for this reason.

- We update $\theta$ via gradient ascent:

$$
\theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta)
$$

#### Computing the Gradient

We want to compute $\nabla_\theta J(\theta)$ for:

$$
J(\theta) = \frac{1}{M} \left(\theta^{T}f_{\tau} - \log \sum_{\tau'}\exp(\theta^{T}f_{\tau'})\right)
$$

**Step-by-step derivation:**

1. The gradient of the first (linear) term is straightforward:

$$
\nabla_\theta \big[\theta^T f_\tau\big] = f_\tau
$$

2. For the second term, apply the chain rule for $\nabla_\theta \log g(\theta) = \frac{\nabla_\theta g(\theta)}{g(\theta)}$, with $g(\theta) = \sum_{\tau'}\exp(\theta^T f_{\tau'})$:

$$
\nabla_\theta \log \sum_{\tau'}\exp(\theta^T f_{\tau'}) = \frac{\sum_{\tau'} \exp(\theta^T f_{\tau'})\, f_{\tau'}}{\sum_{\tau''}\exp(\theta^T f_{\tau''})} = \sum_{\tau'} \left[ \frac{\exp(\theta^T f_{\tau'})}{\sum_{\tau''}\exp(\theta^T f_{\tau''})} \right] f_{\tau'}
$$

3. The bracketed term is exactly $p(\tau')$ as defined above — so this simplifies to an expectation over trajectories, under the distribution induced by our current $\theta$:

$$
\nabla_\theta \log \sum_{\tau'}\exp(\theta^T f_{\tau'}) = \sum_{\tau'} p(\tau' \mid \theta)\, f_{\tau'} = \mathbb{E}_{\tau \sim p(\cdot\mid\theta)}\big[f_\tau\big]
$$

4. This expectation over whole trajectories can equivalently be rewritten as a sum over *individual states*, weighted by how often each state is visited under this trajectory distribution — since $f_\tau = \sum_{s\in\tau} f_s$:

$$
\mathbb{E}_{\tau \sim p(\cdot\mid\theta)}[f_\tau] = \sum_{s} p(s \mid \theta)\, f_s
$$

   where $p(s\mid\theta)$ is the probability of visiting state $s$, under the (soft-optimal) policy induced by the current reward guess $R_\theta$.

5. Putting it all together, and averaging the first term over the $M$ observed expert demonstrations (denoting $\hat f = \frac{1}{M}\sum_\tau f_\tau$, the empirical average feature count from the expert data), we arrive at:

$$
\nabla_{\theta} J(\theta) = \hat{f} - \sum_s p(s \mid \theta)f_s \qquad \hat{f} = \frac{1}{M} \sum_{\tau} f_{\tau}
$$

**Significance of each term:**
- $\hat f$ is the **empirical** (observed) average feature count from the expert's actual demonstrations — a fixed, directly computable quantity.
- $\sum_s p(s\mid\theta) f_s$ is the **expected** feature count under the current reward guess $\theta$ — i.e., "if an agent behaved optimally (in the max-entropy sense) under our current estimate of the reward function, what features would it typically end up visiting?"
- The gradient is the *difference* between these two quantities. When the gradient is zero, the reward parameters $\theta$ produce behavior whose expected feature counts exactly match the expert's observed feature counts — this is the defining condition of maximum entropy IRL: find the reward function under which optimal behavior "moment-matches" the expert's demonstrated behavior, in terms of feature usage.

**How do we compute $p(s\mid\theta)$?** This is done using a two-pass dynamic programming procedure closely related to value iteration:

1. **Backward pass:** run a *soft* (Boltzmann, rather than hard-max) version of value iteration, using our *current estimate* of the reward, $R_\theta(s) = \theta^T f_s$ — **not** the true environment reward, which we don't have access to (that's the entire premise of IRL: we only observe expert behavior, not a reward signal). This produces a stochastic, soft-optimal policy consistent with the current guess $\theta$.
2. **Forward pass:** starting from the initial state distribution, propagate visitation probabilities forward through time, using this soft-optimal policy together with the environment's (known) transition dynamics, accumulating the expected visitation frequency $p(s\mid\theta)$ for every state across the trajectory horizon.

**A critical caution:** it's tempting to think we should use the *true* environment reward for this value iteration step — but we don't have it. The whole point of IRL is that the environment's true reward is unknown; we only ever have the expert's demonstrated *behavior*. The algorithm works by alternating between (a) solving for the (soft-)optimal policy under our current *guess* of the reward $R_\theta$, and (b) updating that guess based on how well the resulting behavior's feature visitation matches the expert's. Using the true reward here would be both circular (we don't have it) and would defeat the purpose of the entire procedure.

### The Maximum Entropy IRL Algorithm

1. Extract feature vectors $f_s$ for every state, and compute the expert's empirical feature expectation $\hat f = \frac{1}{M}\sum_\tau f_\tau$ from the $M$ expert demonstrations.
2. Initialize the reward parameters $\theta$ randomly.
3. Repeat until convergence:
   1. Compute $R_\theta(s) = \theta^T f_s$ for every state, using the current $\theta$.
   2. Run a **backward** soft value iteration pass under $R_\theta$ to obtain the soft-optimal policy $\pi_\theta$.
   3. Run a **forward** pass, propagating state visitation frequencies under $\pi_\theta$ and the environment's transition dynamics, to obtain $p(s\mid\theta)$ for every state.
   4. Compute the gradient: $\nabla_\theta J(\theta) = \hat f - \sum_s p(s\mid\theta) f_s$.
   5. Update the reward parameters: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$.
4. Return the learned reward function $R_\theta(s) = \theta^T f_s$ (and/or the corresponding soft-optimal policy $\pi_\theta$).
5. *(Optional)* Use this learned reward function as the reward signal for any ordinary RL algorithm (e.g. Q-learning, policy gradient) to train a final agent from scratch.

## Generative Adversarial Imitation Learning (GAIL)

*(Brief overview — see Ho & Ermon, "Generative Adversarial Imitation Learning," NeurIPS 2016, for the full treatment.)*

- GAIL frames imitation learning as an adversarial game, directly analogous to a **GAN (Generative Adversarial Network)**.
- Two networks are trained against each other:
  - A **generator** — here, simply the policy $\pi_\theta$, which generates behavior (trajectories) rather than images.
  - A **discriminator** $D_w$, trained to distinguish between state-action pairs coming from the *expert's* demonstrations and state-action pairs generated by the *current policy*.
- The discriminator is trained with an ordinary binary classification loss: output close to $1$ for expert $(s,a)$ pairs, close to $0$ for policy-generated ones.
- The policy is trained (using any policy gradient method — the original paper uses TRPO) to *maximize* the discriminator's output on its own generated behavior — i.e., to fool the discriminator into believing its behavior is expert-like. The discriminator's output is used directly as a surrogate reward signal for policy gradient training, e.g. $r(s,a) = -\log(1 - D_w(s,a))$.
- As training proceeds, the policy's behavior becomes increasingly indistinguishable from the expert's — all without ever explicitly recovering a reward function as an intermediate step (unlike classical IRL, which produces $R_\theta$ along the way; GAIL learns the policy directly).
- **Advantages over classical (maximum entropy) IRL:** GAIL avoids the expensive inner-loop RL solve (e.g. the repeated value iteration sweeps over the entire state space) that classical IRL requires at every gradient step; it scales much more readily to high-dimensional, continuous state-action spaces using neural networks for both the policy and discriminator; and it doesn't require hand-crafted linear feature vectors the way the maximum entropy IRL formulation above does.