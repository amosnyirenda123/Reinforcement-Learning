# Foundations of Reinforcement Learning

- Reinforcement learning (RL) is a trial-and-error process.
- Ultimately, the goal of the agent is to maximize the reward it receives.

## Key Elements

### Agent
- A software program that learns to make intelligent decisions.
- The learner in the RL environment.
- Example: a chess player can be considered an agent, since the player learns to make the best moves.

### Environment
- The world the agent operates in.
- The agent stays within the environment.
- Example: a chessboard is the environment, since the chess player learns to play the game within it.

### State
- A position or moment the agent can be in within the environment.
- Example: a position on the chessboard is a state.

### Action
- The agent interacts with the environment and moves from one state to another by performing an action.
- The set of all possible actions in the environment is called the **action space**. For example: `[up, down, left, right]`.
- Action spaces can be categorized as **discrete** or **continuous**:
  - A **discrete action space** consists of actions that take on discrete values.
  - A **continuous action space** consists of actions that are continuous — for example, the speed at which to drive a car, or the number of degrees to rotate a steering wheel.

### Reward
- Based on the action taken, the agent receives a reward.
- A reward is a numerical value the agent receives upon performing an action.

### Policy
- Defines the agent's behavior in an environment.
- Tells the agent what action to perform in each state.
- A random policy is initialized for the agent's first interactions with the environment.
- Over a series of iterations, the agent learns the optimal policy.
- A policy can be classified as either **deterministic** or **stochastic**.

### Episode
- The agent-environment interaction from the initial state until the final state is called an **episode**, also known as a **trajectory**, denoted $\tau$.
- Playing the game over multiple episodes allows the agent to learn the optimal policy.
- A sample episode from time step $t=0$ to $t=T$:

$$
(s_0, a_0, r_0, s_1, a_1, r_1, \dots, s_T)
$$

## The RL Algorithm

1. The agent interacts with the environment by performing an action.
2. In doing so, the agent moves from one state to another.
3. The agent receives a reward based on the action it performed.
4. Based on the reward, the agent learns whether the action was good or bad.
5. If the agent receives a positive reward, it will prefer performing that action again in the future.

## How RL Differs from Other ML Paradigms

- **Supervised learning** uses training data (a set of labeled input-output pairs) to train the model. In RL, the agent is not given any training data.
- **Unsupervised learning** uses unlabeled training data so the model can discover patterns and hidden structure in the input data. In RL, the model instead learns by maximizing reward.

## Markov Decision Processes

### The Markov Property and Markov Chains

- Provides a mathematical framework for solving RL problems.
- The **Markov property** states that the future depends only on the present, not on the past.
- Moving from one state to another is known as a **transition**.
- The transition probability from $s$ to $s'$ is denoted $P(s' \mid s)$.
- Transitions can be represented using Markov tables, transition matrices, or state diagrams.

### The Markov Reward Process (MRP)
- An extension of the Markov chain that adds a reward function.
- An MRP comprises states, transition probabilities, and a reward function.
- The reward function tells us the reward obtained in each state, denoted $R(s)$.
- So for an MRP, we have $P(s' \mid s)$ and $R(s)$.

### The Markov Decision Process (MDP)
- An extension of the MRP that adds actions.
- An MDP comprises states, transition probabilities, a reward function, and actions.
- An RL environment can be modeled as an MDP.
- The transition probability is denoted $P(s' \mid s, a)$.
- The reward function is denoted $R(s, a, s')$.

## Policy

### Deterministic Policy
- Tells the agent to perform one particular action in a given state.
- Maps each state to a single action.
- Whenever the agent visits that state, it always performs the same action.

$$
a_t = \mu(s_t)
$$

### Stochastic Policy
- Maps a state to a probability distribution over the action space.
- The agent performs different actions each time, sampled from the distribution returned by the policy.

$$
a_t \sim \pi(s_t)
$$

- A stochastic policy is called **categorical** when the action space is discrete. For example, in a grid-world environment where the agent learns the optimal path from position A to position I using actions `[up, down, right, left]`.
- A stochastic policy is called a **Gaussian policy** when the action space is continuous and follows a Gaussian probability distribution.

## Episodic and Continuous Tasks

An RL task can be categorized as episodic or continuous.

### Episodic Task
- Has a terminal (final) state. For example, in a car-racing game, we start from the starting point (initial state) and reach the destination (terminal state).

### Continuous Task
- Has no episodes and therefore no terminal state. For example, a personal assistant robot has no terminal state.

## Horizon

The horizon is the time step until which the agent interacts with the environment.

### Finite Horizon
- The agent-environment interaction stops at a particular time step. For example, in a car-racing (episodic) game, the agent starts interacting at $t=0$ and reaches the final state at time step $T$ (a finite horizon).

### Infinite Horizon
- The agent-environment interaction never stops.

## Return and Discount Factor

- The **return** is the sum of rewards obtained by the agent over an episode, denoted $R$ or $G$.
- For an agent whose interaction with the environment starts at $t=0$ and ends at $t=T$, the return is:

$$
R(\tau) = r_0 + r_1 + r_2 + \dots + r_T = \sum_{t=0}^{T} r_t
$$

- The agent's goal is to maximize the return — i.e., maximize the sum of rewards obtained over the episode.
- The optimal policy gets the agent the maximum return.
- For continuous tasks, the return is:

$$
R(\tau) = r_0 + r_1 + r_2 + \dots + r_\infty = \sum_{t=0}^{\infty} r_t
$$

- To prevent the return from reaching infinity, we introduce the **discount factor** $\gamma$, which controls how much importance is given to future versus immediate rewards. Its value lies between 0 and 1:

$$
R(\tau) = \gamma^0 r_0 + \gamma^1 r_1 + \gamma^2 r_2 + \dots = \sum_{t=0}^{\infty} \gamma^t r_t
$$

- A discount factor close to **0** gives more importance to immediate rewards.
- A discount factor close to **1** gives more importance to future rewards.
- A discount factor of exactly 0 means the agent never learns beyond $r_0$.
- A discount factor of exactly 1 means the agent considers all future rewards equally, which can cause the return to diverge to infinity.
- Whether to favor future or immediate rewards depends on the task.

## The Value Function

- Also known as the **state value function**, denoted $V^{\pi}(s)$.
- Represents the value of a state: the return the agent would obtain starting from that state and following policy $\pi$.
- There is no reward for the final state.
- Using the expected return:

$$
V^{\pi}(s) = \mathbb{E}\left[ R(\tau) \mid s_0 = s, \pi \right]
$$

- The value function depends on the policy.
- The **optimal value function** $V^{*}(s)$ yields the maximum value across all policies:

$$
V^{*}(s) = \max_{\pi} V^{\pi}(s)
$$

## The Q Function

- Also known as the **state-action value function**, denoted $Q^{\pi}(s,a)$.
- Represents the value of a state-action pair: the return the agent would obtain starting from state $s$, performing action $a$, and following policy $\pi$ thereafter.

$$
Q^{\pi}(s,a) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s,\, a_0 = a,\, \pi \right]
$$

- The Q function depends on the policy.
- The **optimal Q function** has the maximum Q value across all policies:

$$
Q^{*}(s,a) = \max_{\pi} Q^{\pi}(s,a)
$$

- The optimal policy is the one that yields the maximum Q value.

## Model-Based and Model-Free Learning

### Model-Based Learning
- The agent has a complete description of the environment.
- The agent uses the model's dynamics to find the optimal policy.

### Model-Free Learning
- The agent does not know the model dynamics of the environment.

## Types of Environments

### Deterministic Environments
- Performing action $a$ in state $s$ always leads to the same next state $s'$.

### Stochastic Environments
- Performing action $a$ in state $s$ does not always lead to the same next state $s'$, due to some randomness.

### Discrete Environments
- The environment's action space is discrete.

### Continuous Environments
- The environment's action space is continuous.

### Episodic Environments
- The agent's current action does not affect future actions.
- Also called a **non-sequential environment**.

### Non-Episodic Environments
- The agent's current actions affect future actions.
- Also called a **sequential environment**. Example: a chessboard is a sequential environment.

### Single- and Multi-Agent Environments
- A **single-agent environment** consists of only one agent.
- A **multi-agent environment** consists of multiple agents.