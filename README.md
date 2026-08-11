# Reinforcement Learning

A research-oriented journey through **Reinforcement Learning (RL)**, from the mathematical foundations of sequential decision-making to modern deep and multi-agent reinforcement learning.

This repository contains my notes, implementations, experiments, and research projects developed while studying reinforcement learning in depth.

The goal is not simply to implement algorithms, but to understand **why they work, where they fail, and how the underlying theory leads to modern reinforcement learning methods**.

---

## Topics

### 1. Multi-Armed Bandits

Foundations of exploration and exploitation.

* Epsilon-Greedy
* Optimistic Initial Values
* Upper Confidence Bound (UCB)
* Gradient Bandit Algorithms
* Non-stationary bandits
* Experimental comparisons

### 2. Markov Chains

Mathematical foundations for stochastic processes.

* Markov property
* Transition matrices
* State distributions
* Hitting probabilities
* Mean hitting times
* Absorbing Markov chains

### 3. Markov Decision Processes

The mathematical framework underlying reinforcement learning.

* States and actions
* Transition dynamics
* Rewards
* Policies
* Returns
* Value functions
* Bellman equations
* Discounting
* Optimal policies

### 4. Dynamic Programming

Planning when the environment model is known.

* Policy Evaluation
* Policy Improvement
* Policy Iteration
* Value Iteration
* Bellman optimality equations

### 5. Monte Carlo Methods

Learning from complete episodes.

* Monte Carlo prediction
* Monte Carlo control
* Exploring starts
* ε-soft policies

### 6. Temporal-Difference Learning

Learning directly from experience.

* TD(0)
* SARSA
* Q-Learning
* Expected SARSA
* On-policy vs. off-policy learning
* Eligibility traces

### 7. Policy-Based Methods

Learning policies directly.

* Policy gradients
* REINFORCE
* Baselines
* Advantage functions
* Actor-Critic methods

### 8. Deep Reinforcement Learning

Combining reinforcement learning with deep neural networks.

* Deep Q-Networks (DQN)
* Experience replay
* Target networks
* Double DQN
* Dueling DQN
* Policy gradient methods
* Actor-Critic architectures
* PPO

### 9. Multi-Agent Reinforcement Learning

Exploring decision-making in environments containing multiple interacting agents.

* Independent learners
* Cooperative environments
* Competitive environments
* Centralized training
* Decentralized execution
* Multi-agent policy learning

### 10. Research Experiments

Experiments designed to investigate reinforcement learning concepts rather than simply reproduce algorithms.

These include:

* Algorithm comparisons
* Ablation studies
* Exploration strategies
* Learning stability
* Sample efficiency
* Reward design
* Environment complexity
* Failure analysis

---

## Mathematical Foundations

Reinforcement learning sits at the intersection of several areas of mathematics and computer science.

This repository therefore also contains supporting material covering:

* Probability
* Statistics
* Linear algebra
* Calculus
* Optimization
* Markov processes
* Dynamic programming
* Stochastic processes

The objective is to develop enough mathematical understanding to move beyond treating RL algorithms as black boxes.

---

## Tools and Technologies

* Python
* NumPy
* PyTorch
* Gymnasium
* Matplotlib
* Jupyter
* CUDA
* Git / GitHub

---

## Repository Structure

```text
reinforcement-learning/
│
├── 01_bandits/
│
├── 02_markov_chains/
│
├── 03_markov_decision_processes/
│
├── 04_dynamic_programming/
│
├── 05_monte_carlo/
│
├── 06_temporal_difference/
│
├── 07_policy_methods/
│
├── 08_deep_reinforcement_learning/
│
├── 09_multi_agent_rl/
│
├── 10_research/
│
├── environments/
│
├── experiments/
│
├── notebooks/
│
├── notes/
│
└── README.md
```

---

## Research Approach

I am approaching reinforcement learning from both the theoretical and experimental perspectives.

For each major algorithm or concept, the goal is to understand:

1. **The problem** — What are we trying to solve?
2. **The assumptions** — What must be true for the method to work?
3. **The mathematics** — Where does the algorithm come from?
4. **The algorithm** — How is the mathematical idea implemented?
5. **The experiment** — Does it actually work?
6. **The limitations** — When does it fail?
7. **The connection** — How does it lead to more advanced methods?

This approach is intended to build a strong foundation for eventually working on reinforcement learning research.

---

## References

The primary theoretical reference for this repository is:

Richard S. Sutton and Andrew G. Barto,
*Reinforcement Learning: An Introduction*, 2nd Edition.

Additional papers, textbooks, and research articles will be added as the repository develops.

---

## Status

This repository is actively under development.

The material will evolve alongside my study of reinforcement learning, with new implementations, experiments, notes, and research projects added progressively.

---

## Long-Term Goal

The long-term objective is to progress from understanding the fundamental principles of reinforcement learning to being able to:

* Implement RL algorithms from first principles
* Analyze their mathematical foundations
* Design meaningful experiments
* Build custom RL environments
* Investigate algorithmic limitations
* Develop deep RL systems
* Explore multi-agent reinforcement learning
* Read and critically analyze RL research papers
* Eventually contribute to reinforcement learning research

> **Understand the theory. Implement the algorithm. Run the experiment. Question the result.**

