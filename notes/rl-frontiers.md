# RL Frontiers: Multi-Agent RL, Meta-RL, and RL for Reasoning Models

*A snapshot of where these three research directions stand as of mid-2026, written to connect back to the algorithms in your earlier notes and to help you decide where to dig in next. Research in these areas moves quickly — treat this as a map of the landscape and its vocabulary, not a final word.*

---

## 1. Multi-Agent Reinforcement Learning (MARL)

MARL extends everything in your earlier notes — Q-learning, DDPG, PPO, actor-critic — from a single agent to *many* agents acting in a shared environment, whether cooperating toward a common goal, competing, or some mix of both.

### Why MARL Is Harder Than Single-Agent RL

Three problems show up immediately once you add more agents:

- **Non-stationarity.** In single-agent RL, the environment's dynamics are fixed. With multiple *learning* agents, every other agent's policy is also changing over time — so from any one agent's point of view, the "environment" (which now includes the other agents' behavior) is constantly shifting underneath it, violating the stationarity assumptions most single-agent algorithms rely on.
- **Scalability.** The joint action space grows combinatorially with the number of agents — a setting with $N$ agents each choosing from $k$ actions has a joint action space of size $k^N$, which quickly becomes intractable for value-based methods that need to consider all joint actions.
- **Credit assignment.** When agents share a team reward, it's not obvious how much of that reward any single agent's action was responsible for — the same problem you saw in single-agent reward-to-go, but now spread across agents instead of just across time.

### Centralized Training, Decentralized Execution (CTDE)

The dominant paradigm for cooperative MARL is **CTDE**: agents get access to global information (other agents' observations, actions, or the full state) *during training*, but at execution/deployment time, each agent acts using only its own local observation<cite index="90-1,95-1">, letting agents exploit centralized information to coordinate during training while remaining independently deployable afterward, without needing communication at execution time</cite>. This maps directly onto your actor-critic notes: the actor stays decentralized and local, while the critic is where the "cheating" (access to global information) happens, since the critic is never needed at deployment.

Two direct extensions of algorithms you already know:

- **MADDPG (Multi-Agent DDPG):** <cite index="92-1,95-1">Each agent has its own actor network mapping its local observation to an action, exactly like DDPG's actor. The critic, however, is centralized — it's conditioned on the joint observations and actions of all agents, not just the one agent it belongs to, so it can account for how every agent's action affects the outcome.</cite> At execution time, only the local actors are needed, so agents act independently.
- **MAPPO (Multi-Agent PPO):** the same idea applied to PPO instead of DDPG — <cite index="92-1,93-1">a centralized critic conditioned on global state or shared observations, paired with decentralized actors trained via the same PPO-clipped objective from your PPO notes</cite>.

### Value Decomposition Methods (VDN, QMIX)

An alternative to a centralized critic, specifically for cooperative settings: decompose the *joint* Q-value into a combination of individual per-agent Q-values.

- **VDN (Value Decomposition Networks):** <cite index="93-1">simply sums each agent's individual Q-value to form the joint Q-value</cite> — simple, but assumes the team value is just the sum of individual contributions, which isn't always true.
- **QMIX:** <cite index="93-1">relaxes this by combining individual Q-values through a learned mixing network, constrained so the combination is monotonic in each agent's individual utility</cite> — meaning if any single agent's local Q-value improves, the joint Q-value can only improve too. This constraint (called the "Individual-Global-Max" or IGM property) is what makes it possible to have each agent act *greedily* on its own local Q-value at execution time, and still guarantee the resulting joint action is the one the centralized joint Q-value would have picked — solving the "how do decentralized agents replicate a centralized decision" problem cleanly, without needing a communication channel between them at test time.

### Scaling, Coordination, and Newer Paradigms

Beyond the CTDE/value-decomposition core, several other threads are active:

- **Communication and graph-based coordination** — agents learn what to communicate to each other (rather than being hand-designed a protocol), often using graph neural networks to model which agents' information is relevant to which others.
- **Population-based and game-theoretic methods** — training against a *population* of past or diverse opponents/teammates (self-play, league training) rather than a single fixed set of other agents, connecting MARL back to classical game theory (Nash equilibria, empirical game-theoretic analysis).
- **Mean-field RL** — for settings with *very* many agents (e.g. traffic systems), approximating the effect of all other agents as a single population density, rather than tracking each one individually — sidestepping the combinatorial joint-action-space problem entirely.
- **Offline MARL** — learning from a fixed, pre-collected multi-agent dataset without further environment interaction, the multi-agent analogue of offline single-agent RL.
- **Scaling to large agent teams and long horizons** — <cite index="14-1">recent surveys focus specifically on scaling MARL to large agent teams and long-horizon tasks</cite>, an increasingly practical concern as MARL moves from small benchmark environments toward real-world deployments with dozens or hundreds of agents.

### The Newest Thread: LLM-Based Multi-Agent RL

A fast-growing and somewhat distinct branch: instead of small neural network agents, each "agent" is an LLM, and multiple LLM agents must coordinate, communicate, or compete to solve a shared task. <cite index="24-1">Extending single-agent LLM-based RL to genuinely multi-agent settings isn't trivial, since coordination and communication between agents aren't considered at all in ordinary single-agent RL frameworks.</cite> Recent work in this space includes training LLM agents to negotiate, debate, or divide labor via RL objectives, and post-training multiple collaborating LLMs jointly with RL rather than training each in isolation.

### Benchmarks and Frameworks

If you want to get hands-on: <cite index="13-1">common current benchmarks include SMACv2, Melting Pot 2.0, Neural MMO 2.0, and MAgent2, with reproducible open-source frameworks such as EPyMARL, MARLlib, and JaxMARL</cite> making it feasible to run these algorithms without building everything from scratch.

---

## 2. Meta-Reinforcement Learning (Meta-RL)

Meta-RL asks a different question than everything above: instead of training one agent to solve one task well, can we train an agent that *learns how to learn* — so it can adapt quickly to a *new* task from the same family, using only a handful of episodes, rather than being retrained from scratch?

### The Core Idea

Meta-RL is organized around an **outer loop** and an **inner loop**: <cite index="40-1">the outer loop shapes the network's weights across many different tasks, while the inner loop is the fast adaptation that happens on a new task — and in many meta-RL settings, this inner-loop adaptation happens through memory and internal activation dynamics rather than through further weight updates</cite>. This is the key conceptual shift from ordinary RL, where "learning" always means updating weights via gradient descent — in meta-RL, "learning" during deployment can instead mean something that happens purely inside the network's activations.

### Two Classical Approaches

- **MAML (Model-Agnostic Meta-Learning):** <cite index="30-1,39-1">a general meta-learning method, applicable to RL by treating "model-agnostic meta-learning for fast adaptation of deep networks" as the outer-loop objective</cite> — the outer loop searches for a network *initialization* such that a small number of ordinary gradient steps on a new task lead to good performance on it. The inner loop here is explicit: real gradient updates, just very few of them.
- **RL² (RL-squared):** <cite index="28-1,39-1">described as "fast reinforcement learning via slow reinforcement learning"</cite> — instead of explicit inner-loop gradient steps, a recurrent (memory-based) policy is trained, via an ordinary RL algorithm, across many different tasks. The recurrent hidden state itself comes to encode what the agent has learned about the current task so far, purely through its dynamics as it accumulates experience within an episode — no inner-loop weight updates at all.

### The Current Frontier: In-Context Reinforcement Learning (ICRL)

The most active corner of meta-RL right now, and the one most directly connected to the rise of large transformer models: <cite index="34-1">the study of RL algorithms that operate purely at inference time dates back to RL² -style work, with the term "in-context reinforcement learning" coined more recently to describe this rapidly growing subfield</cite>.

The idea: train a large sequence model (usually a transformer) so that, at deployment, it can *improve its own behavior* purely by conditioning on the growing context of its own past states, actions, and rewards within the current task — with **zero gradient updates** at deployment. <cite index="37-1">Formally, the policy is conditioned on both the current state and a dynamic context — typically the historical trajectory seen so far — with actions sampled from $\pi_\theta(a_t \mid s_t, C_t)$</cite>. This is exactly the same phenomenon as in-context learning in LLMs (where a model "learns" a new task from examples in its prompt, without any weight updates) — except here the "examples" are the agent's own trial-and-error experience unfolding within a single episode.

- **Algorithm Distillation** is a key technique here: <cite index="28-1">rather than imitating a fixed dataset policy — which makes prior sequence-modeling approaches unsuitable for in-context RL on novel tasks — this method trains the model on the learning *history* of an actual RL algorithm, so the model distills the algorithm's own improvement pattern into its weights</cite>, letting it reproduce (and generalize) that improvement process in-context on new tasks it wasn't explicitly trained on.

### Meta-RL Meets LLM Agents

A very recent and notable development: applying meta-RL's core insight *to* LLM-based agents themselves, not just to small custom-built models. <cite index="32-1">One recent framework trains an LLM agent by maximizing a discounted *cross-episode* return, rather than a single episode's return — naturally teaching the agent when to explore versus exploit, so that the exploration strategies learned during training transfer into rapid in-context adaptation at test time.</cite> This is a genuinely new angle: rather than treating each interaction with an LLM agent as an independent episode, framing the *training itself* as a meta-RL problem across episodes, so the resulting agent arrives already equipped with good exploration habits, rather than having to rediscover them from scratch on every new task.

### Where This Is Heading

Meta-RL and ICRL are becoming less about small custom RL benchmarks and more about explaining and improving *emergent* in-context learning behavior in large models generally — a genuine convergence point between classical meta-RL theory and the practical training of LLM-based agents. If you found the policy gradient and actor-critic derivations satisfying, MAML is the most mathematically direct next step (it's literally a second-order gradient-through-gradient-steps optimization problem); if you're more interested in the LLM connection, ICRL and Algorithm Distillation are the more relevant thread.

---

## 3. RL and Reasoning Models

This is the area that's moved fastest since early 2025, and it connects directly back to your PPO and actor-critic notes — the core algorithm involved, GRPO, is best understood as "PPO, with the critic removed and replaced by a clever trick."

### The Shift: From RLHF to RLVR

Standard **RLHF** (Reinforcement Learning from Human Feedback) — the method historically used to align chat models — trains a separate **reward model** on human preference data, then optimizes the policy against that learned reward model using PPO. This requires holding *three* models in memory during training: the policy, a value/critic network, and the reward model (often all comparably sized to the LLM itself) — expensive in both memory and compute.

**RLVR (Reinforcement Learning with Verifiable Rewards)** sidesteps the reward-model half of this: instead of a *learned* reward model trained on subjective human preferences, use a reward that can be checked *automatically and unambiguously* — did the generated math answer match the ground truth, did the generated code pass its test cases, does the output follow the required format. <cite index="43-1">This approach, RLVR combined with GRPO, eliminates two expensive models from the training procedure at once: the reward model and the critic (value model).</cite>

### GRPO: Removing the Critic

**GRPO (Group Relative Policy Optimization)** was introduced in the DeepSeekMath paper and became central to DeepSeek-R1. The key idea, in terms you've already seen in your PPO and actor-critic notes: instead of learning a value function $V_\phi(s)$ to use as a baseline (as in REINFORCE with baseline, or as a critic in PPO), GRPO gets its baseline "for free" from a *group* of sampled responses.

<cite index="59-1">For a given input with its known correct answer, the current policy generates a group of $G$ sampled responses. Each response's advantage is then computed by normalizing its reward against the mean and standard deviation of rewards across that same group</cite>:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \dots, r_G\})}{\text{std}(\{r_1, \dots, r_G\}) + \delta}
$$

This is a direct substitute for the learned baseline in your REINFORCE-with-baseline and actor-critic notes — instead of training a separate critic network to estimate "what return is typical from this state," GRPO simply asks "was this particular response better or worse than the other attempts at the same question, sampled right now?" No critic network, no separate training loop for it.

The policy is then updated with the same clipped surrogate objective from your PPO notes, applied per-token, with an explicit KL penalty against a fixed reference policy baked directly into the objective (rather than handled via the separate constraint or adaptive-penalty schemes in your PPO notes):

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \left( \min\big(r_{i,t}(\theta)\hat{A}_{i,t},\, \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_{i,t}\big) - \beta D_{KL}(\pi_\theta \,\|\, \pi_{ref}) \right) \right]
$$

<cite index="65-1">where the probability ratio $r_{i,t}(\theta)$ compares the new and old policy's probability of generating each token, exactly as in ordinary PPO.</cite> If you followed the PPO-clipped derivation in your earlier notes, this equation should look immediately familiar — it's the same clipped objective, with a group-relative advantage swapping in for a learned critic, and the KL penalty term folded directly into the loss rather than treated as a separate constraint.

### DeepSeek-R1: What Happened, and Why It Was Surprising

<cite index="56-1">DeepSeek-R1-Zero was trained using GRPO with a reward based solely on the correctness of the final answer against ground truth, without any constraints on the reasoning process itself — and, notably, without any supervised fine-tuning step beforehand at all.</cite> <cite index="55-1">The reward function combined accuracy (was the final answer correct) with a format reward (requiring the reasoning process to be wrapped in specific tags).</cite>

What made this notable: <cite index="46-1">sophisticated reasoning behaviors — self-verification, reflection, and dynamic strategy exploration — emerged organically during training without being explicitly instructed, including a widely-discussed "aha moment" where the model spontaneously learned to pause and re-evaluate its own reasoning mid-generation.</cite> None of this chain-of-thought behavior was demonstrated to the model beforehand — it emerged purely from optimizing a simple correctness signal via RL.

The full **DeepSeek-R1** (as opposed to R1-Zero) addressed some practical rough edges — R1-Zero's outputs suffered from <cite index="46-1">poor readability and mixing of languages within a single response</cite> — <cite index="46-1">by adopting a multi-stage pipeline alternating supervised fine-tuning on curated long chain-of-thought examples with further rounds of reasoning-focused RL</cite>.

### Beyond Math and Code: Agentic RL

The RLVR + GRPO recipe has since spread well beyond its original math/coding domains — into medical question answering, visual and video reasoning, and, most actively right now, **agentic RL**: extending this same RL-for-reasoning approach from single-turn problems (one question, one verifiable answer) to genuinely **multi-turn agents** that interleave reasoning with real tool calls — web search, code execution, file or browser interaction — across long horizons.

<cite index="75-1">In these settings, a multi-turn agent is naturally formulated as a POMDP, where the agent produces actions through tool calls and receives environment observations at each step</cite> — directly the same partially-observable framework from your very first foundations notes, now applied at the scale of an LLM making dozens or hundreds of tool calls per task. Notable systems in this space include agents trained to interleave web search with reasoning, tools for strategic code-execution use, and RL applied directly to software-engineering tasks.

### Open Challenges (as of mid-2026)

A few problems the field is actively working through right now:

- **Long-horizon credit assignment.** <cite index="85-1">When an agentic rollout spans 100+ turns and up to a million tokens, but the only reward signal is a single outcome-level judgment at the very end, figuring out which specific actions deserved credit or blame becomes increasingly difficult</cite> — this is the same credit-assignment problem from your foundations notes, just at a scale several orders of magnitude larger than anything in classical RL.
- **Reward hacking.** Purely rule-based/verifiable rewards can still be gamed in unexpected ways (e.g. a model learning to produce outputs that satisfy the letter of a format check without genuinely reasoning) — an ongoing concern as RLVR is applied more broadly.
- **Generalizing beyond verifiable domains.** Math and code have a rare property: correctness is cheap and unambiguous to check automatically. Most real-world tasks don't have this property, so extending RLVR's strong results to more subjective domains (an area sometimes explored via rubric-based or AI-judged rewards) is an active area bridging back toward classical RLHF-style approaches.
- **Training stability at scale.** Long RL training runs on reasoning models remain sensitive to hyperparameters and prone to instability over extended training — an engineering-heavy area of ongoing work, distinct from (but related to) the trust-region concerns you saw in TRPO/PPO.

---

## Where to Go From Here

Given what you've already built up in these notes, here's how each frontier connects back and what it would take to go deeper:

- **MARL** is the most direct extension of what you already know — MADDPG and MAPPO are literally your DDPG and PPO notes with a centralized critic bolted on. If you want to continue the pattern of these notes (careful derivations, full algorithms), this is the most natural next chapter.
- **Meta-RL / ICRL** is more conceptually distinct — it's less about a new update rule and more about a different notion of what "learning" means at deployment time. MAML is the more classical, derivation-friendly entry point; ICRL is the more research-frontier, LLM-adjacent one.
- **RL for reasoning models** is the most immediately "hot" topic and the one most likely to keep changing fast — but it's also the one that reuses the most machinery you've already built (PPO's clipped objective, KL trust regions, REINFORCE-style baselines), just reassembled around LLM-specific constraints (per-token ratios, verifiable rewards, group-relative advantages instead of a learned critic).