from bandits import BanditEnv
import numpy as np


class Algo:
    def apply(self, env, Q, count, sum_rewards, num_rounds):
        """Runs num_rounds of pulls, mutating Q, count, sum_rewards in place."""
        raise NotImplementedError("Subclasses must implement apply()")


class EpsilonGreedy(Algo):
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def apply(self, env, Q, count, sum_rewards, num_rounds):
        for i in range(num_rounds):
            if np.random.uniform(0, 1) < self.epsilon:
                arm = env.action_space.sample()  # explore
            else:
                arm = np.argmax(Q)  # exploit

            _, reward, _, _, _ = env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]


class Softmax(Algo):
    def __init__(self, temperature=50, reduction=0.99, constant=False):
        self.temperature = temperature
        self.reduction = reduction
        self.constant = constant

    def apply(self, env, Q, count, sum_rewards, num_rounds):
        T = self.temperature

        for i in range(num_rounds):
            denom = sum([np.exp(q / T) for q in Q])
            probs = [np.exp(q / T) / denom for q in Q]
            arm = np.random.choice(env.action_space.n, p=probs)

            _, reward, _, _, _ = env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]

            if not self.constant:
                T = T * self.reduction


class UCB(Algo):
    def apply(self, env, Q, count, sum_rewards, num_rounds):
        n_arms = env.action_space.n

        for i in range(num_rounds):
            ucb = np.zeros(n_arms)
            for arm in range(n_arms):
                if count[arm] == 0:
                    ucb[arm] = float('inf')
                else:
                    ucb[arm] = Q[arm] + np.sqrt((2 * np.log(sum(count))) / count[arm])

            arm = np.argmax(ucb)
            _, reward, _, _, _ = env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]


class ThompsonSampling(Algo):
    def __init__(self, alpha=1, beta=1):
        self.init_alpha = alpha
        self.init_beta = beta

    def apply(self, env, Q, count, sum_rewards, num_rounds):
        n_arms = env.action_space.n
        alpha = np.full(n_arms, self.init_alpha, dtype=float)
        beta = np.full(n_arms, self.init_beta, dtype=float)

        for i in range(num_rounds):
            samples = [np.random.beta(alpha[arm], beta[arm]) for arm in range(n_arms)]
            arm = np.argmax(samples)

            _, reward, _, _, _ = env.step(arm)
            count[arm] += 1
            sum_rewards[arm] += reward
            Q[arm] = sum_rewards[arm] / count[arm]

            if reward == 1:
                alpha[arm] += 1
            else:
                beta[arm] += 1


class BanditSolver:
    """
    env: k-armed bandit environment
    algorithm: exploration-exploitation algorithm to use. ['epsilon-greedy', 'ucb', 'thompson', 'softmax']
    epsilon: value of epsilon to use in epsilon-greedy algorithm. [0, 1]
    temperature: temperature value to use in softmax
    reduction: how much to decrease the temperature value by in softmax
    constant: whether to use constant temperature or not
    alpha: alpha value to use in thompson
    beta: beta value to use in thompson
    """

    def __init__(self,
                 env: BanditEnv,
                 algorithm='',
                 epsilon=0.1,
                 temperature=50,
                 reduction=0.99,
                 constant=False,
                 alpha=1,
                 beta=1):

        self.env = env
        self.num_arms = env.action_space.n
        self.count = np.zeros(self.num_arms)
        self.sum_rewards = np.zeros(self.num_arms)
        self.Q = np.zeros(self.num_arms)

        self.algorithm = self._init_algo(
            algorithm,
            epsilon=epsilon,
            temperature=temperature,
            reduction=reduction,
            constant=constant,
            alpha=alpha,
            beta=beta,
        )

    def _init_algo(self, algorithm, **parameters) -> Algo:
        algorithm = algorithm.lower()

        if algorithm == 'epsilon-greedy':
            return EpsilonGreedy(epsilon=parameters['epsilon'])
        elif algorithm == 'ucb':
            return UCB()
        elif algorithm == 'thompson':
            return ThompsonSampling(alpha=parameters['alpha'], beta=parameters['beta'])
        elif algorithm == 'softmax':
            return Softmax(
                temperature=parameters['temperature'],
                reduction=parameters['reduction'],
                constant=parameters['constant'],
            )
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. "
                "Choose from ['epsilon-greedy', 'ucb', 'thompson', 'softmax']."
            )

    def pull(self, num_rounds=100):
        self.env.reset()
        self.algorithm.apply(self.env, self.Q, self.count, self.sum_rewards, num_rounds)
        return self.Q

    def get_optimal_arm(self):
        return np.argmax(self.Q)