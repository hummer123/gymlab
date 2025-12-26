import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    def __init__(self, K):
        self.K = int(K)
        # 随机生成K个 0～1 之间的数，作为每个臂的成功概率
        self.probs = np.atleast_1d(np.random.uniform(size=self.K))

        self.best_idx = int(np.argmax(self.probs))
        self.best_prob = self.probs[self.best_idx]
        print(f"BernoulliBandit initialized with probs: {self.probs}")

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"Randomly generated probabilities for {K} arms")
print(f"Best arm index: {bandit_10_arm.best_idx}, with probability: {bandit_10_arm.best_prob}")


class Solver:
    '''
    Solver Framework
    '''
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每个臂被选择的次数
        self.regret = 0  # 累计遗憾值
        self.actions = []
        self.regrets = []

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self):
        raise NotImplementedError
    
    def run(self, num_steps):
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob]*self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            # print(f"    --> Exploration, epsilon={self.epsilon}")
            k = np.random.randint(0, self.bandit.K) # 探索
        else:
            k = int(np.argmax(self.estimates))      # 利用
            # print(f"  => Exploitation, k={k}")

        reward = self.bandit.step(k)
        # 更新估计值
        # Q_{n}   = {R_1 + R_2 + ... + R_n} / n
        # Q_{n+1} = {R_1 + R_2 + ... + R_n + R_{n+1}} / (n+1)
        #         = [n * Q_{n} + R_{n+1}] / (n+1)
        #         = Q_{n} + 1/(n+1) * (R_{n+1} - Q_{n})
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])
        # print(f"Updated estimate: {self.estimates}, reward: {reward}")
        return k 


def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time Steps')
    plt.ylabel('Cumulative Regrets')
    plt.title('%d-Armed Bernoulli Bandit Problem' % solvers[0].bandit.K)
    plt.legend()
    plt.show()


############# main 
np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(num_steps=5000)
print(f"Epsilon-Greedy sum regret: {epsilon_greedy_solver.regret}")
# print(f"Epsilon-Greedy list regrets: {epsilon_greedy_solver.regrets}")
plot_results([epsilon_greedy_solver], ['Epsilon-Greedy'])


np.random.seed(0)
epsilon = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilon]
epsilon_greedy_solver_names = [f"epsilon={e}" for e in epsilon]
for solver in epsilon_greedy_solver_list:
    solver.run(num_steps=5000)

print("Epsilon-Greedy list regrets:", [solver.regret for solver in epsilon_greedy_solver_list])
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob]*self.bandit.K)

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < (1 / self.total_count):
            k = np.random.randint(0, self.bandit.K) # 探索
        else:
            k = int(np.argmax(self.estimates))      # 利用

        reward = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])

        return k

np.random.seed(1)
decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
decaying_epsilon_greedy_solver.run(num_steps=5000)
print(f"Decaying Epsilon-Greedy sum regret: {decaying_epsilon_greedy_solver.regret}")
plot_results([decaying_epsilon_greedy_solver], ['Decaying Epsilon-Greedy']) 



class UCB(Solver):
    def __init__(self, bandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob]*self.bandit.K)
        self.coef = coef

    def run_one_step(self):
        self.total_count += 1
        ucb_values = self.estimates + self.coef * np.sqrt(np.log(self.total_count) / (2 * (self.counts + 1)))
        k = int(np.argmax(ucb_values))

        reward = self.bandit.step(k)
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])

        return k


np.random.seed(1)
ucb_solver = UCB(bandit_10_arm, coef=0.5)
ucb_solver.run(num_steps=5000)
print(f"UCB sum regret: {ucb_solver.regret}")
plot_results([ucb_solver], ['UCB'])



class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.zeros(self.bandit.K)
        self._b = np.zeros(self.bandit.K)

    def run_one_step(self):
        samples = np.random.beta(self._a + 1, self._b + 1)
        k = int(np.argmax(samples))
        reward = self.bandit.step(k)

        self._a[k] += reward
        self._b[k] += (1 - reward)

        return k


np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(num_steps=5000)
print(f"Thompson Sampling sum regret: {thompson_sampling_solver.regret}")
plot_results([thompson_sampling_solver], ['Thompson Sampling'])



