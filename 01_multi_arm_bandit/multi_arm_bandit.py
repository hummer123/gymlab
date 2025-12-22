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
            k = np.random.randint(0, self.bandit.K) # 探索
        else:
            k = int(np.argmax(self.estimates))      # 利用

        reward = self.bandit.step(k)
        # 更新估计值
        self.estimates[k] += 1.0 / (self.counts[k] + 1) * (reward - self.estimates[k])
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
plot_results([epsilon_greedy_solver], ['Epsilon-Greedy'])


np.random.seed(0)
epsilon = [1e-4, 0.01, 0.1, 0.25, 0.5]
epsilon_greedy_solver_list = [EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilon]
epsilon_greedy_solver_names = [f"epsilon={e}" for e in epsilon]
for solver in epsilon_greedy_solver_list:
    solver.run(num_steps=5000)
plot_results(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

