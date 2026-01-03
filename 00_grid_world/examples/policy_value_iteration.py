import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np


def policy_iteration(env, gamma=0.9, theta=1e-6, max_it=1000):
    '''
    在给定环境上执行策略迭代算法（Policy Iteration）。

    算法步骤：
    1. 随机初始化策略（这里用随机概率矩阵并归一化）
    2. 迭代执行策略评估（Policy Evaluation）直到值函数收敛
    3. 基于当前值函数进行策略改进（Policy Improvement）
    4. 重复直到策略稳定或达到最大迭代次数

    :param env: 环境对象（需提供属性 `num_states`, `action_space`，以及方法
                            `state_index_to_xy`, `xy_to_state_index`, `_get_next_state_and_reward`）
    :param gamma: 折扣因子 (float)，用于累计未来奖励 (0 <= gamma <= 1)
    :param theta: 策略评估的收敛阈值 (float)。当值函数最大更新幅度小于该值时停止评估
    :param max_it: 最大迭代次数 (int)，用于策略改进外层和策略评估内层的上限
    :return: (pi_matrix, V)
                        - pi_matrix: numpy.ndarray，形状为 `(num_states, n_actions)` 的策略矩阵，
                            每行表示该状态下针对每个动作的概率（确定性策略对应 one-hot）
                        - V: numpy.ndarray，形状为 `(num_states,)` 的状态值估计
    '''
    # stochastic policy initialization (keep original simple behavior)
    stochastic_matrix = np.random.rand(env.num_states, len(env.action_space))
    pi_matrix = stochastic_matrix / stochastic_matrix.sum(axis=1)[:, np.newaxis]

    # ensure `V` is always defined (in case max_it == 0 or loop exits early)
    V = np.zeros(env.num_states)

    for _iter in range(max_it):
        # Policy Evaluation (iterative until convergence)
        for _eval in range(max_it):
            delta = 0
            for s in range(env.num_states):
                state_xy = env.state_index_to_xy(s)
                new_v = 0.0
                # v_k+1 = sum_a pi(a|s) * [R_pi(a) + gamma * P_pi(a) V_k(s')] == bootstrap
                for a, action in enumerate(env.action_space):
                    # GridWorld is deterministic: P(s'|s,a)=1, so just use the single next state
                    next_state, reward = env._get_next_state_and_reward(state_xy, action)
                    s_next = env.xy_to_state_index(next_state)
                    new_v += pi_matrix[s, a] * (reward + gamma * V[s_next])

                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(env.num_states):
            old_action = np.argmax(pi_matrix[s])
            action_values = np.zeros(len(env.action_space))
            state_xy = env.state_index_to_xy(s)
            # q_pi(s,a) = R_pi(a) + gamma * P_pi(a) V(s') == one-step lookahead
            # a* = argmax_a q_pi(s,a)
            for a, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state_xy, action)
                s_next = env.xy_to_state_index(next_state)
                action_values[a] = reward + gamma * V[s_next]
            best_action = np.argmax(action_values)

            if old_action != best_action:
                policy_stable = False

            pi_matrix[s] = np.eye(len(env.action_space))[best_action]

        print(f"Iteration {_iter}:")
        env.add_policy(pi_matrix)
        env.add_state_values(V)
        env.render()
        if policy_stable:
            print("Policy converged.")
            break

    return pi_matrix, V


def value_iteration(env, gamma=0.9, theta=1e-6, max_it=1000):
    '''
    值迭代算法
    '''
    n_actions = len(env.action_space)
    # 初始化: v0
    V = np.zeros(env.num_states)
    pi_matrix = np.zeros((env.num_states, n_actions))

    # 当 vk 尚未收敛时
    for _iter in range(max_it):
        delta = 0.0
        # 注意：这里不再需要 V_new = np.copy(V)
        
        # 对每个状态 s
        for s in range(env.num_states):
            state_xy = env.state_index_to_xy(s)
            q_values = np.zeros(n_actions)
            
            # 对每个动作 a
            # 计算 q 值: qk(s, a)
            for a, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state_xy, action)
                s_next = env.xy_to_state_index(next_state)
                # 关键点：这里用到的 V[s_next] 可能是本轮刚刚更新过的（如果 s_next < s），
                # 也可能是上一轮的（如果 s_next > s）。信息的传播速度更快。
                q_values[a] = reward + gamma * V[s_next]
            
            # 最大价值动作: ak*(s)
            best_action = np.argmax(q_values)
            
            # 策略更新: pi_k+1
            pi_matrix[s] = np.eye(n_actions)[best_action]

            # 值更新: vk+1(s), v_{k+1} = max_a qk(s, a)
            v_new_s = np.max(q_values)
            delta = max(delta, abs(v_new_s - V[s]))
            
            # 关键点：直接原地更新
            V[s] = v_new_s

        print(f"Iteration {_iter} ---> delta: {delta}")
        env.add_policy(pi_matrix)
        env.add_state_values(V)
        env.render()

        if delta < theta:
            print(f"Value function converged.")
            break

    return pi_matrix, V


def truncated_policy_iteration(env, gamma=0.9, theta=1e-6, max_it=1000, eval_it=5):
    '''
    截断策略迭代算法 (Truncated Policy Iteration)
    
    与标准策略迭代的区别在于：策略评估步骤不等到完全收敛，而是固定迭代一定次数 (eval_it)。
    这是一种介于值迭代（eval_it=1）和策略迭代（eval_it=inf）之间的方法。

    :param env: 环境对象
    :param gamma: 折扣因子
    :param theta: 策略评估的收敛阈值
    :param max_it: 最大迭代次数
    :param eval_it: 策略评估阶段的最大迭代次数
    :return: (pi_matrix, V)
    '''
    # stochastic policy initialization
    stochastic_matrix = np.random.rand(env.num_states, len(env.action_space))
    pi_matrix = stochastic_matrix / stochastic_matrix.sum(axis=1)[:, np.newaxis]

    V = np.zeros(env.num_states)

    for _iter in range(max_it):
        # Policy Evaluation (Truncated)
        for _eval in range(eval_it):
            delta = 0
            for s in range(env.num_states):
                state_xy = env.state_index_to_xy(s)
                new_v = 0.0
                for a, action in enumerate(env.action_space):
                    next_state, reward = env._get_next_state_and_reward(state_xy, action)
                    s_next = env.xy_to_state_index(next_state)
                    new_v += pi_matrix[s, a] * (reward + gamma * V[s_next])

                delta = max(delta, abs(new_v - V[s]))
                V[s] = new_v
            
            # 即使是截断策略迭代，如果已经收敛也可以提前退出评估
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(env.num_states):
            old_action = np.argmax(pi_matrix[s])
            action_values = np.zeros(len(env.action_space))
            state_xy = env.state_index_to_xy(s)
            
            for a, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state_xy, action)
                s_next = env.xy_to_state_index(next_state)
                action_values[a] = reward + gamma * V[s_next]
            best_action = np.argmax(action_values)

            if old_action != best_action:
                policy_stable = False

            pi_matrix[s] = np.eye(len(env.action_space))[best_action]

        print(f"Iteration {_iter}:")
        env.add_policy(pi_matrix)
        env.add_state_values(V)
        env.render()
        if policy_stable:
            print("Policy converged.")
            break

    return pi_matrix, V


# Main function
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    env = GridWorld()
    state = env.reset()
    env.render()

    policy_iteration(env, gamma=0.9, theta=1e-6, max_it=100)
    input('===> End of Policy Iteration...')
    
    value_iteration(env, gamma=0.9, theta=1e-6, max_it=100)
    input('===> End of Value Iteration...')

    truncated_policy_iteration(env, gamma=0.9, theta=1e-6, max_it=100, eval_it=5)
    input('===> End of Truncated Policy Iteration...')


