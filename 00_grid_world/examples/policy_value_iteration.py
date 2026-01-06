import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np
import matplotlib.pyplot as plt


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


def mc_epsilon_greedy(env, gamma=0.9, epsilon=0.1, max_episodes=5000):
    '''
    MC epsilon-Greedy Algorithm
    '''
    # Initialization
    # Initial policy pi_0(a|s), all (s,a) initial values q(s,a)
    # Return(s,a) = 0 and Number(s,a) = 0 for all (s,a)
    n_states = env.num_states
    n_actions = len(env.action_space)
    
    Q = np.zeros((n_states, n_actions))
    return_sum = np.zeros((n_states, n_actions))
    return_count = np.zeros((n_states, n_actions))
    
    # Initialize policy to be epsilon-soft (uniform random is a valid start)
    pi = np.ones((n_states, n_actions)) / n_actions
    
    for episode_idx in range(max_episodes):
        # Generate episode
        # Execute current policy, generate episode of length T
        episode = []
        # state, _ = env.reset()
        # state_idx = env.xy_to_state_index(state)
        # ==> Randomly init the starting state, increase exploration
        state_idx = np.random.choice(env.num_states)
        env.agent_state = env.state_index_to_xy(state_idx)
        env.traj = [env.agent_state] 

        # We assume the episode terminates. Add a safety limit.
        step_limit = 1000
        for _ in range(step_limit):
            # Choose action based on policy pi
            action_idx = np.random.choice(n_actions, p=pi[state_idx])
            action = env.action_space[action_idx]

            next_state, reward, done, _ = env.step(action)
            episode.append((state_idx, action_idx, reward))
            env.render(animation_interval=0.001)

            if done:    
                break

            state = next_state
            state_idx = env.xy_to_state_index(state)
            
        # Initialize g <- 0
        G = 0
        
        # Loop for each step of episode, t = T-1, T-2, ..., 0
        for t in range(len(episode) - 1, -1, -1):
            s_t, a_t, r_next = episode[t]
            
            # g <- gamma * g + r_{t+1}
            G = gamma * G + r_next
            return_sum[s_t, a_t] += G
            return_count[s_t, a_t] += 1
            
            # Policy Evaluation:
            # q(s_t, a_t) <- Return(s_t, a_t) / Number(s_t, a_t)
            Q[s_t, a_t] = return_sum[s_t, a_t] / return_count[s_t, a_t]
            
            # Policy Improvement:
            # a* = argmax_a q(s_t, a)
            best_a = np.argmax(Q[s_t])
            
            # Update pi(a|s_t) = {
            #     1 - epsilon + epsilon / |A(s_t)|, if a = a*
            #     epsilon / |A(s_t)|,               if a != a*
            # }
            pi[s_t, :] = epsilon / n_actions
            pi[s_t, best_a] += 1 - epsilon
        
        # Visualization and logging
        print(f"Episode {episode_idx + 1}/{max_episodes}")
    
    # Show the final policy to a greedy (deterministic) policy: choose the highest-probability action in each state and set other actions' probabilities to 0
    input(f"Show the final epsilon-Greedy policy...")

    env.traj = [env.agent_state] 
    greedy_pi = np.zeros_like(pi)
    best_actions = np.argmax(pi, axis=1)
    greedy_pi[np.arange(n_states), best_actions] = 1.0
    env.add_policy(greedy_pi)

    # V(s) = max_a Q(s, a) for visualization
    V = np.max(Q, axis=1)
    env.add_state_values(V)
    env.render(animation_interval=0.02) # Faster render during training

    return pi, Q


def sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, max_episodes=5000, headless=True):
    '''
    Sarsa Algorithm Implementation
    '''
    n_states = env.num_states
    n_actions = len(env.action_space)

    # Initialize Q(s, a)
    Q = np.zeros((n_states, n_actions))
    pi = np.ones((n_states, n_actions)) / n_actions
    
    # 记录每回合的总奖励和回合长度
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(max_episodes):
        # ==> Randomly init the starting state, increase exploration
        # state_idx = np.random.choice(env.num_states)
        state_idx = 0
        env.agent_state = env.state_index_to_xy(state_idx)
        env.traj = [env.agent_state] 
        
        action_idx = np.random.choice(n_actions, p=pi[state_idx])
        
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action = env.action_space[action_idx]
            
            # Take action A, observe R, S'
            next_state, reward, done, _ = env.step(action)
            next_state_idx = env.xy_to_state_index(next_state)

            if not headless:
                env.render(animation_interval=0.001)

            total_reward += reward
            step_count += 1

            next_action_idx = np.random.choice(n_actions, p=pi[next_state_idx])

            # Update Q(S, A) 
            # Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
            target = reward + gamma * Q[next_state_idx, next_action_idx]
            Q[state_idx, action_idx] += alpha * (target - Q[state_idx, action_idx])

            # Update pi(a|s_t) = {
            #     1 - epsilon + epsilon / |A(s_t)|, if a = a*
            #     epsilon / |A(s_t)|,               if a != a*
            # }
            best_a = np.argmax(Q[state_idx])
            pi[state_idx, :] = epsilon / n_actions
            pi[state_idx, best_a] += 1 - epsilon

            state_idx = next_state_idx
            action_idx = next_action_idx
        
        episode_rewards.append(total_reward)
        episode_lengths.append(step_count)
        print(f"Sarsa Episode {episode + 1}/{max_episodes}, Reward: {total_reward}, Length: {step_count}")
            

    print("Sarsa training finished.")
    
    # 绘制回合次数图
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Total reward vs Episode
    axes[0].plot(range(max_episodes), episode_rewards, color='gray', linewidth=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('Sarsa Training Curve')
    
    # Episode length vs Episode
    axes[1].plot(range(max_episodes), episode_lengths, color='gray', linewidth=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('../plots/sarsa_training_curve.png', dpi=150)
    
    # Show the final policy (Greedy)
    greedy_pi = np.zeros_like(pi)
    best_actions = np.argmax(pi, axis=1)
    greedy_pi[np.arange(n_states), best_actions] = 1.0

    # V(s) = max_a Q(s, a) for visualization
    V = np.max(Q, axis=1)
    
    input("Press Enter to show the final Sarsa policy...")
    env.agent_state = env.start_state
    env.traj = [env.agent_state]
    env.add_policy(greedy_pi)
    env.add_state_values(V)
    env.render()

    return greedy_pi, Q


# Main function
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)

    env = GridWorld()
    state = env.reset()
    env.render()

    # policy_iteration(env, gamma=0.9, theta=1e-6, max_it=100)
    # input('===> End of Policy Iteration...')
    
    # value_iteration(env, gamma=0.9, theta=1e-6, max_it=100)
    # input('===> End of Value Iteration...')

    # truncated_policy_iteration(env, gamma=0.9, theta=1e-6, max_it=100, eval_it=5)
    # input('===> End of Truncated Policy Iteration...')

    # print("Starting MC epsilon-Greedy...")
    # pi, Q = mc_epsilon_greedy(env, gamma=0.9, epsilon=0.2, max_episodes=5000)
    # input('===> End of MC epsilon-Greedy...')

    # Run Sarsa
    print("Starting Sarsa...")
    pi, Q = sarsa(env, gamma=0.9, alpha=0.1, epsilon=0.1, max_episodes=500)
    input('===> End of Sarsa...')


