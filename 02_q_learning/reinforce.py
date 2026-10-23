import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class PolicyNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class REINFORCEAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, device):
        self.policy_network = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # make state shape: (1, state_dim)
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action_probs = self.policy_network(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transitions_dict):
        # 切换以下函数调用来对比效果
        # self.update_naive(transitions_dict)        # 方案A：原始版本（效果差，不稳定）
        self.update_normalized(transitions_dict)   # 方案B：优化版本（Baseline + Batch，效果好）

    def update_native(self, transitions_dict):
        state_list  = transitions_dict['states']
        action_list = transitions_dict['actions']
        reward_list = transitions_dict['rewards']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            # state: (1, state_dim), action: (1, 1)
            state = torch.as_tensor(state_list[i], dtype=torch.float, device=self.device).unsqueeze(0)
            action = torch.tensor([action_list[i]], dtype=torch.int64, device=self.device).unsqueeze(1)
            
            # 策略网络的目标是：最大化回报， 即 sum(gamma^t * R)
            # 球最大化 sum(gamma^t * R), 所以求导时是：最小化 -sum(gamma^t * R)，即 loss = -log π(a|s) * G
            # loss = -log π(a|s) * G
            probs = self.policy_network(state)
            log_prob = torch.log(probs.gather(1, action))
            G  = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

    def update_normalized(self, transitions_dict):
        state_list = transitions_dict['states']
        action_list = transitions_dict['actions']
        reward_list = transitions_dict['rewards']

        G = 0
        self.optimizer.zero_grad()
        return_list = []
        # 计算每一步的回报 G
        for i in reversed(range(len(reward_list))):
            reward = reward_list[i]
            G = self.gamma * G + reward
            return_list.append(G)
        return_list.reverse()

        # [关键优化] 标准化回报：(G - mean) / std
        # 这能显著降低方差，防止 Reward 数值过大导致梯度爆炸或不稳定
        returns = torch.tensor(return_list, dtype=torch.float, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 批量转换状态和动作为张量
        states = torch.tensor(np.array(state_list), dtype=torch.float, device=self.device)
        actions = torch.tensor(action_list, dtype=torch.int64, device=self.device).view(-1, 1)

        # 一次性计算所有步的概率
        probs = self.policy_network(states)
        log_probs = torch.log(probs.gather(1, actions))
        
        # 计算损失 loss = -sum(log_prob * normalized_G)
        loss = -torch.sum(log_probs * returns.view(-1, 1))
        loss.backward()
        self.optimizer.step()



# def train_agent(env, agent, num_episodes):
#     return_list = []
#     for i in range(10):
#         with tqdm(total=int(num_episodes / 10), desc=f"Iteration: {i + 1}", unit="ep", dynamic_ncols=True) as pbar:
#             for ep in range(int(num_episodes / 10)):
#                 state, _ = env.reset()
#                 transitions_dict = {'states': [], 'actions': [], 'rewards': []}
#                 episode_reward = 0
#                 done = False

#                 while not done:
#                     action = agent.take_action(state)
#                     next_state, reward, terminated, truncated, _ = env.step(action)
#                     done = terminated or truncated

#                     transitions_dict['states'].append(state)
#                     transitions_dict['actions'].append(action)
#                     transitions_dict['rewards'].append(reward)

#                     state = next_state
#                     episode_reward += reward

#                 agent.update(transitions_dict)
#                 return_list.append(episode_reward)
#                 if (ep + 1) % 50 == 0:
#                     avg_return = np.mean(return_list[-50:])
#                     pbar.set_postfix({'Avg Return': f'{avg_return:.3f}'})
#                 pbar.update(1)

#     return return_list


# def evaluate_agent(env, agent, n_episodes=100, render=False):
#     rewards = []
#     for _ in range(n_episodes):
#         state, _ = env.reset()
#         episode_reward = 0
#         done = False

#         while not done:
#             if render:
#                 env.render()
#             action = agent.take_action(state)
#             next_state, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             state = next_state
#             episode_reward += reward

#         rewards.append(episode_reward)

#     avg_reward = np.mean(rewards)
#     print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
#     return rewards



##------------------------------##

if __name__ == "__main__":

    params = {
        'gamma': 0.98,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'num_episodes': 1000
    }
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = gym.make('CartPole-v1')
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Box)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(state_dim, params['hidden_dim'], action_dim,
                     learning_rate=params['learning_rate'],
                     gamma=params['gamma'],
                     device=device)

    return_list = []
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes=params['num_episodes'])
    env.close()

    episode_list = np.arange(len(return_list))
    mv_return_list = rl_utils.moving_average(return_list, window_size=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(episode_list, return_list)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('REINFORCE on CartPole-v1')

    axes[1].plot(episode_list, mv_return_list)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Moving Average Return')
    axes[1].set_title('REINFORCE on CartPole-v1 (Moving Average)')
    plt.tight_layout()
    plt.show()

    show_env = gym.make('CartPole-v1', render_mode='human')
    rl_utils.evaluate_on_policy_agent(show_env, agent, n_episodes=1, render=True)
    show_env.close()

