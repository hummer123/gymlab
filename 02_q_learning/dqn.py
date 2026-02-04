import random
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils


class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQNAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update_freq, device):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update_freq = target_update_freq
        self.device = device

        self.q_network = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_network = QNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.learn_step_counter = 0

    def take_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.argmax().item()

    def max_q_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return q_values.max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 让 q_network 逼近 target_q_network
        # target: y = r + γ max_a' Q_target(s', a')  (if not done)
        q_values = self.q_network(states).gather(1, actions)
        #----------------- DQN -----------------------#
        # with torch.no_grad():
            # next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            # target_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        #----------------- Double DQN ----------------#
        with torch.no_grad():
            max_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_q_network(next_states).gather(1, max_actions)
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 均方误差损失函数 
        # 最小化目标损失 L = 1/n * Σ(y - Q(s, a))^2
        # 也叫优势函数 A(s, a) = Q(s, a) - V(s)
        loss = torch.mean(F.mse_loss(q_values, target_q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())



def train_dqn(env, agent, replay_buffer, num_episodes, batch_size, minim_buffer_size):
    return_list = []

    epsilon_start = 0.98
    epsilon_end = 0.01
    epsilon_decay = 200
    total_episodes_counter = 0

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i + 1}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state, _ = env.reset()
                episode_reward = 0
                done = False

                # Linearly decay epsilon
                if total_episodes_counter < epsilon_decay:
                    agent.epsilon = epsilon_start - (epsilon_start - epsilon_end) * (total_episodes_counter / epsilon_decay)
                else:
                    agent.epsilon = epsilon_end
                total_episodes_counter += 1

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    # 重定义奖励函数：鼓励智能体保持静止和直立，减少无效抖动
                    # state: [x, x_dot, theta, theta_dot]
                    x, x_dot, theta, theta_dot = next_state
                    
                    # 惩罚偏离中心和倾斜
                    r_pos = -0.1 * abs(x)
                    r_ang = -0.1 * abs(theta)
                    # 惩罚速度（抑制抖动）
                    r_vel = -0.1 * abs(x_dot) - 0.1 * abs(theta_dot)
                    
                    # 综合奖励，保持原有生存奖励 1.0 的基础
                    train_reward = reward + r_pos + r_ang + r_vel
                    
                    replay_buffer.add(state, action, train_reward, next_state, done)
                    state = next_state
                    episode_reward += reward

                    if len(replay_buffer) >= minim_buffer_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_reward)
                pbar.update(1)
                if (i_episode + 1) % 50 == 0:
                    avg_reward = np.mean(return_list[-10:])
                    pbar.set_postfix({'Episode': f'{(i * (num_episodes // 10)) + i_episode + 1}',
                                      'Avg Reward': f'{avg_reward:.3f}'})
    return return_list



def show_agent(env, agent, max_steps=400):
    # Set epsilon to 0 for evaluation to ensure deterministic behavior
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    for i_step in range(max_steps):
        env.render()
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        total_reward += reward

        if done:
            break
            
    # Restore epsilon (good practice, though not strictly needed here)
    agent.epsilon = old_epsilon
    print(f"Total Reward: {total_reward}")

##------------------------------##

if __name__ == "__main__":

    params = {
        'epsilon': 0.01,
        'gamma': 0.98,
        'learning_rate': 1e-3,
        'hidden_dim': 128,
        'target_update_freq': 50,
        'buffer_capacity': 5000,
        'batch_size': 64,
        'minim_buffer_size': 1000,
        'num_episodes': 600
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

    agent = DQNAgent(state_dim, params['hidden_dim'], action_dim,
                     learning_rate=params['learning_rate'],
                     gamma=params['gamma'],
                     epsilon=params['epsilon'],
                     target_update_freq=params['target_update_freq'],
                     device=device)
    
    replay_buffer = rl_utils.ReplayBuffer(params['buffer_capacity'])
    return_list = []
    return_list = train_dqn(env, agent, replay_buffer,
                            num_episodes=params['num_episodes'],
                            batch_size=params['batch_size'],
                            minim_buffer_size=params['minim_buffer_size'])
    env.close()

    episode_list = np.arange(len(return_list))
    mv_return_list = rl_utils.moving_average(return_list, window_size=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(episode_list, return_list)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('DQN on CartPole-v1')

    axes[1].plot(episode_list, mv_return_list)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Moving Average Return')
    axes[1].set_title('DQN on CartPole-v1 (Moving Average)')
    plt.tight_layout()
    plt.show()

    show_env = gym.make('CartPole-v1', render_mode='human')
    show_agent(show_env, agent, max_steps=400)
    show_env.close()

    

