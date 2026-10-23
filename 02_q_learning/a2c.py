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


class ValueNetwork(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ActorCriticAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        self.actor  = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNetwork(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        # make state shape: (1, state_dim)
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transitions_dict):
        state_list  = transitions_dict['states']
        action_list = transitions_dict['actions']
        reward_list = transitions_dict['rewards']
        next_state_list = transitions_dict['next_states']
        done_list   = transitions_dict['dones']

        # convert lists of numpy arrays to a single numpy array to avoid slow tensor creation
        states = torch.as_tensor(np.array(state_list), dtype=torch.float32, device=self.device)
        actions = torch.tensor(action_list, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(reward_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.tensor(done_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(next_state_list), dtype=torch.float32, device=self.device)

        # 只计算一次 critic(states)，防止重复前向计算
        values = self.critic(states)
        # target = r + γ * V(s') * (1 - done)
        td_targets = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        # critic loss = [r + γ * V(s') - V(s)]^2
        critic_loss = F.mse_loss(values, td_targets.detach())

        log_probs = torch.log(self.actor(states).gather(1, actions) + 1e-8)
        # td = r + γ * V(s') - V(s)
        td_delta = td_targets - values
        # actor loss = -log π(a|s) * TD
        actor_loss = torch.mean(-log_probs * td_delta.detach())

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()





##------------------------------##

if __name__ == "__main__":

    params = {
        'gamma': 0.98,
        'actor_lr': 1e-3,
        'critic_lr': 1e-2,
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

    agent = ActorCriticAgent(state_dim, params['hidden_dim'], action_dim,
                     actor_lr=params['actor_lr'],
                     critic_lr=params['critic_lr'],
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
    axes[0].set_title('Actor-Critic on CartPole-v1')

    axes[1].plot(episode_list, mv_return_list)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Moving Average Return')
    axes[1].set_title('Actor-Critic on CartPole-v1 (Moving Average)')
    plt.tight_layout()
    plt.show()

    show_env = gym.make('CartPole-v1', render_mode='human')
    rl_utils.evaluate_on_policy_agent(show_env, agent, n_episodes=1, render=True)
    show_env.close()

