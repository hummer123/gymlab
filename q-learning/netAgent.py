import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
import os

import time
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Neural Network for Q-learning
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


class baseQNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.q_net = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg['lr'])
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = cfg['batch_size']
        self.gamma = cfg['gamma']
        self.epsilon = cfg['epsilon']
        self.epsilon_decay = cfg['epsilon_decay']
        self.epsilon_min = cfg['epsilon_min']
        self.best_avg_reward = 0

    def choose_action(self, state, train=False):
        if train:
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            
            if np.random.rand() < self.epsilon:
                return np.random.randint(0, 2) # Explore
        
        # Exploit: use Q-network for both train and eval modes
        state_tensor = torch.FloatTensor(state)
        q_values = self.q_net(state_tensor)
        return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        pass # To be implemented in subclasses

    def save_model(self, path='./model/best_q_model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        torch.save(self.q_net.state_dict(), path)
        print(f" -> Model saved to {path}")

    def load_model(self, path='./model/best_q_model.pth'):
        self.q_net.load_state_dict(torch.load(path, weights_only=True), strict=True)
        print(f" -> Model loaded from {path}")

    def setBestAvgReward(self, reward):
        self.best_avg_reward = reward
    
    def getBestAvgReward(self):
        return self.best_avg_reward
    
    def getEpsilon(self):
        return self.epsilon

    # 输出 评估结果：平均奖励
    def evaluate(self, env, eval_episodes=5):
        total_rewards = []

        with torch.no_grad(): # 评估时不需要计算梯度
            for _ in range(eval_episodes):
                t0 = time.time()
                state = env.reset()[0]
                episode_reward = 0
                while True:
                    action = self.choose_action(state, train=False)
                    # new_observation, reward, terminated, truncated, info
                    next_state, reward, done, _, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    if done or episode_reward > 500:
                        break

                total_rewards.append(episode_reward)
                print(f"Evaluate | Episode Reward: {episode_reward}, Duration: {time.time() - t0:.2f}s")

        return np.mean(total_rewards)



# DQN Agent
class DQNAgent(baseQNAgent):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim, action_dim, cfg)
        self.target_net = QNetwork(state_dim, action_dim)
        self.update_target_freq = cfg['update_target_freq'] # Update target network every 100 steps
        self.step_count = 0
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute current Q values 
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # DQN的目标：让 Q(s,a) 接近 r + γ * max Q(s', a')
        #                 ↑              ↑
        #             current_q      target_q
        # 损失越小 = Q值预测越准确 = 策略越好
        # Compute loss, === loss = target_q - current_q
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.q_net.state_dict().items()
            })

# 测试仅使用一个Q网络的DQNAgent
class BadQNAgent(baseQNAgent):
    def __init__(self, state_dim, action_dim, cfg):
        super().__init__(state_dim, action_dim, cfg)
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Compute current Q values 
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values
        with torch.no_grad():
            # ❌ 错误:使用同一个网络计算 target Q 值
            next_q = self.q_net(next_states).max(1)[0]  
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # DQN的目标：让 Q(s,a) 接近 r + γ * max Q(s', a')
        #                 ↑              ↑
        #             current_q      target_q
        # 损失越小 = Q值预测越准确 = 策略越好
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

