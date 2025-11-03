import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np
import os


class PGNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # hidden layer
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PGMemory:
    def __init__(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []

    def store(self, state, action, reward):
        self.state_pool.append(np.array(state, dtype=np.float32))
        self.action_pool.append(action)
        self.reward_pool.append(reward)

    def sample(self):
        return self.state_pool, self.action_pool, self.reward_pool

    def clear(self):
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []



class PolicyGradient:

    def __init__(self, model, cfg):
        self.gamma = cfg['gamma']
        self.device = torch.device(cfg['device'])
        self.memory = PGMemory()
        self.policy_net = model.to(self.device)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg['lr'])

    def sample_action(self, state):
        """训练时使用：从策略分布中随机采样动作（带探索）"""
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)  # 伯努里分布
        action = m.sample()

        action = action.data.numpy().astype(int)[0] # 转为标量
        return action 
    
    def predict_action(self, state):
        """评估时使用：选择概率最大的动作（确定性，无探索）"""
        state = torch.from_numpy(state).float()
        state = Variable(state)
        with torch.no_grad():  # 评估时不需要计算梯度
            probs = self.policy_net(state)
        # 对于二分类问题，概率 > 0.5 选择动作 1，否则选择动作 0
        action = (probs.data.numpy()[0] > 0.5).astype(int)
        return action
    
    def store_transition(self, state, action, reward):
        self.memory.store(state, action, reward)

    def learn(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(state_pool), list(action_pool), list(reward_pool)

        # 计算每个时间点的累计奖励（回报）- 从后往前计算折扣回报
        discounted_rewards = []
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            running_add = running_add * self.gamma + reward_pool[i]
            discounted_rewards.insert(0, running_add)
        
        # 标准化奖励 - 使训练更稳定
        discounted_rewards = np.array(discounted_rewards)
        reward_mean = np.mean(discounted_rewards)
        reward_std = np.std(discounted_rewards)
        discounted_rewards = (discounted_rewards - reward_mean) / (reward_std + 1e-8)

        # Gradient Desent
        self.optimizer.zero_grad()

        total_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(len(discounted_rewards)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = discounted_rewards[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            # loss 根据 G 值调整
            loss = -m.log_prob(action) * reward  # 负号表示梯度上升
            total_loss = total_loss + loss.sum()  # 累积 loss
            # print("loss:", loss.item())
        
        total_loss.backward()  # 对累积的总 loss 进行反向传播
        self.optimizer.step()
        self.memory.clear()

    def save_model(self, path = './model/best_pg_model.pth'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path = './model/best_pg_model.pth'):
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))

