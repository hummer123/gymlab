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


class Network(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Network, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)




class TRPOAgent:
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda, kl_constraint, alpha, critic_lr, gamma, device):
        self.actor = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        self.critic  = Network(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lmbda = lmbda  # GAE的λ参数，控制bias-variance trade-off
        self.alpha = alpha  # TRPO算法中的步长缩放因子，控制更新步长
        self.gamma = gamma
        self.device = device
        self.kl_constraint = kl_constraint # TRPO算法中的KL散度约束，控制更新步长

    def take_action(self, state):
        # make state shape: (1, state_dim)
        state = torch.as_tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算Hessian矩阵与向量的乘积，使用自动微分实现
        # 这里的Hessian矩阵是TRPO算法中的Fisher信息矩阵
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算KL散度，并对其求导得到Hessian矩阵的近似
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        kl_grad_vector_product = (kl_grad_vector * vector).sum()
        
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters(), create_graph=True)
        grad2_vector = torch.cat([grad.contiguous().view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, state, old_action_dists):
        # 使用共轭梯度法求解Hessian矩阵的逆乘积，得到TRPO算法中的更新方向
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = r.clone()
        rdotr = torch.dot(r, r)

        for _ in range(10):  # 共轭梯度迭代次数
            Hp = self.hessian_matrix_vector_product(state, old_action_dists, p)
            
            alpha = rdotr / (torch.dot(p, Hp) + 1e-8)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if torch.sqrt(new_rdotr) < 1e-10:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr

        return x

    def compute_surrogate_obj(self, states, actions, advantages, old_action_dists, actor):
        log_probs = torch.log(actor(states).gather(1, actions) + 1e-8)
        ratio = torch.exp(log_probs - torch.log(old_action_dists.gather(1, actions) + 1e-8))
        return torch.mean(ratio * advantages)


    def update(self, transitions_dict):
        pass