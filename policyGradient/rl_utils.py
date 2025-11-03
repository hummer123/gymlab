import random
import numpy as np
from tqdm import tqdm
import collections
import torch



class ReplayBuffer:
    '''Experience Replay Buffer for storing and sampling transitions.
    Args:
        capacity (int): Maximum number of transitions to store in the buffer.
    Methods:
        add(state, action, reward, next_state, done): Add a transition to the buffer
        sample(batch_size): Sample a batch of transitions from the buffer
        clear(): Clear all transitions from the buffer
    '''
    def __init__(self, capacity=10000):
        self.buffer = collections.deque(maxlen=capacity)

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def clear(self):
        self.buffer.clear()


def moving_average(data, window_size):
    """计算数据的滑动窗口平均值
    Args:
        data (list or np.array): 输入数据
        window_size (int): 窗口大小
    """
    cumulative_sum = np.cumsum(np.insert(data, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(data[:window_size - 1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    """训练基于策略的智能体"""
    all_episode_rewards = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                state = env.reset()[0]
                episode_reward = 0
                transition_dict = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}

                while True:
                    action = agent.choose_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['rewards'].append(reward)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_reward += reward

                    if done or episode_reward > 500:
                        break

                all_episode_rewards.append(episode_reward)
                agent.update(transition_dict)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(all_episode_rewards[-10:])
                    })
                pbar.update(1)

    return all_episode_rewards


def evaluate_agent(env, agent, num_episodes=10):
    """评估智能体性能
    Args:
        env: 环境
        agent: 智能体
        num_episodes (int): 评估的回合数
    Returns:
        list: 每个回合的奖励列表
    """
    eval_rewards = []
    with torch.no_grad():
        for episode in range(num_episodes):
            state = env.reset()[0]
            episode_reward = 0
            done = False
            
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                episode_reward += reward
            
            eval_rewards.append(episode_reward)
            print(f"Evaluating | {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        mean_reward = np.mean(eval_rewards)
        std_reward = np.std(eval_rewards)
    print(f"\nEvaluating : Avg Reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return eval_rewards
