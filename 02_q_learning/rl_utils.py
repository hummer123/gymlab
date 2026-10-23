from tqdm import tqdm
import numpy as np
import torch 
import collections
import random




class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transtions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transtions)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


def moving_average(data, window_size):
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='same')



def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration: {i + 1}") as pbar:
            for ep in range(int(num_episodes / 10)):
                state, _ = env.reset()
                transitions_dict = {
                    'states': [],
                    'actions': [],
                    'rewards': [],
                    'next_states': [],
                    'dones': []
                }
                episode_return = 0
                done = False

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    transitions_dict['states'].append(state)
                    transitions_dict['actions'].append(action)
                    transitions_dict['rewards'].append(reward)
                    transitions_dict['next_states'].append(next_state)
                    transitions_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward

                agent.update(transitions_dict)
                return_list.append(episode_return)
                pbar.update(1)
                if (ep + 1) % 60 == 0:
                    avg_return = np.mean(return_list[-60:])
                    pbar.set_postfix({'Avg Return': f'{avg_return:.3f}'})
    return return_list


def evaluate_on_policy_agent(env, agent, n_episodes=1, render=False):
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            if render:
                env.render()
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

        rewards.append(episode_reward)
    
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {n_episodes} episodes: {avg_reward:.2f}")
    return rewards


