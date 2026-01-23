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
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    moving_aves = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(data[:window_size -1])[::2] / r
    end = (np.cumsum(data[:-window_size:-1])[::2] / r)[::-1]
    moving_aves = np.concatenate((begin, moving_aves, end))
    return moving_aves




