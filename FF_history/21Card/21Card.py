from collections import defaultdict
import gymnasium as gym
import numpy as np
from typing import Tuple

class BlackJackAgent:
    def __init__(
            self, 
            env:gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95
        ):
        """Initialize a Q-learning agent.

        """
        self.env = env

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: Tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.
        
        observation: (player_sum, dealer_card, usable_ace)

        Return:
            action: 0(stand) or 1(hit)
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(
            self,
            obs: Tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: Tuple[int, int, bool]
        ):
        """Update Q-value based on experience.

        """
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate
        temporal_difference = target - self.q_values[obs][action]

        self.q_values[obs][action] = (self.q_values[obs][action] + self.lr * temporal_difference)

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode. """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# Training hyperparameters
learning_rate = 0.1
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # Decay over half the episodes
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackJackAgent(
    env = env,
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon
)

# action loop
from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        # Move to the next state
        done = terminated or truncated
        obs = next_obs

    # Reduce exploration rate
    agent.decay_epsilon()


# gui 
import matplotlib.pyplot as plt

def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

rolling_length = 500
fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

axs[0].set_title("Episode Rewards")
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Average Reward")

# Episode lengths
axs[1].set_title("Episode Lengths")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
axs[1].plot(range(len(length_moving_average)), length_moving_average)
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Average Length")

# Training error
axs[2].set_title("Training Error")
training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "valid")
axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
axs[2].set_xlabel("Step")
axs[2].set_ylabel("Average TD Error")

plt.tight_layout()
plt.show()
