import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

import gymnasium as gym
from rl_utils import CartPole_Discretizer



class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon_rate = min_epsilon_rate
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.randint(0, self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        '''
        Q_{t+1}(s, a) = Q_t(s, a) + α [r + γ max_a' Q_t(s', a') - Q_t(s, a)]
          -> td_target = r + γ max_a' Q_t(s', a')
          -> td_error = td_target - Q_t(s, a)
          -> Q_{t+1}(s, a) = Q_t(s, a) + α * td_error
        '''
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def decay_epsilon(self):
        if self.epsilon > self.min_epsilon_rate:
            self.epsilon *= self.epsilon_decay

    def save_model(self, filePath):
        if not filePath:
            filePath = os.path.join(os.getcwd(), 'models', 'q_table.npy')

        parent_dir = os.path.dirname(filePath) or os.getcwd()
        os.makedirs(parent_dir, exist_ok=True)

        # 如果文件不存在，先创建一个有效的 .npy（写入当前 q_table）
        if not os.path.exists(filePath):
            np.save(filePath, self.q_table)
            return
        np.save(filePath, self.q_table)
    
    def load_model(self, filePath):
        if not os.path.exists(filePath):
            raise FileNotFoundError(f"No model found at {filePath}")
        self.q_table = np.load(filePath)


##############################

def train_q_learning(env, agent, discretizer, num_episodes=5000, max_steps=200):
    rewards = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f"Episodes", unit="ep", dynamic_ncols=True) as pbar:
            for ep in range(int(num_episodes / 10)):
                observation, _ = env.reset()
                state = discretizer(observation)
                episode_reward = 0
                done = False

                for i_step in range(max_steps):
                    action = agent.choose_action(state)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_state = discretizer(next_observation)

                    agent.learn(state, action, reward, next_state)
                    
                    state = next_state
                    episode_reward += reward

                    if terminated or truncated:
                        break

                rewards.append(episode_reward)
                agent.decay_epsilon()
                pbar.update(1)
                if (ep + 1) % 100 == 0:
                    avg_reward = np.mean(rewards[-100:])
                    pbar.set_postfix({'epsilon': f'{agent.epsilon:.3f}','Avg Reward': f'{avg_reward:.3f}'})

    return rewards


##------------------------------##

if __name__ == "__main__":

    params = {
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'min_epsilon_rate': 0.01,
        'gamma': 0.95,
        'learning_rate': 0.1,
        'cart_pos_num': 8,
        'cart_v_num': 8,
        'pole_angle_num': 8,
        'pole_v_num': 8,
        'model_path': 'models/best_q_table.npy',
        'max_episodes': 5000,
        'max_steps': 200
    }
    
    np.random.seed(42)
    
    env = gym.make('CartPole-v1', render_mode='human')
    # "This code only works for discrete action spaces."
    assert isinstance(env.action_space, gym.spaces.Discrete)

    discretizer = CartPole_Discretizer(params)
    state_size = params['cart_pos_num'] * params['cart_v_num'] * params['pole_angle_num'] * params['pole_v_num']
    action_size = env.action_space.n

    agent = QLearningAgent(state_size, action_size, 
                           learning_rate=params['learning_rate'], 
                           gamma=params['gamma'], 
                           epsilon=params['epsilon'], 
                           epsilon_decay=params['epsilon_decay'], 
                           min_epsilon_rate=params['min_epsilon_rate'])

    rewards = train_q_learning(env, agent, discretizer,
                              num_episodes=params['max_episodes'],
                              max_steps=params['max_steps'])

    # Save the trained model
    agent.save_model('models/q_table.npy')

    # Plot the rewards
    episode_list = list(range(len(rewards)))
    plt.plot(episode_list, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning on CartPole-v1')
    plt.show()
