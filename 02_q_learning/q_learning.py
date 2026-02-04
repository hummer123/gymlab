import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import gymnasium as gym
import os





class CartPole_Discretizer:
    def __init__(self, params):
        self.cart_pos_num = params['cart_pos_num']
        self.cart_v_num = params['cart_v_num']
        self.pole_angle_num = params['pole_angle_num']
        self.pole_v_num = params['pole_v_num']

        self.cart_pos_bins = np.linspace(-2.4, 2.4, self.cart_pos_num + 1)[1:-1]
        self.cart_v_bins = np.linspace(-3, 3, self.cart_v_num + 1)[1:-1]
        self.pole_angle_bins = np.linspace(-0.5, 0.5, self.pole_angle_num + 1)[1:-1]
        self.pole_v_bins = np.linspace(-5, 5, self.pole_v_num + 1)[1:-1]

    def __call__(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation

        cart_pos_discrete = np.digitize(cart_pos, self.cart_pos_bins)
        cart_v_discrete = np.digitize(cart_v, self.cart_v_bins)
        pole_angle_discrete = np.digitize(pole_angle, self.pole_angle_bins)
        pole_v_discrete = np.digitize(pole_v, self.pole_v_bins)

        # merge to a single discrete state
        cur_state = cart_pos_discrete
        cur_state = cur_state * self.cart_v_num + cart_v_discrete
        cur_state = cur_state * self.pole_angle_num + pole_angle_discrete
        cur_state = cur_state * self.pole_v_num + pole_v_discrete

        return cur_state

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
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        '''
        Q_{t+1}(s, a) = Q_t(s, a) + α [r + γ max_a' Q_t(s', a') - Q_t(s, a)]
            -> td_target = r + γ max_a' Q_t(s', a')  (if not done)
            -> td_target = r                         (if done, no future reward!)
          -> td_error = td_target - Q_t(s, a)
          -> Q_{t+1}(s, a) = Q_t(s, a) + α * td_error
        '''
        if done:
            td_target = reward  # 终止状态没有未来回报
        else:
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
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration: {i + 1}", unit="ep", dynamic_ncols=True) as pbar:
            for ep in range(int(num_episodes / 10)):
                observation, _ = env.reset()
                state = discretizer(observation)
                episode_reward = 0
                done = False

                for i_step in range(max_steps):
                    action = agent.choose_action(state)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    next_state = discretizer(next_observation)
                    done = terminated or truncated

                    agent.learn(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward

                    if done:
                        break

                rewards.append(episode_reward)
                agent.decay_epsilon()
                pbar.update(1)
                if (ep + 1) % 100 == 0:
                    avg_reward = np.mean(rewards[-100:])
                    pbar.set_postfix({'Episode': f'{(i * (num_episodes // 10)) + ep + 1}',
                                      'epsilon': f'{agent.epsilon:.3f}',
                                      'Avg Reward': f'{avg_reward:.3f}'})

    return rewards


def show_agent(env, agent, discretizer, max_steps=400):
    observation, _ = env.reset()
    state = discretizer(observation)
    total_reward = 0
    done = False

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for i_step in range(max_steps):
        env.render()
        action = agent.choose_action(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = discretizer(next_observation)
        done = terminated or truncated

        state = next_state
        total_reward += reward

        if done:
            break

    agent.epsilon = old_epsilon

    print(f"Total Reward: {total_reward}")
    print(f"This agent table shape: {agent.q_table.shape}")


##------------------------------##

if __name__ == "__main__":

    params = {
        'epsilon': 1.0,
        'epsilon_decay': 0.999,
        'min_epsilon_rate': 0.01,
        'gamma': 0.95,
        'learning_rate': 0.1,
        'cart_pos_num': 8,
        'cart_v_num': 8,
        'pole_angle_num': 8,
        'pole_v_num': 8,
        'model_path': 'models/best_q_table.npy',
        'max_episodes': 5000,
        'max_steps': 400
    }
    
    np.random.seed(42)
    
    # env = gym.make('CartPole-v1', render_mode='human')
    env = gym.make('CartPole-v1')
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
    env.close()

    # Show the trained agent
    show_env = gym.make('CartPole-v1', render_mode='human')
    show_agent(show_env, agent, discretizer, max_steps=400)
    show_env.close()

    # Plot the rewards
    episode_list = list(range(len(rewards)))
    plt.plot(episode_list, rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Q-Learning on CartPole-v1')
    plt.show()
