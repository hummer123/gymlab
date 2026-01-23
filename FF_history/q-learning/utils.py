# Utilizations for Cartpole Q-learning
import random
import numpy as np



class QlAgent:
    def __init__(self, num_states, num_actions, discretizer, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = np.random.uniform(low=-1, high=1, size=(num_states, num_actions)).astype(np.float32)
        self.action_space = range(self.q_table.shape[-1]) # e.g., range(2) for CartPole
        self.discretizer = discretizer  # Discretizer instance
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor

    def decide_action(self, observation, train_mode=False):
        state = self.discretizer(observation)
        if random.random() < self.epsilon and train_mode:   # when evaluating, do not do random (?)
            return random.choice(self.action_space)
        else:
            return self.q_table[state].argmax()

    def update_q_value(self, observation_cur, action, reward, observation_next):
        state_cur = self.discretizer(observation_cur)
        state_next = self.discretizer(observation_next)

        max_q_next = self.q_table[state_next].max()
        current_q = self.q_table[state_cur][action]

        # Q-learning update rule
        self.q_table[state_cur][action] = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)

    def save_q_table(self, fn_out):
        np.save(fn_out, self.q_table)

    def load_q_table(self, fn_in):
        self.q_table = np.load(fn_in)



class cartPole_discretizer:
    def __init__(self, params):
        
        # range numbers
        self.cart_pos_num   = params['cart_pos_num']
        self.cart_v_num     = params['cart_v_num']
        self.pole_angle_num = params['pole_angle_num']
        self.pole_v_num     = params['pole_v_num']
        
        # pre-compute digitization ranges
        '''
        cartPole observation space:
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

        We only care about the following practical ranges:
        1. Cart Position: -2.4 ~ 2.4
        2. Cart Velocity: -3.0 ~ 3.0
        3. Pole Angle: -0.5 ~ 0.5
        4. Pole Velocity At Tip: -3.0 ~ 3.0
        '''
        self.cart_pos_bins = \
            np.linspace(-2.4, 2.4, self.cart_pos_num+1)[1:-1]
        self.cart_v_bins = \
            np.linspace(-3, 3, self.cart_v_num+1)[1:-1]
        self.pole_angle_bins = \
            np.linspace(-0.5, 0.5, self.pole_angle_num+1)[1:-1]
        self.pole_v_bins = \
            np.linspace(-3, 3, self.pole_v_num+1)[1:-1]

        # print(f"== cart_pos_bins: {self.cart_pos_bins}")
        # print(f"== cart_v_bins: {self.cart_v_bins}")
        # print(f"== pole_angle_bins: {self.pole_angle_bins}")
        # print(f"== pole_v_bins: {self.pole_v_bins}")

    def __call__(self, observation):
        cart_pos, cart_v, pole_angle, pole_v = observation
        # print(f"## observation: {observation}")
        
        # convert to the index in discretized ranges
        cart_pos_ix   = np.digitize(cart_pos, self.cart_pos_bins)
        cart_v_ix     = np.digitize(cart_v, self.cart_v_bins)
        pole_angle_ix = np.digitize(pole_angle, self.pole_angle_bins)
        pole_v_ix     = np.digitize(pole_v, self.pole_v_bins)
        # print(f"  == [{cart_pos_ix}, {cart_v_ix}, {pole_angle_ix}, {pole_v_ix}]")

        # convert all ix to unique state index
        '''
        四组状态融合
        四个维状态转为一个唯一状态，方便在Q-table（以state为key）中使用：
        1. 乘积编码法（当前实现方法）
        每一维的类别数分别为 N0, N1, N2, N3，四维索引分别为 i0, i1, i2, i3，则唯一状态为：
        state = ((i0 * N1 + i1) * N2 + i2) * N3 + i3
        '''
        cur_state = cart_pos_ix
        cur_state = self.cart_v_num * cur_state + cart_v_ix
        cur_state = self.pole_angle_num * cur_state + pole_angle_ix
        cur_state = self.pole_v_num * cur_state + pole_v_ix
        # print(f"  == cur_state: {cur_state}")
        
        return cur_state
