import numpy as np



class CartPole_Discretizer:
    def __init__(self, params):
        self.cart_pos_num = params['cart_pos_num']
        self.cart_v_num = params['cart_v_num']
        self.pole_angle_num = params['pole_angle_num']
        self.pole_v_num = params['pole_v_num']

        self.cart_pos_bins = np.linspace(-2.4, 2.4, self.cart_pos_num + 1)[1:-1]
        self.cart_v_bins = np.linspace(-3, 3, self.cart_v_num + 1)[1:-1]
        self.pole_angle_bins = np.linspace(-0.5, 0.5, self.pole_angle_num + 1)[1:-1]
        self.pole_v_bins = np.linspace(-3, 3, self.pole_v_num + 1)[1:-1]

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


