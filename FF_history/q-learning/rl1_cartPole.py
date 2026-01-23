import warnings
# 忽略 DeprecationWarning,避免 pygame 的 pkg_resources 警告导致程序崩溃
warnings.filterwarnings('ignore', category=DeprecationWarning)
# 如果需要捕获其他警告,可以针对特定类型设置为 error
# warnings.filterwarnings('error', category=RuntimeWarning)

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from utils import cartPole_discretizer, QlAgent


def train_episode(env, model, num_episodes):
    observation, info = env.reset()
    
    step_ix = 0
    for step_ix in range(1000):
        try:
            # e-greedy policy
            action = model.decide_action(observation, train_mode=True)
            # take action
            new_observation, reward, terminated, truncated, info = env.step(action)
            # punish early stop
            if terminated and step_ix < 180:
                reward = -2
            # update Q-table
            model.update_q_value(observation, action, reward, new_observation)
            # update observation
            observation = new_observation
            # check termination
            if terminated or truncated:
                break
        except Warning:
            print("Warning catched!")
            step_ix -= 1
            break

    return step_ix+1


def eval_episode(env, model, num_trials, render=False):
    acc_steps = 0
    succeeded_trials = 0
    
    for trial_ix in range(num_trials):
        observation, info = env.reset()
        step_ix = 0
        for step_ix in range(1000):
            try:
                # e-greedy policy
                action = model.decide_action(observation, train_mode=False)
                # take action
                new_observation, reward, terminated, truncated, info = env.step(action)
                # punish early stop
                if terminated:
                    succeeded_trials += int(step_ix>=199)
                    if render:
                        print(f"Trial {trial_ix+1} finished in {step_ix+1} steps.")
                    break
                # update observation
                observation = new_observation
            except Warning:
                print("Warning catched!")
                step_ix -= 1
                break
        acc_steps += step_ix + 1

    print('\nEvaluation results:')
    print('  Average steps: {}. Succeeded trials: {}/{}'.format(acc_steps / num_trials, succeeded_trials, num_trials))
    return succeeded_trials


def build_model(params):
    env = gym.make('CartPole-v1')
    assert isinstance(env.action_space, Discrete)

    discretizer = cartPole_discretizer(params)
    num_states = params['cart_pos_num'] * params['cart_v_num'] * params['pole_angle_num'] * params['pole_v_num']
    model = QlAgent(num_states, env.action_space.n, discretizer, params['epsilon'], params['alpha'], params['gamma']) 

    # training model
    acc_steps = 0
    best_suc_rate = 0.
    for episode in range(params['build_episode']):
        step_ix = train_episode(env, model, episode)
        acc_steps += step_ix

        if (episode+1) % params['eval_interval'] == 0:
            suc_trials = eval_episode(env, model, params['num_eval_trials'])
            suc_rate = suc_trials / params['num_eval_trials']
            print('Episode: {}. Average steps: {}. Success rate: {:.2f}'.format(
                episode+1, acc_steps/params['eval_interval'], suc_rate))
            
            acc_steps = 0
            if suc_rate > best_suc_rate:
                best_suc_rate = suc_rate
                model.save_q_table(params['model_save_path'].replace('.npy', ''))
                print(f"New best model saved to {params['model_save_path']}.")

            if best_suc_rate >= 0.95:
                print("Early stopping as success rate reached 95%.")
                break
    # final evaluation
    eval_episode(env, model, params['num_eval_trials'])
    
    env.close()


def show_model(params):
    env = gym.make('CartPole-v1', render_mode="human")
    assert isinstance(env.action_space, Discrete)

    discretizer = cartPole_discretizer(params)
    num_states = params['cart_pos_num'] * params['cart_v_num'] * params['pole_angle_num'] * params['pole_v_num']
    model = QlAgent(num_states, env.action_space.n, discretizer, params['epsilon'], params['alpha'], params['gamma']) 

    # show trained model
    model.load_q_table(params['model_save_path'])
    print(f"Model loaded from {params['model_save_path']}.")
    
    suc_trials = eval_episode(env, model, params['num_eval_trials'], True)
    suc_rate = suc_trials / params['num_eval_trials']
    print('Final success rate over {} trials: {:.2f}'.format(params['num_eval_trials'], suc_rate))

    env.close()


if __name__ == "__main__":
    params = {
        'epsilon': 0.05,
        'alpha': 0.2,
        'gamma': 0.95,
        'build_episode': 500000,
        'num_eval_trials': 1000,
        'eval_interval': 1000,

        'cart_pos_num': 8,
        'cart_v_num': 8,
        'pole_angle_num': 8,
        'pole_v_num': 8,
        'model_save_path': './model/best_model.npy'
    }
    build_model(params)
    print("\n----------- show model -----------\n")
    show_model(params)
