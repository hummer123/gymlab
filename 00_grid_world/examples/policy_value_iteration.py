import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np




def policy_iteration(env, gamma=0.9, theta=1e-6, max_it=1000):
    '''
    policy_iteration 的 Docstring
    
    :param env: 说明
    :param gamma: 说明
    :param theta: 说明
    :param max_it: 说明
    '''
    # stochastic policy initialization (keep original simple behavior)
    stochastic_matrix = np.random.rand(env.num_states, len(env.action_space))
    pi_matrix = stochastic_matrix / stochastic_matrix.sum(axis=1)[:, np.newaxis]

    # ensure `V` is always defined (in case max_it == 0 or loop exits early)
    V = np.zeros(env.num_states)

    for _iter in range(max_it):
        env.render()
 
        # Policy Evaluation (iterative until convergence)
        V = np.zeros(env.num_states)
        for _eval in range(max_it):
            delta = 0
            for s in range(env.num_states):
                v = V[s]
                state_xy = env.state_index_to_xy(s)
                new_v = 0.0
                # v_k+1 = sum_a pi(a|s) * [R_pi(a) + gamma * P_pi(a) V_k(s')] == bootstrap
                for a, action in enumerate(env.action_space):
                    # GridWorld is deterministic: P(s'|s,a)=1, so just use the single next state
                    next_state, reward = env._get_next_state_and_reward(state_xy, action)
                    s_next = env.xy_to_state_index(next_state)
                    new_v += pi_matrix[s, a] * (reward + gamma * V[s_next])
                V[s] = new_v
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(env.num_states):
            old_action = np.argmax(pi_matrix[s])
            action_values = np.zeros(len(env.action_space))
            state_xy = env.state_index_to_xy(s)
            # q_pi(s,a) = R_pi(a) + gamma * P_pi(a) V(s') == one-step lookahead
            # a* = argmax_a q_pi(s,a)
            for a, action in enumerate(env.action_space):
                next_state, reward = env._get_next_state_and_reward(state_xy, action)
                s_next = env.xy_to_state_index(next_state)
                action_values[a] = reward + gamma * V[s_next]
            best_action = np.argmax(action_values)

            if old_action != best_action:
                policy_stable = False

            pi_matrix[s] = np.eye(len(env.action_space))[best_action]

        print(f"Iteration {_iter}:")
        env.add_policy(pi_matrix)
        env.add_state_values(V)
        env.render(animation_interval=2)
        if policy_stable:
            print("Policy converged.")
            break

    return pi_matrix, V





# Main function
if __name__ == "__main__":
    env = GridWorld()
    state = env.reset()

    policy_iteration(env, gamma=0.9, theta=1e-6, max_it=100)
    input('### End of Policy Iteration...')


