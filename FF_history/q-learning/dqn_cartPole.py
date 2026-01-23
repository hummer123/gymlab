import gymnasium as gym
from netAgent import DQNAgent, BadQNAgent


def train_episode(env, agent, episodes, model_path='./model/best_q_model.pth'):
    for step_ix in range(episodes):
        try:
            state = env.reset()[0]
            total_reward = 0

            while True:
                action = agent.choose_action(state, train=True)
                next_state, reward, done, _, _ = env.step(action)
                agent.store_experience(state, action, reward, next_state, done)
                agent.learn()

                total_reward += reward
                state = next_state
                if done or total_reward > 500:
                    break

            # Evaluate and save best model
            if (step_ix + 1) % 10 == 0:
                eval_env = gym.make('CartPole-v1', render_mode=None)
                avg_reward = agent.evaluate(eval_env)
                eval_env.close()

                if avg_reward > agent.getBestAvgReward():
                    agent.setBestAvgReward(avg_reward)
                    agent.save_model(path=model_path)
                    print(f"Train | New best average reward: {avg_reward:.2f} at episode {step_ix+1}, model saved.")

            print(f"Train | Episode {step_ix+1}, Total Reward: {total_reward}, "
                  f"Best Avg Reward: {agent.getBestAvgReward():.2f}, Epsilon: {agent.getEpsilon():.3f}")

        except Warning:
            print("Warning catched!")
            step_ix -= 1
            break

if __name__ == "__main__":

    cfg = {
        'lr': 0.001,  # 降低学习率以提高训练稳定性
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'update_target_freq': 100
    }

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore
    dqn_agent = DQNAgent(state_dim, action_dim, cfg)
    
    # 训练并评估 DQN Agent
    train_episode(env, dqn_agent, episodes=100, model_path="./model/best_q_model.pth")
    eval_env = gym.make('CartPole-v1', render_mode="human")
    dqn_agent.load_model(path="./model/best_q_model.pth")
    avg_reward = dqn_agent.evaluate(eval_env, eval_episodes=10)
    print(f"Evaluation | Best DQN, Avg reward: {avg_reward:.2f}.")

    # 训练一个性能较差的 DQN 作为对比
    bad_agent = BadQNAgent(state_dim, action_dim, cfg)
    train_episode(env, bad_agent, episodes=100, model_path="./model/best_bad_q_model.pth")
    bad_agent.load_model(path="./model/best_bad_q_model.pth")
    avg_reward = bad_agent.evaluate(eval_env, eval_episodes=10)
    print(f"Evaluation | Best Bad DQN, Avg reward: {avg_reward:.2f}.")

    eval_env.close()
    env.close()
