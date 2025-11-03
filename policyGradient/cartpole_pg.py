import time
import gymnasium as gym
from policyGradient_agent import PolicyGradient, PGNet
import matplotlib.pyplot as plt


def train_episode(env, agent, episodes):
    t0 = time.time()
    all_episode_rewards = []
    best_avg_reward = float('-inf')
    SAVE_INTERVAL = 20  # 每 20 个 episode 检查一次
    WINDOW_SIZE = 10    # 用最近 10 个 episode 的平均值判断
    
    for step_ix in range(episodes):
        try:
            state = env.reset()[0]
            episode_reward = 0
            t1 = time.time()

            while True:
                action = agent.sample_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward
                if done:
                    break

            agent.learn()
            
            # 记录奖励
            all_episode_rewards.append(episode_reward)
            
            print(f"Training | Episode: {step_ix+1}/{episodes}, "
                  f"Reward: {episode_reward}, Time Elapsed: {time.time() - t1:.2f}s")
            
            # 定期检查是否保存最好模型（基于滑动窗口平均）
            if (step_ix + 1) % SAVE_INTERVAL == 0 and len(all_episode_rewards) >= WINDOW_SIZE:
                recent_avg = sum(all_episode_rewards[-WINDOW_SIZE:]) / WINDOW_SIZE
                # print(f"  Recent {WINDOW_SIZE}-episode average: {recent_avg:.2f}")
                
                if recent_avg > best_avg_reward:
                    best_avg_reward = recent_avg
                    agent.save_model(path='./model/best_pg_model.pth')
                    print(f"  → Best model saved! Avg reward: {best_avg_reward:.2f}")
        
        except Exception as e:
            print(f"Exception caught in episode {step_ix+1}: {e}")
            continue
    
    # 训练结束后保存最终模型
    agent.save_model(path='./model/final_pg_model.pth')
    print(f"\n{'='*60}")
    print(f"Training Summary:")
    print(f"  Total episodes: {episodes}")
    print(f"  Best average reward: {best_avg_reward:.2f}")
    print(f"  Final model saved to: ./model/final_pg_model.pth")
    print(f"  Best model saved to: ./model/best_pg_model.pth")
    print(f"  Total training time: {time.time() - t0:.2f}s")
    print(f"{'='*60}\n")
    
    return all_episode_rewards


def eval_episode(env, agent, episodes, is_best_model=False):
    total_rewards = []
    
    try:
        if is_best_model:
            path = './model/best_pg_model.pth'
        else:
            path = './model/final_pg_model.pth'

        agent.load_model(path)
        print(f"Model({path}) loaded successfully for evaluation.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0  # 如果模型加载失败，直接返回
    
    for step_ix in range(episodes):
        try:
            state = env.reset()[0]
            episode_reward = 0
            t1 = time.time()

            while True:
                # action = agent.sample_action(state)
                action = agent.predict_action(state)  # 评估时使用确定性策略
                next_state, reward, done, _, _ = env.step(action)
                state = next_state
                episode_reward += reward
                if done:
                    break

            total_rewards.append(episode_reward)
            print(f"Evaluation | Episode: {step_ix+1}/{episodes}, "
                   f"Reward: {episode_reward}, Time Elapsed: {time.time() - t1:.2f}s")

        except Exception as e:
            print(f"Exception caught in episode {step_ix+1}: {e}")
            continue

    # 计算平均奖励（只统计成功的 episode）
    if total_rewards:
        avg_reward = sum(total_rewards) / len(total_rewards)
        max_reward = max(total_rewards)
        min_reward = min(total_rewards)
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Max reward: {max_reward:.2f}")
        print(f"  Min reward: {min_reward:.2f}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"Evaluation failed: No successful episodes!")
        print(f"{'='*60}\n")
        avg_reward = 0
    
    return avg_reward




if __name__ == "__main__":
    env = gym.make('CartPole-v1', render_mode="human")
    state_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore

    model = PGNet(state_dim, action_dim, hidden_dim=64)
    cfg = {
        'gamma': 0.99,
        'lr': 0.001,  # 降低学习率以提高训练稳定性
        'device': 'cpu'
    }
    agent = PolicyGradient(model, cfg)

    # Training loop would go here
    print(f"Starting training...")
    plt.plot(train_episode(env, agent, episodes=600))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards over Episodes')
    plt.show()
    print(f"Training completed. Starting evaluation...")

    eval_episode(env, agent, episodes=3, is_best_model=True)
    print(f"Evaluation completed.")

    env.close()
