"""
对比分析4种DQN训练场景：
1. DQNAgent - 只保存最后模型
2. DQNAgent - 保存最好模型
3. BadQNAgent - 只保存最后模型
4. BadQNAgent - 保存最好模型
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from netAgent import DQNAgent, BadQNAgent


def train_episode_with_logging(env, agent, episodes, save_best=True, model_path='./model/temp_model.pth'):
    """
    训练并记录每个episode的奖励
    
    Args:
        env: 训练环境
        agent: 训练的agent
        episodes: 训练的episode数量
        save_best: 是否保存最好的模型
        model_path: 模型保存路径
    
    Returns:
        episode_rewards: 每个episode的总奖励列表
        eval_rewards: 每10个episode的评估奖励列表
    """
    episode_rewards = []
    eval_rewards = []
    
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

            episode_rewards.append(total_reward)

            # 评估和保存模型
            if save_best:
                if (step_ix + 1) % 10 == 0:
                    eval_env = gym.make('CartPole-v1', render_mode=None)
                    avg_reward = agent.evaluate(eval_env, eval_episodes=5)
                    eval_env.close()
                    eval_rewards.append(avg_reward)

                    if avg_reward > agent.getBestAvgReward():
                        agent.setBestAvgReward(avg_reward)
                        agent.save_model(path=model_path)
                        print(f"  -> New best average reward: {avg_reward:.2f} at episode {step_ix+1}")

            if (step_ix + 1) % 20 == 0:
                print(f"Episode {step_ix+1}/{episodes}, Reward: {total_reward:.2f}, "
                    f"Best Avg: {agent.getBestAvgReward():.2f}, Epsilon: {agent.getEpsilon():.3f}")

        except Warning:
            print("Warning catched!")
            continue
    
    # 如果不保存最好的模型，则保存最后的模型
    if not save_best:
        agent.save_model(path=model_path)
        print(f"  -> Final model saved to {model_path}")
    
    return episode_rewards, eval_rewards


def evaluate_final_model(agent, model_path, eval_episodes=10):
    """
    加载模型并评估
    """
    eval_env = gym.make('CartPole-v1', render_mode=None)
    agent.load_model(path=model_path)
    avg_reward = agent.evaluate(eval_env, eval_episodes=eval_episodes)
    eval_env.close()
    return avg_reward


def plot_comparison(results, save_path='./result/comparison.png'):
    """
    绘制4个场景的对比图
    
    Args:
        results: 包含4个场景结果的字典
        save_path: 图片保存路径
    """
    # 确保picture目录存在
    os.makedirs('./result', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DQN Training Scenarios Comparison', fontsize=16, fontweight='bold')
    
    scenarios = [
        ('dqn_final', 'DQNAgent - Save Final Model Only', axes[0, 0]),
        ('dqn_best', 'DQNAgent - Save Best Model', axes[0, 1]),
        ('bad_dqn_final', 'BadQNAgent - Save Final Model Only', axes[1, 0]),
        ('bad_dqn_best', 'BadQNAgent - Save Best Model', axes[1, 1])
    ]
    
    for key, title, ax in scenarios:
        data = results[key]
        episode_rewards = data['episode_rewards']
        eval_rewards = data['eval_rewards']
        final_eval = data['final_eval']
        
        # 绘制每个episode的奖励
        episodes = list(range(1, len(episode_rewards) + 1))
        ax.plot(episodes, episode_rewards, alpha=0.5, color='royalblue', linewidth=0.8, label='Episode Reward')
        
        # 绘制移动平均（平滑曲线）
        window_size = 10
        if len(episode_rewards) >= window_size:
            moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(range(window_size, len(episode_rewards) + 1), moving_avg, 
                   color='red', linewidth=2, label=f'Moving Avg (window={window_size})')
        
        # 绘制评估点
        eval_episodes = list(range(10, len(episode_rewards) + 1, 10))
        if len(eval_rewards) > 0:
            ax.scatter(eval_episodes[:len(eval_rewards)], eval_rewards, 
                      color='green', s=50, marker='o', zorder=5, label='Evaluation Reward')
        
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.set_title(f'{title}\nFinal Eval: {final_eval:.2f}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=8)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to {save_path}")
    plt.show()
    plt.close()


def main():
    """
    主函数：运行4种场景并绘制对比图
    """
    # 配置参数
    cfg = {
        'lr': 0.001,
        'batch_size': 64,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'update_target_freq': 100,
        'episodes': 1000
    }
    
    results = {}
    
    # 获取环境的维度信息
    temp_env = gym.make('CartPole-v1')
    state_dim = temp_env.observation_space.shape[0] # type: ignore
    action_dim = temp_env.action_space.n # type: ignore
    temp_env.close()
    
    print("=" * 80)
    print("场景1: DQNAgent - 只保存最后模型")
    print("=" * 80)
    env1 = gym.make('CartPole-v1')
    agent1 = DQNAgent(state_dim, action_dim, cfg)
    episode_rewards1, eval_rewards1 = train_episode_with_logging(
        env1, agent1, episodes=cfg['episodes'], save_best=False, 
        model_path='./model/dqn_final_model.pth'
    )
    final_eval1 = evaluate_final_model(agent1, './model/dqn_final_model.pth')
    env1.close()
    print(f"✅ 场景1完成 - 最终评估奖励: {final_eval1:.2f}\n")
    results['dqn_final'] = {
        'episode_rewards': episode_rewards1,
        'eval_rewards': eval_rewards1,
        'final_eval': final_eval1
    }
    
    print("=" * 80)
    print("场景2: DQNAgent - 保存最好模型")
    print("=" * 80)
    env2 = gym.make('CartPole-v1')
    agent2 = DQNAgent(state_dim, action_dim, cfg)
    episode_rewards2, eval_rewards2 = train_episode_with_logging(
        env2, agent2, episodes=cfg['episodes'], save_best=True, 
        model_path='./model/dqn_best_model.pth'
    )
    final_eval2 = evaluate_final_model(agent2, './model/dqn_best_model.pth')
    env2.close()
    print(f"✅ 场景2完成 - 最终评估奖励: {final_eval2:.2f}\n")
    results['dqn_best'] = {
        'episode_rewards': episode_rewards2,
        'eval_rewards': eval_rewards2,
        'final_eval': final_eval2
    }
    
    print("=" * 80)
    print("场景3: BadQNAgent - 只保存最后模型")
    print("=" * 80)
    env3 = gym.make('CartPole-v1')
    agent3 = BadQNAgent(state_dim, action_dim, cfg)
    episode_rewards3, eval_rewards3 = train_episode_with_logging(
        env3, agent3, episodes=cfg['episodes'], save_best=False, 
        model_path='./model/bad_dqn_final_model.pth'
    )
    final_eval3 = evaluate_final_model(agent3, './model/bad_dqn_final_model.pth')
    env3.close()
    print(f"✅ 场景3完成 - 最终评估奖励: {final_eval3:.2f}\n")
    results['bad_dqn_final'] = {
        'episode_rewards': episode_rewards3,
        'eval_rewards': eval_rewards3,
        'final_eval': final_eval3
    }
    
    print("=" * 80)
    print("场景4: BadQNAgent - 保存最好模型")
    print("=" * 80)
    env4 = gym.make('CartPole-v1')
    agent4 = BadQNAgent(state_dim, action_dim, cfg)
    episode_rewards4, eval_rewards4 = train_episode_with_logging(
        env4, agent4, episodes=cfg['episodes'], save_best=True, 
        model_path='./model/bad_dqn_best_model.pth'
    )
    final_eval4 = evaluate_final_model(agent4, './model/bad_dqn_best_model.pth')
    env4.close()
    print(f"✅ 场景4完成 - 最终评估奖励: {final_eval4:.2f}\n")
    results['bad_dqn_best'] = {
        'episode_rewards': episode_rewards4,
        'eval_rewards': eval_rewards4,
        'final_eval': final_eval4
    }
    
    # 绘制对比图
    print("=" * 80)
    print("正在绘制对比图...")
    print("=" * 80)
    plot_comparison(results)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("📊 实验结果总结")
    print("=" * 80)
    print(f"场景1 (DQN - Final Model):     {final_eval1:.2f}")
    print(f"场景2 (DQN - Best Model):      {final_eval2:.2f}")
    print(f"场景3 (BadDQN - Final Model):  {final_eval3:.2f}")
    print(f"场景4 (BadDQN - Best Model):   {final_eval4:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
