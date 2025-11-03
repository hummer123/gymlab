"""
独立评估脚本 - 用于加载已保存的 Actor-Critic 模型并进行评估
"""
import gymnasium as gym
from actor_critic_agent import ActorCritic
import matplotlib.pyplot as plt
import torch
import rl_utils
import os
import numpy as np


def evaluate_saved_model(model_path, num_episodes=50, render=False):
    """加载并评估已保存的模型
    Args:
        model_path (str): 模型文件路径
        num_episodes (int): 评估回合数
        render (bool): 是否渲染环境
    """
    # 配置参数（需要与训练时一致）
    cfg = {
        'gamma': 0.98,
        'actor_lr': 1e-3, 
        'critic_lr': 1e-2,
        'hidden_dim': 128
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 创建环境
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore

    # 创建智能体并加载模型
    agent = ActorCritic(state_dim, cfg['hidden_dim'], action_dim, cfg['actor_lr'], 
                        cfg['critic_lr'], cfg['gamma'], device)
    agent.load_model(model_path)

    # 评估模型
    print(f"\n{'='*50}")
    print(f"正在评估模型: {model_path}")
    print(f"评估回合数: {num_episodes}")
    print(f"{'='*50}\n")
    
    eval_rewards = rl_utils.evaluate_agent(env, agent, num_episodes=num_episodes)
    env.close()

    # 绘制评估结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：每回合奖励柱状图
    episode_list = list(range(1, len(eval_rewards) + 1))
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    ax1.bar(episode_list, eval_rewards, alpha=0.7, color='green')
    ax1.axhline(y=mean_reward, color='r', linestyle='--', 
                label=f'Mean: {mean_reward:.2f}')
    ax1.axhline(y=mean_reward + std_reward, color='orange', linestyle=':', 
                label=f'Std: ±{std_reward:.2f}')
    ax1.axhline(y=mean_reward - std_reward, color='orange', linestyle=':')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Evaluation Rewards per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：奖励分布直方图
    ax2.hist(eval_rewards, bins=15, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(x=mean_reward, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {mean_reward:.2f}')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存评估结果图像
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, 'ac_evaluation_results.png')
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    print(f"\n评估结果图像已保存到: {result_path}")
    
    plt.show()
    
    return eval_rewards


if __name__ == "__main__":
    # 设置模型路径
    model_path = './model/ac_cartpole_model.pth'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行 cartpole_ac.py 训练模型")
    else:
        # 评估模型
        evaluate_saved_model(model_path, num_episodes=50, render=False)
