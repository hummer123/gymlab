import time
import gymnasium as gym
from actor_critic_agent import ActorCritic
import matplotlib.pyplot as plt
import torch
import rl_utils
import os



if __name__ == "__main__":

    cfg = {
        'gamma': 0.98,
        'actor_lr': 1e-3, 
        'critic_lr': 1e-2,
        'num_episodes': 1000,
        'hidden_dim': 128
    }
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0] # type: ignore
    action_dim = env.action_space.n # type: ignore

    agent = ActorCritic(state_dim, cfg['hidden_dim'], action_dim, cfg['actor_lr'], 
                        cfg['critic_lr'], cfg['gamma'], device)

    return_list = rl_utils.train_on_policy_agent(env, agent, cfg['num_episodes'])
    env.close()

    # 保存训练后的模型
    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'ac_cartpole_model.pth')
    agent.save_model(model_path)

    # 评估训练后的模型
    print("\n" + "="*50)
    print("Evaluating trained model...")
    print("="*50)
    eval_env = gym.make('CartPole-v1')
    eval_rewards = rl_utils.evaluate_agent(eval_env, agent, num_episodes=20)
    eval_env.close()

    # 绘制训练过程的回报曲线 vs. 平滑回报曲线
    episode_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 10)
    
    # 在一个窗口中绘制三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    
    # 左图：原始回报
    ax1.plot(episode_list, return_list)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Actor-Critic on CartPole-v1')
    ax1.grid(True, alpha=0.3)
    
    # 中图：平滑后的回报
    ax2.plot(episode_list, mv_return)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Moving Average Return')
    ax2.set_title('Actor-Critic on CartPole-v1 (Moving Average)')
    ax2.grid(True, alpha=0.3)
    
    # 右图：评估奖励柱状图
    eval_episode_list = list(range(1, len(eval_rewards) + 1))
    ax3.bar(eval_episode_list, eval_rewards, alpha=0.7, color='green')
    ax3.axhline(y=sum(eval_rewards)/len(eval_rewards), color='r', linestyle='--', 
                label=f'Mean: {sum(eval_rewards)/len(eval_rewards):.2f}')
    ax3.set_xlabel('Evaluation Episode')
    ax3.set_ylabel('Reward')
    ax3.set_title('Evaluation Rewards')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)
    plt.savefig(os.path.join(result_dir, 'ac_training_evaluation.png'), dpi=300, bbox_inches='tight')
    print(f"\Picture saved at: {os.path.join(result_dir, 'ac_training_evaluation.png')}")
    
    plt.show()

