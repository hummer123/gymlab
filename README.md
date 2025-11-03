# 🤖 强化学习实践实验室 (GymLab)

> 个人强化学习学习过程中的实践代码集合，通过动手实现加深对RL算法的理解

## 📚 项目简介

本项目是本人在学习强化学习过程中编写的实践代码，涵盖了从传统强化学习算法到深度强化学习的各种实现。项目包含了一个自定义的OpenAI Gym环境框架和多个经典算法的具体实现，旨在通过"动手实践"的方式加深对强化学习核心概念的理解。

## 🎯 学习目标

- 🔄 深入理解强化学习的核心算法原理
- 💪 通过代码实现掌握算法细节
- 📊 可视化训练过程和性能对比
- 🧮 建立完整��RL算法开发流程
- 🎲 体验不同环境下的算法表现

## 📁 项目结构

```
gymlab/
├── 🏗️ gym/                    # OpenAI Gym框架（fork版本）
│   ├── gym/envs/              # 标准环境实现
│   │   ├── classic_control/   # 经典控制环境（CartPole等）
│   │   ├── mujoco/            # 物理仿真环境
│   │   └── toy_text/          # 简���文本环境
│   └── tests/                 # 完整测试套件
│
├── 🚀 cartPole/               # CartPole算法实现
│   ├── rl1_cartPole.py       # Q-learning + 状态离散化
│   ├── dqn_cartPole.py       # 深度Q网络
│   └── netAgent.py           # 神经网络智能体
│
├── 📈 policyGradient/         # 策略梯度算法
│   ├── cartpole_pg.py        # REINFORCE算法
│   ├── cartpole_ac.py        # Actor-Critic算法
│   └── policyGradient_agent.py # 基础策略梯度智能体
│
├── 🎴 21Card/                 # 21点游戏自定义环境
│   ├── 21card_v0.py          # 游戏环境实现
│   └── 21Card.py             # 游戏逻辑
│
├── 🎲 mdp/                    # 马尔可夫决策过程
│   └── mdp_v0.py             # MDP环境实现
│
├── 📊 plot/                   # 可视化工具
│   ├── sbus_data_simulator.py # 数据仿真
│   └── sbus_realtime_plotter.py # 实时绘图
│
└── 🔧 unit/                   # 工具模块
    └── softmax_un.py         # Softmax工具函数
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone [repository-url]
cd gymlab

# 安装基础依赖
pip install numpy matplotlib gymnasium torch

# 安装完整Gym环境（可选）
cd gym
pip install -e .[classic_control,box2d,toy_text]
```

### 运行示例

```bash
# 🚀 DQN算法训练CartPole
cd cartPole
python dqn_cartPole.py

# 📈 策略梯度算法训练
cd ../policyGradient
python cartpole_pg.py

# 🎭 Actor-Critic算法训练
python cartpole_ac.py
```

## 🧠 算法实现

### 已实现算法

| 算法类型 | 具体算法 | 实现位置 | 适用环境 |
|---------|---------|---------|---------|
| **值函数方法** | Q-Learning | `cartPole/rl1_cartPole.py` | CartPole |
| | Deep Q-Network | `cartPole/dqn_cartPole.py` | CartPole |
| **策略梯度方法** | REINFORCE | `policyGradient/cartpole_pg.py` | CartPole |
| | Actor-Critic | `policyGradient/cartpole_ac.py` | CartPole |
| **自定义环境** | 21点游戏 | `21Card/` | 21Card-v0 |
| | MDP基础 | `mdp/` | MDP-v0 |

### 算法特点

- **🔄 完整训练流程**: 从环境初始化到模型保存的完整pipeline
- **📊 性能监控**: 实时显示训练进度和奖励变化
- **💾 模型持久化**: 自动保存最佳模型和最终模型
- **🎯 超参数调优**: 通过配置字典轻松调整超参数
- **📈 可视化支持**: 训练曲线和性能对比图表

## 🎓 学习笔记

### 核心概念实现

1. **环境交互模式**
   ```python
   state = env.reset()[0]      # 重置环境
   action = agent.choose_action(state)  # 选择动作
   next_state, reward, terminated, truncated, info = env.step(action)
   ```

2. **经验存储与学习**
   - DQN: 经验回放池 + 目标网络更新
   - Policy Gradient: 轨迹存储 + 策略梯度上升
   - Actor-Critic: 同时学习策略和价值函数

3. **探索与利用平衡**
   - ε-greedy策略 (DQN)
   - 随机策略采样 (Policy Gradient)

## 📊 性能对比

各算法在CartPole-v1环境上的典型表现：

| 算法 | 平均奖励 | 训练稳定性 | 收敛速度 |
|------|---------|-----------|---------|
| Q-Learning | ~150 | 中等 | 较快 |
| DQN | ~200+ | 良好 | 中等 |
| REINFORCE | ~180 | 中等 | 中等 |
| Actor-Critic | ~200+ | 良好 | 较快 |

## 🛠️ 开发工具

```bash
# 代码格式化
black cartPole/ policyGradient/

# 类型检查
pyright cartPole/

# 运行测试
cd gym && python -m pytest tests/ -v

# 预提交检查
pre-commit run --all-files
```

## 📚 学习资源

- [OpenAI Gym文档](https://www.gymlibrary.dev/)
- [强化学习入门](https://hrl.boyuai.com/)
- [DeepMind RL课程](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series)

## 🤝 贡献指南

欢迎提出Issue和PR！本项目主要用于学习实践，特别欢迎：

- 🐛 Bug修复和改进建议
- ✨ 新算法实现
- 📝 文档完善
- 🎨 可视化改进

## 📄 许可证

MIT License - 可自由学习和使用

---

**📌 学习提示**: 建议按照以下顺序学习实践：
1. 先运行`cartPole/rl1_cartPole.py`理解基础Q-Learning
2. 体验`cartPole/dqn_cartPole.py`学习深度强化学习
3. 实现`policyGradient/`中的策略梯度算法
4. 尝试自定义环境`21Card/`和`mdp/`

**⭐ 如果这个项目对您的强化学习学习有帮助，请给个Star支持！**
