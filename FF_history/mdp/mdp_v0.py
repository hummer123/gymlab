import os

S = [i for i in range(16)] # 状态空间 0-15 表示16个状态
A = ["n", "e", "s", "w"]   # 动作空间 北、东、南、西

# 行为对状态的改变 
ds_actions = {"n": -4, "e": 1, "s": 4, "w": -1}



def dynamics(s, a):
    ''' 模拟小型方格世界的环境动力学特征
    Args:
        s: state int 0 - 15 表示16个状态
        a: action str in ['n', 'e', 's', 'w'] 表示4个动作：北、东、南、西
    Returns: tuple (s_next, reward, done)
        s_next: int 下一个状态
        reward: float 奖励值
        done: bool 是否进入终止状态
    '''
    s_next = s
    if (s%4 == 0 and a == "w") or (s<4 and a == "n") \
         or ((s+1)%4 == 0 and a == "e") or (s>11 and a == "s") \
         or s in [0, 15]:
        pass
    else:
        ds = ds_actions[a]
        s_next = s + ds
    reward = 0 if s in [0, 15] else -1
    done = True if s in [0, 15] else False

    return s_next, reward, done

def P(s, a, s_next):
    ''' 状态转移概率函数
    Args:
        s: state int 0 - 15 表示16个状态
        a: action str in ['n', 'e', 's', 'w'] 表示4个动作：北、东、南、西
        s_next: state int 0 - 15 表示16个状态
    Returns:
        prob: float 状态转移概率
    '''
    s_next_, _, _ = dynamics(s, a)
    return s_next_ == s_next

def R(s, a):
    ''' 奖励函数
    Args:
        s: state int 0 - 15 表示16个状态
        a: action str in ['n', 'e', 's', 'w'] 表示4个动作：北、东、南、西
    Returns:
        reward: float 奖励值
    '''
    _, reward, _ = dynamics(s, a)
    return reward


gamma = 1.00  # 折扣因子
MDP = S, A, R, P, gamma # MDP元组

# == 1/4 
def uniform_random_policy(MDP, V = None, s = None, a = None):
    ''' 均匀随机策略
    Args:
        MDP: tuple (S, A, R, P, gamma) MDP元组
        V: dict 状态值函数
        s: state int 当前状态
        a: action str 当前动作
    Returns:
        prob: float 动作概率
    '''
    _, A, _, _, _ = MDP
    n = len(A)
    return 0 if n == 0 else 1.0/n

# == 贪婪策略
def greedy_policy(MDP, V, s, a):
    ''' 贪婪策略
    Args:
        MDP: tuple (S, A, R, P, gamma) MDP元组
        V: dict 状态值函数
        s: state int 当前状态
        a: action str 当前动作
    Returns:
        prob: float 动作概率
    '''
    _, A, _, _, _ = MDP
    max_v, a_max_v = -float("inf"), []
    # 找到在状态s下使得状态值函数V最大的动作
    for a_opt in A:
        s_next, _, _ = dynamics(s, a_opt)
        v_s_next = get_value(V, s_next)
        # print(f"   - In({s}, {a}), Action: {a_opt}, V({s_next}): {v_s_next}, ", end="")
        if v_s_next > max_v:
            max_v = v_s_next
            a_max_v = [a_opt]
        elif(v_s_next == max_v):
            a_max_v.append(a_opt)
        # print(f"max_v: {max_v}, a_max_v: {a_max_v}")
    n = len(a_max_v)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_v else 0.0


# ==== 辅助函数 ====
def get_pi(Pi, s, a, MDP, V):
    return Pi(MDP, V, s, a)

def get_prob(P, s, a, s_next):
    return P(s, a, s_next)

def get_reward(R, s, a):
    return R(s, a)

def get_value(V, s):
    return V[s]

def set_value(V, s, value):
    V[s] = value

def display_V(V):
    ''' 打印状态值函数V
    Args:
        V: dict 状态值函数
    '''
    for i in range(16):
        print(f"{V[i]:6.2f}", end=" ")
        if (i+1) % 4 == 0:
            print("")
    print()


#==== 策略计算&评估 ====
def compute_q(MDP, V, s, a):
    ''' 根据给定的 MDP，状态值函数 V，状态 s 和动作 a 计算动作值函数 Q(s,a)
    Args:
        MDP: tuple (S, A, R, P, gamma) MDP元组
        V: dict 状态值函数
        s: state int 当前状态
        a: action str 当前动作
    Returns:
        q_sa: float 动作值函数Q(s,a)
    '''
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_next in S:
        q_sa += get_prob(P, s, a, s_next) * get_value(V, s_next)
        q_sa = get_reward(R, s, a) + gamma * q_sa
    return q_sa

def compute_v(MDP, V, Pi, s):
    ''' 根据给定的 MDP，状态值函数 V，策略 Pi 和状态 s 计算状态值函数 V(s)
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        Pi: function 策略函数
        s: state int 当前状态
    Returns:
        v_s: float 状态值函数V(s)
    '''
    _, A, _, _, _ = MDP
    v_s = 0
    for a in A:
        # print(f"  - Pi({s}, {a}) computing V({s})")
        v_s += get_pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return v_s

def update_V(MDP, V, Pi):
    ''' 根据给定的 MDP，状态值函数 V 和策略 Pi 更新状态值函数 V(s)
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        Pi: function 策略函数
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        print(f"  - Computing V({s})")
        set_value(V_prime, s, compute_v(MDP, V_prime, Pi, s))
        display_V(V_prime)
    return V_prime

def policy_evaluation(MDP, V, Pi, n):
    ''' 策略评估
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        Pi: function 策略函数
        n: int 迭代次数
    Returns:
        V: dict 更新后的状态值函数
    '''
    for _ in range(n):
        V = update_V(MDP, V, Pi)
    return V

def policy_iterate(MDP, V, Pi, n, m):
    ''' 策略迭代
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        Pi: function 策略函数
        n: int 策略评估的迭代次数
        m: int 策略迭代的次数
    Returns:
        V: dict 更新后的状态值函数
        Pi: function 更新后的策略函数
    '''
    for _ in range(m):
        V = policy_evaluation(MDP, V, Pi, n)
    return V, Pi

def compute_v_from_max_q(MDP, V, s):
    ''' 根据给定的 MDP，状态值函数 V 和状态 s 计算状态值函数 V(s)（基于最大动作值函数）
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        s: state int 当前状态
    Returns:
        v_s: float 状态值函数V(s)
    '''
    _, A, _, _, gamma = MDP
    max_q = -float("inf")
    for a in A:
        q_sa = compute_q(MDP, V, s, a)
        if q_sa > max_q:
            max_q = q_sa
    return max_q

def update_V_without_pi(MDP, V):
    ''' 不依赖策略的情况下，直接通过后续状态的价值来更新状态值函数 V(s)
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
    Returns:
        V: dict 更新后的状态值函数
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        set_value(V_prime, s, compute_v_from_max_q(MDP, V_prime, s))
    return V_prime

def value_iterate(MDP, V, n):
    ''' 价值迭代
    Args:
        MDP: tuple (S, A, P, R, gamma) MDP元组
        V: dict 状态值函数
        n: int 迭代次数
    Returns:
        V: dict 更新后的状态值函数
    '''
    for _ in range(n):
        V = update_V_without_pi(MDP, V)
    return V


#==== 策略评估 ====
# V = [0 for _ in range(16)] 
# V_pi = policy_evaluation(MDP, V, uniform_random_policy, 16)
# print(f"\n=== Uniform Random Policy ===")
# display_V(V_pi)

V = [0 for _ in range(16)] 
V_pi = policy_evaluation(MDP, V, greedy_policy, 100)
display_V(V_pi)

# #=== 策略迭代 ====
# V = [0 for _ in range(16)] 
# V_pi, Pi = policy_iterate(MDP, V, greedy_policy, 1, 100)
# display_V(V_pi)

# #=== 价值迭代 ====
# V = [0 for _ in range(16)] 
# V_star = value_iterate(MDP, V, 4)
# display_V(V_star)





