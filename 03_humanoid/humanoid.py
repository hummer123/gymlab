"""Humanoid-v5 的局部站立控制，无需强化学习训练。

运行演示：python humanoid.py
无窗口验证：python humanoid.py --headless --episodes 10 --seed 0

仅依赖 Gymnasium、MuJoCo 和 NumPy，不需要 SciPy。
先用逆动力学求双脚着地的平衡点，再线性化环境并求有限时域离散 LQR：
    action = equilibrium_action - K @ [position_error, velocity]

每次 reset 从平衡点附近开始，默认扰动为 0.001；动作始终受环境限幅。
该算法用于小扰动下保持站立，不包含倒地起身或行走策略。
"""

import argparse

import gymnasium as gym
import mujoco
import numpy as np


class StandingController:
    """针对标准 Humanoid-v5 模型计算一次反馈增益。"""

    def __init__(self, env):
        self.model = env.unwrapped.model
        self.frame_skip = env.unwrapped.frame_skip
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high
        self._scratch = mujoco.MjData(self.model)
        self._zero_velocity = np.zeros(self.model.nv)
        self.reference_qpos, self.reference_action = self._find_equilibrium()
        self.gain = self._compute_gain()

    def _state_error(self, reference_qpos, reference_qvel, qpos, qvel):
        # 浮动基座含四元数，不能直接将两个 qpos 相减作为角度误差。
        position_error = np.empty(self.model.nv)
        mujoco.mj_differentiatePos(
            self.model, position_error, 1.0, reference_qpos, qpos
        )
        return np.concatenate((position_error, qvel - reference_qvel))

    def _find_equilibrium(self):
        model, data = self.model, self._scratch
        reference = model.qpos0.copy()
        # 膝盖稍微弯曲，避开模型中 -2° 的伸直限位。
        for name, angle in (
            ("right_hip_y", -0.05),
            ("left_hip_y", -0.05),
            ("right_knee", -0.10),
            ("left_knee", -0.10),
        ):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"Humanoid 模型缺少关节：{name}")
            reference[model.jnt_qposadr[joint]] = angle

        data.qpos[:] = reference
        mujoco.mj_forward(model, data)
        foot = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot"
        )
        if foot < 0:
            raise ValueError("Humanoid 模型缺少 right_foot 接触几何体")
        # 用脚底高度估计接触位置，稍微压入地面以启动软接触求解。
        height = (
            reference[2] - data.geom_xpos[foot, 2]
            + model.geom_size[foot, 0] - 0.001
        )

        def floating_base_force(height_pitch):
            mujoco.mj_resetData(model, data)
            data.qpos[:] = reference
            data.qpos[2] = height_pitch[0]
            pitch = height_pitch[1]
            data.qpos[3:7] = [np.cos(pitch / 2), 0, np.sin(pitch / 2), 0]
            # qvel、qacc 均为零：求保持此姿态需要的广义力。
            mujoco.mj_inverse(model, data)
            # 基座没有电机，支撑必须由重力、接触力和身体姿态平衡。
            return data.qfrc_inverse[:6].copy()

        # 只优化高度和俯仰角，用带回溯的 Gauss-Newton 法求解。
        height_pitch = np.array([height, 0.0])
        for _ in range(30):
            residual = floating_base_force(height_pitch)
            residual_norm = np.linalg.norm(residual)
            if residual_norm < 0.001:
                break
            jacobian = np.empty((6, 2))
            for index, epsilon in enumerate((1e-6, 1e-5)):
                offset = np.zeros(2)
                offset[index] = epsilon
                jacobian[:, index] = (
                    floating_base_force(height_pitch + offset)
                    - floating_base_force(height_pitch - offset)
                ) / (2 * epsilon)
            direction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            for backtrack in range(12):
                candidate = height_pitch + (0.5 ** backtrack) * direction
                if np.linalg.norm(floating_base_force(candidate)) < residual_norm:
                    height_pitch = candidate
                    break
            else:
                break
        # 重新计算，确保 scratch 中保存的是最终选中的姿态。
        residual = floating_base_force(height_pitch)
        if not np.all(np.isfinite(residual)) or np.linalg.norm(residual) > 0.05:
            raise RuntimeError("未找到站立平衡点：基座受力残差过大")

        # actuator 顺序与 qpos 顺序不同，按传动关节映射并除以 gear。
        joint_ids = model.actuator_trnid[:, 0]
        dofs = model.jnt_dofadr[joint_ids]
        action = data.qfrc_inverse[dofs] / model.actuator_gear[:, 0]
        if not np.all(np.isfinite(action)) or np.any(
            (action < self.action_low) | (action > self.action_high)
        ):
            raise RuntimeError("站立所需的支撑力矩超出环境动作范围")
        return data.qpos.copy(), action.copy()

    def _transition(self, qpos, qvel, action):
        model, data = self.model, self._scratch
        # 使用独立数据，不让线性化过程改变正在运行的仿真。
        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = action
        # 与 env.step 相同：一次动作保持 frame_skip 个物理步长。
        mujoco.mj_step(model, data, nstep=self.frame_skip)
        return data.qpos.copy(), data.qvel.copy()

    def _compute_gain(self):
        model = self.model
        state_size = 2 * model.nv
        epsilon = 1e-6
        a_matrix = np.empty((state_size, state_size))
        b_matrix = np.empty((state_size, model.nu))

        def transition_difference(qminus, vminus, uminus, qplus, vplus, uplus):
            before = self._transition(qminus, vminus, uminus)
            after = self._transition(qplus, vplus, uplus)
            return self._state_error(*before, *after) / (2 * epsilon)

        for index in range(state_size):
            qplus = self.reference_qpos.copy()
            qminus = self.reference_qpos.copy()
            vplus = self._zero_velocity.copy()
            vminus = self._zero_velocity.copy()
            if index < model.nv:
                tangent = np.zeros(model.nv)
                tangent[index] = epsilon
                mujoco.mj_integratePos(model, qplus, tangent, 1.0)
                mujoco.mj_integratePos(model, qminus, tangent, -1.0)
            else:
                vplus[index - model.nv] = epsilon
                vminus[index - model.nv] = -epsilon
            a_matrix[:, index] = transition_difference(
                qminus, vminus, self.reference_action,
                qplus, vplus, self.reference_action,
            )

        for index in range(model.nu):
            uplus = self.reference_action.copy()
            uminus = self.reference_action.copy()
            uplus[index] += epsilon
            uminus[index] -= epsilon
            b_matrix[:, index] = transition_difference(
                self.reference_qpos, self._zero_velocity, uminus,
                self.reference_qpos, self._zero_velocity, uplus,
            )

        # 优先约束躯干高度、倾斜角，其次是关节姿态和速度。
        q_cost = np.diag(np.concatenate((
            [5, 5, 200, 500, 500, 50],
            np.full(model.nv - 6, 20),
            np.ones(model.nv),
        )))
        r_cost = 0.1 * np.eye(model.nu)
        try:
            # Riccati 倍增法：每次将规划时域加倍，只需 NumPy 线性求解。
            # 12 次对应 4096 步；标准模型控制周期下约为 61 秒。
            work_a = a_matrix.copy()
            work_g = b_matrix @ np.linalg.solve(r_cost, b_matrix.T)
            value = q_cost.copy()
            identity = np.eye(state_size)
            for _ in range(12):
                system = identity + work_g @ value
                propagated_a = np.linalg.solve(system, work_a)
                next_g = (
                    work_g + work_a @ np.linalg.solve(system, work_g) @ work_a.T
                )
                next_value = value + work_a.T @ value @ propagated_a
                work_a = work_a @ propagated_a
                # 消除浮点舍入带来的非对称误差。
                work_g = (next_g + next_g.T) / 2
                value = (next_value + next_value.T) / 2
            gain = np.linalg.solve(
                r_cost + b_matrix.T @ value @ b_matrix,
                b_matrix.T @ value @ a_matrix,
            )
        except np.linalg.LinAlgError as error:
            raise RuntimeError("站立反馈增益求解失败，请检查模型和接触状态") from error
        if not np.all(np.isfinite(gain)):
            raise RuntimeError("站立反馈增益包含非有限值")
        return gain

    def action(self, data):
        error = self._state_error(
            self.reference_qpos, self._zero_velocity, data.qpos, data.qvel
        )
        if not np.all(np.isfinite(error)):
            raise RuntimeError("仿真状态出现非有限值，停止控制")
        return np.clip(
            self.reference_action - self.gain @ error,
            self.action_low, self.action_high,
        )


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="无窗口运行")
    parser.add_argument("--episodes", type=positive_int, default=1, help="演示轮数")
    parser.add_argument("--steps", type=positive_int, default=1000, help="每轮步数")
    parser.add_argument("--seed", type=int, default=0, help="首轮随机种子")
    args = parser.parse_args()

    env = gym.make(
        "Humanoid-v5",
        ctrl_cost_weight=0.1,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.001,
        max_episode_steps=args.steps,
        render_mode=None if args.headless else "human",
    )
    try:
        base_env = env.unwrapped
        # Newton 收敛精度适合接触动力学的有限差分；保留模型的 RK4 和步长。
        base_env.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        base_env.model.opt.tolerance = 1e-10
        controller = StandingController(env)
        # 仅调整 reset 的初始姿态；运行中完全由受限力矩和物理仿真驱动。
        base_env.init_qpos[:] = controller.reference_qpos
        base_env.init_qvel[:] = 0
        print(
            f"LQR 就绪：平衡高度 {controller.reference_qpos[2]:.3f} m，"
            f"控制周期 {base_env.dt:.3f} s"
        )
        failures = 0
        for episode in range(args.episodes):
            env.reset(seed=args.seed + episode)
            heights = [float(base_env.data.qpos[2])]
            terminated = False
            for step in range(1, args.steps + 1):
                _, _, terminated, truncated, _ = env.step(
                    controller.action(base_env.data)
                )
                heights.append(float(base_env.data.qpos[2]))
                if terminated or truncated:
                    break
            failures += int(terminated)
            status = "跌倒" if terminated else "保持站立"
            print(
                f"第 {episode + 1} 轮（seed={args.seed + episode}）：{status}，"
                f"{step} 步 / {step * base_env.dt:.2f} s，"
                f"躯干高度 {min(heights):.3f}–{max(heights):.3f} m"
            )
        if failures:
            raise SystemExit(f"{failures}/{args.episodes} 轮未能保持站立")
    finally:
        env.close()


if __name__ == "__main__":
    main()
