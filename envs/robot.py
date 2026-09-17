"""2D 矢状面四足机器人：参数与运动学（腿视为无质量连杆）。

腿序与论文2 观测一致：0=RR(右后), 1=FR(右前), 2=RL(左后), 3=FL(左前)。
2D 简化：四条腿投影到同一矢状面，每条腿 2 关节（髋+膝），共 8 关节。
"""
import numpy as np

NUM_LEGS = 4
NUM_JOINTS = 8
LEG_NAMES = ['RR', 'FR', 'RL', 'FL']


class RobotParams:
    """MARVEL 2D 简化模型参数（论文1 Fig.2/Fig.5B）。"""

    mass = 8.0                # 躯干质量 [kg]（论文1: 8 kg）
    inertia = 0.084           # 躯干绕横轴转动惯量 [kg·m²]（330×131mm 箱体近似）
    body_length = 0.33        # 前髋到后髋距离 [m]（论文1: 330 mm）
    hip_height = 0.0          # 髋关节相对质心高度 [m]（2D 简化为同高）
    l1 = 0.2                  # 大腿长 [m]
    l2 = 0.2                  # 小腿长 [m]
    body_height_nom = 0.15    # 标称爬行体高（质心到壁面）[m]（论文1: 0.15 m）

    # 关节 PD 与限制
    # kp=600：足端力通过"目标位置−锚点弹簧偏移"实现，偏移量 u/k_t 只有
    # 0.1-0.3 mm 量级。旧值 kp=60 时关节对每步变化的目标有 ~v/50 的
    # 稳态速度滞后（q_des 每 20ms 更新，kd/kp 比例决定滞后），滞后量
    # ~270 µm 与偏移量同量级 → 切向力实际只实现 ~5%（实测 MPC 推
    # +115 N 而 F_env_x ≈ 0），机器人原地不动。kp=600 后滞后 ~30 µm。
    kp = 600.0                # 关节位置增益 [N·m/rad]
    kd = 5.0                  # 关节速度增益 [N·m·s/rad]（ζ≈0.46）
    tau_max = 30.0            # 关节力矩上限 [N·m]（MPC |Jᵀu| 约束的持续值）
    tau_peak = 150.0          # 执行器峰值 [N·m]（env 限幅用）
    # 落地/离地瞬态时接触负载 Jᵀ·(f_spring−f_mag) 瞬时可达 ~100-250N·m
    # （磁吸 697N×0.37m 力臂）。真实机器人靠足端质量+弹性体吸收该瞬态，
    # 30N·m 只是持续值；无质量足端模型里若按 30 限幅，膝关节被折叠、
    # 足端反向弹飞（实测墙面步态落地腿 gap 弹到 +6mm）。峰值 150 与
    # 磁力滞后 τ=5ms 配合：瞬态负载 ~77-185N·m 全部落在执行器能力内。
    joint_inertia = 0.05      # 虚拟关节惯量（无质量腿按 q̈ = τ/I_v 积分）

    # 标称站立位型：对称折叠，足端在髋正下方 0.15 m（即体高 0.15 m）
    # 2·l1·cos(q_hip) = 0.15 → q_hip = 1.1864 rad
    q_nominal = np.array([1.1864, -2.3728])

    @property
    def hip_offsets(self):
        """髋关节在体坐标系中的位置 (4, 2)。"""
        hx = self.body_length / 2.0
        return np.array([[-hx, self.hip_height],   # RR
                         [hx, self.hip_height],   # FR
                         [-hx, self.hip_height],   # RL
                         [hx, self.hip_height]])  # FL


def foot_pos_body(q, hip_offsets, l1, l2):
    """足端在体坐标系中的位置。

    q: (..., 8)，每腿 [q_hip, q_knee]，q_hip=0 表示腿竖直向下
    返回 (..., 4, 2)：
        x = hip_x + l1·sin(qh) + l2·sin(qh+qk)
        z = hip_z − l1·cos(qh) − l2·cos(qh+qk)
    """
    qh = q[..., 0::2]
    qk = q[..., 1::2]
    x = l1 * np.sin(qh) + l2 * np.sin(qh + qk)
    z = -l1 * np.cos(qh) - l2 * np.cos(qh + qk)
    foot = np.stack([x, z], axis=-1)          # (..., 4, 2)
    return foot + hip_offsets


def foot_jac(q, l1, l2):
    """足端位置对关节角的雅可比 (..., 4, 2, 2)，行=x/z，列=髋/膝。"""
    qh = q[..., 0::2]
    qk = q[..., 1::2]
    J = np.empty(q.shape[:-1] + (4, 2, 2))
    J[..., 0, 0] = l1 * np.cos(qh) + l2 * np.cos(qh + qk)   # ∂x/∂qh
    J[..., 0, 1] = l2 * np.cos(qh + qk)                     # ∂x/∂qk
    J[..., 1, 0] = l1 * np.sin(qh) + l2 * np.sin(qh + qk)   # ∂z/∂qh
    J[..., 1, 1] = l2 * np.sin(qh + qk)                     # ∂z/∂qk
    return J
