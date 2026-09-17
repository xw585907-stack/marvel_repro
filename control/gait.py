"""步态调度器：对角小跑 + EPM 5ms 提前开关（论文1 的开关时序）。"""
import numpy as np


class GaitScheduler:
    """对角小跑步态时钟。

    每条腿相位 φ ∈ [0,1)：φ < swing_fraction 为摆动相，其余为支撑相。
    腿序 0=RR, 1=FR, 2=RL, 3=FL（与论文2 观测一致）；对角组 (RR,FL) 与
    (FR,RL) 相位差 0.5。

    EPM 开关时序（论文1）：电永磁开关需要 5ms，因此磁开关命令按
    t + epm_lead 时刻的支撑相表提前给出——落地前 5ms 提前开磁
    （悬空时气隙大、吸力≈0，无副作用），离地前 5ms 提前断磁。
    """

    def __init__(self, period=0.5, swing_fraction=0.45, epm_lead=5e-3):
        self.period = period               # 步态周期 [s]
        self.swing_fraction = swing_fraction
        self.epm_lead = epm_lead           # EPM 开关提前量 [s]（论文1: 5ms）
        self.offsets = np.array([0.0, 0.5, 0.5, 0.0])
        self.t = 0.0

    def reset(self, t=0.0):
        self.t = t

    def advance(self, dt):
        self.t += dt

    def phase(self, t):
        """t 时刻（或时间数组）各腿相位。标量 → (4,)，数组 (...,) → (..., 4)。"""
        t = np.asarray(t, dtype=float)
        return np.mod(t / self.period + self.offsets, 1.0)

    def stance(self, t):
        """t 时刻支撑相 bool。"""
        return self.phase(t) >= self.swing_fraction

    def swing_progress(self, t):
        """t 时刻摆动相归一化进度 s ∈ [0,1]（支撑相时无意义，返回 0）。"""
        ph = self.phase(t)
        return np.where(ph < self.swing_fraction,
                        ph / self.swing_fraction, 0.0)

    def contact_schedule(self, t0, n_steps, dt):
        """MPC 预测时域 [t0, t0 + n_steps·dt) 的接触表 (n_steps, 4) bool。"""
        tt = t0 + np.arange(n_steps) * dt
        return self.stance(tt[:, None])

    def magnet_cmd(self):
        """当前控制步的磁开关命令 (4,) bool：按提前 5ms 的支撑相表给出。"""
        return self.stance(self.t + self.epm_lead)
