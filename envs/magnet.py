"""EPM（电永磁）吸附模型：论文2 II-B 的判定链 + 气隙-力曲线。

论文2 的四步吸附判定（依次满足才产生吸力）：
  (1) 接触识别   c̃_foot ≥ 0.5（仿真中用几何间隙 gap < 0 判定）
  (2) 磁开关动作 a_magnet ≥ 0.5
  (3) 随机吸附   X ~ U(0,1) ≤ Prob_attach（X 在摆动相采样，Eq.3）
  (4) 几何对齐   S_EPM = S_wall（2D 点足模型下用气隙力曲线近似：
      不完全接触 → 气隙 → 吸力按指数衰减，论文2 Fig.3）
"""
import numpy as np


def magnet_force(gap, attach_ok, magnet_on, f_max, lam):
    """气隙-力曲线：F = F_max·exp(−gap/λ)，gap 钳制到 ≥ 0（穿透时饱和在 F_max）。

    gap: (N,4) 脚到表面的距离 [m]；attach_ok: (N,4) 本次落地吸附是否成功；
    magnet_on: (N,4) 磁开关命令 a≥0.5；f_max: (N,) 最大吸力 [N]；
    lam: (N,) 衰减系数 [m]（论文2 Fig.3：1mm 气隙 → 7%）
    返回 (N,4) 法向吸力 [N]（指向表面，正值）。
    """
    g = np.clip(gap, 0.0, None)
    f = f_max[..., None] * np.exp(-g / lam[..., None])
    return np.where(attach_ok & magnet_on, f, 0.0)


def roll_attach(landing, prob_attach, rng):
    """落地瞬间掷骰子（论文2 Eq.3 的随机吸附）。

    landing: (N,4) 本次 substep 是否刚落地（接触建立）
    prob_attach: (N,) 吸附成功概率（课程调度 1.0→0.85）
    返回 (N,4) bool：本次落地是否吸附成功；失败后整个接触周期无吸力，
    重新抬脚再落地时重掷（与论文"失败→重新尝试吸附"的恢复逻辑一致）。
    """
    x = rng.uniform(size=landing.shape)
    return x <= prob_attach[..., None]
