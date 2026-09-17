"""两连杆腿逆运动学（与 envs/robot.py 的正运动学约定一致）。"""
import numpy as np


def two_link_ik(r, l1=0.2, l2=0.2):
    """足端相对髋的目标位置 → 关节角 (qh, qk)。

    r: (..., 2) 体坐标（x 前，z 上，足端在髋下方时 z<0）
    约定：qh=0 腿竖直向下，qk<0 膝盖向后弯（同 robot.py：
    x = l1·sin(qh) + l2·sin(qh+qk)，z = −l1·cos(qh) − l2·cos(qh+qk)）
    返回 (..., 2) [qh, qk]；目标不可达时钳制到最大伸展。
    """
    x, z = r[..., 0], r[..., 1]
    d = np.sqrt(x * x + z * z)
    cos_qk = np.clip((d * d - l1 * l1 - l2 * l2) / (2.0 * l1 * l2), -1.0, 1.0)
    qk = -np.arccos(cos_qk)
    psi = np.arctan2(x, -z)
    beta = np.arctan2(l2 * np.sin(qk), l1 + l2 * np.cos(qk))
    qh = psi - beta
    return np.stack([qh, qk], axis=-1)
