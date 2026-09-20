"""真实闭环 z 扰动恢复：平地小跑到稳态后注入 vz=+0.5，观察恢复。

验证 Q_z=3000 的快速 z 反馈在真实力实现路径（力矩前馈 + 世界 IK）
下是否稳定：应把躯干拉回 0.15，不振荡发散。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(500):
        if k == 300:
            env.v[0, 1] += 0.5              # 注入向上速度扰动
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        if (k >= 295 and k <= 330) or k % 25 == 0:
            s = env.snapshot()
            u = ctrl.info['u']
            print(f'k={k:3d} z={s["base_pos"][1]:.4f} vz={s["base_vel"][1]:+.3f} '
                  f'u_z和={u[:,1].sum():+7.1f} F_env_z和={s["F_env"][:,1].sum():+7.1f}')


if __name__ == '__main__':
    main()
