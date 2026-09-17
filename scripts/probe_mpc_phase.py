"""逐拍对比闭环热启动解 vs 同状态同调度冷启动解。

若闭环 u 与冷解一致 → 体高偏置来自状态/调度本身；
若闭环 u 系统性偏离冷解 → coneqp 初值依赖把闭环钉在次优点。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.controller import MPCController

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(250):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        u_cl = ctrl.info['u']                     # 闭环（热启动）命令
        u_prev_save = ctrl.mpc.u_prev.copy()
        snap = env.snapshot()
        # 同状态冷启动解（u_prev=None，同一 t 的调度）
        sched = ctrl.gait.contact_schedule(ctrl.gait.t, ctrl.mpc.H, ctrl.mpc.dt)
        u_cold = ctrl.mpc.solve(snap, sched, 0.3, (0.0, -G))
        # 恢复闭环热启动状态（u_prev 已被冷解覆盖）
        ctrl.mpc.u_prev = u_prev_save
        if k >= 125 and (k % 5 == 0 or k == 250 - 1):
            print(f'k={k:3d} t={ctrl.gait.t:.3f} '
                  f'z={snap["base_pos"][1]:.4f} vz={snap["base_vel"][1]:+.3f} '
                  f'phi={snap["base_phi"]*1e3:+.1f} '
                  f'om={snap["base_omega"]*1e3:+.1f} '
                  f'闭环u_z和={u_cl[:,1].sum():+6.1f} '
                  f'冷解u_z和={u_cold[:,1].sum():+6.1f} '
                  f'差={u_cl[:,1].sum()-u_cold[:,1].sum():+6.1f}')


if __name__ == '__main__':
    main()
