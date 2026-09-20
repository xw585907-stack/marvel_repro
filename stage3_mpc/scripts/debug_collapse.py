"""定位 k=12 力实现崩溃：重放前 12 步，第 12 步逐子步打印内部状态。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.mpc import SRB_MPC
from scripts.debug_mpc import realize_step

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    for k in range(80):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        tau, q_des, target_w = realize_step(env, u, snap)
        if k == 12:
            # 手动重放第 12 步的 100 个子步，打印关键子步
            foot_start, _ = env._foot_state()
            target_v = (target_w - foot_start) / env.control_dt
            print(f'步 12 开始: z={env.p[0,1]:.5f} vz={env.v[0,1]:+.4f} '
                  f'phi={env.phi[0]:+.4f} u_z和={u[:,1].sum():.1f}')
            for s_i in range(env.substeps):
                frac = (s_i + 1) / env.substeps
                tw = foot_start + (target_w - foot_start) * frac
                q_sub, qd_sub = env._world_ik(tw, target_v)
                env._substep(tau[None, :], np.ones((1, 4)), q_sub,
                             np.stack([0.0, -G]), qd_sub)
                if s_i in (0, 1, 2, 5, 10, 20, 50, 99):
                    print(f'  s={s_i:3d} z={env.p[0,1]:.5f} vz={env.v[0,1]:+.4f} '
                          f'F_env_z={np.round(env.F_env[0,:,1], 1)} '
                          f'gap={np.round(env.gap[0]*1000, 1)}mm')
                    print(f'      q1={np.round(env.q[0,::2], 4)} '
                          f'q2={np.round(env.q[0,1::2], 4)}')
                    print(f'      qd1={np.round(env.qd[0,::2], 3)} '
                          f'qd2={np.round(env.qd[0,1::2], 3)}')
        else:
            env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        if k % 5 == 0 and k <= 12:
            print(f'k={k:2d} z={env.p[0,1]:.5f} vz={env.v[0,1]:+.4f} '
                  f'u_z和={u[:,1].sum():.1f} F_env_z和={env.F_env[0,:,1].sum():.1f}')


if __name__ == '__main__':
    main()
