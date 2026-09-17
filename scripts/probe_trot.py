"""探查平地小跑闭环前 50 步：求解路径（热启动/冷启动）、迭代数、u、z。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.controller import MPCController


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(300):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        mpc = ctrl.mpc
        s = env.snapshot()
        stance = ctrl.info['stance'].astype(int)
        u = ctrl.info['u']
        if k < 7 or k % 25 == 0:
            print(f'k={k:3d} z={s["base_pos"][1]:.4f} vx={s["base_vel"][0]:+.3f} '
                  f'vz={s["base_vel"][1]:+.3f} phi={s["base_phi"]*1e3:+.1f}mrad '
                  f'stance={stance} '
                  f'u_z和={u[:, 1].sum():+6.1f} u_x和={u[:, 0].sum():+6.1f} '
                  f'F_env_z和={s["F_env"][:, 1].sum():+6.1f} '
                  f'F_env_x和={s["F_env"][:, 0].sum():+6.1f} '
                  f'solve={mpc.solve_ms:5.1f}ms it={mpc.last_nit:2d}')
        if k == 6:
            print(f'  u/腿   = {np.round(u, 1)}')
            print(f'  F_env = {np.round(s["F_env"], 1)}')


if __name__ == '__main__':
    main()
