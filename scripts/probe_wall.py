"""竖直墙原地保持探针：逐步打印 MPCController 闭环，定位倾倒过程。

对应 test_mpc.py 场景3（θ=90°，v_cmd=0）——爬壁论文的核心场景。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.controller import MPCController


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.0)
    for k in range(120):                       # 2.4s（测试中 1.8s 终止）
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        s = env.snapshot()
        if k % 5 == 0 or k < 15 or (k > 60 and s['terminated']):
            u = ctrl.info['u']
            print(f'k={k:3d} t={k*0.02:4.2f} '
                  f'p=({s["base_pos"][0]:+.3f},{s["base_pos"][1]:+.3f}) '
                  f'v=({s["base_vel"][0]:+.3f},{s["base_vel"][1]:+.3f}) '
                  f'phi={s["base_phi"]:+.3f} om={s["base_omega"]:+.3f} '
                  f'u_z和={u[:,1].sum():+7.1f} u_x和={u[:,0].sum():+7.1f} '
                  f'Fz={s["F_env"][:,1].sum():+7.1f} '
                  f'gap={np.round(env.gap[0]*1000,1)}mm '
                  f'stance={ctrl.info["stance"].astype(int)} '
                  f'[{ctrl.mpc.last_msg}]')
        if s['terminated']:
            print(f'终止于 k={k} ({k*0.02:.2f}s)')
            break


if __name__ == '__main__':
    main()
