"""验证力矩前馈 τ_ff = −J_wᵀ·u 的符号与实现率：
全支撑静态持形，命令 u=(30, 19.6)/腿，看 F_env 是否 ≈ u（切向 ~97%）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from envs.robot import foot_jac
from control.ik import two_link_ik


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    # 全支撑：四腿目标 = 当前足端位置（原地钉扎），力矩前馈命令 u
    u_cmd = np.array([[30.0, 19.6]] * 4)
    snap = env.snapshot()
    target_w = snap['foot_pos'].copy()
    q_des = None
    print('k   F_env_x和   F_env_z和   vx      vz      gap[0]mm')
    for k in range(8):
        snap = env.snapshot()
        c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
        R = np.array([[c, -s], [s, c]])
        J = foot_jac(snap['q'], env.l1, env.l2)
        J_w = np.einsum('ij,fjk->fik', R, J)
        tau = -np.einsum('fij,fj->fi', J_w, u_cmd).ravel()
        if q_des is None:
            q_des = np.zeros(8)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :],
                 target_w[None, :, :])
        snap = env.snapshot()
        print(f'{k}   {snap["F_env"][:, 0].sum():+8.1f} '
              f'{snap["F_env"][:, 1].sum():+8.1f}  '
              f'{snap["base_vel"][0]:+.3f} {snap["base_vel"][1]:+.3f} '
              f'{snap["foot_pos"][0, 1] * 1000:.3f}')
    print('期望: F_env_x和 ≈ 120 (=30×4), F_env_z和 ≈ 78.4 (=19.6×4)')


if __name__ == '__main__':
    main()
