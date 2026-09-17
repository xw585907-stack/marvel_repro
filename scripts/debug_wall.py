"""调试：垂直墙吸附瞬态，逐子步打印整体受力与四足细节。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv, G

env = ClimbEnv(num_envs=1, seed=0)
env.set_gravity_theta(np.pi / 2)
env.reset()

q_des = env.q_nominal.copy()
torque = np.zeros((1, 8))
magnet = np.ones((1, 4))
g_vec = np.stack([-G * np.sin(env.gravity_theta),
                  -G * np.cos(env.gravity_theta)], axis=-1)

print('substep: F_body(x,z)  torque  | foot gaps(4)  | F_env z(4) | F_env x(4)')
for i in range(20):
    env._substep(torque, magnet, q_des, g_vec)
    foot_w, foot_v = env._foot_state()
    r = foot_w - env.p[:, None, :]
    F_tot = env.F_env.sum(axis=1) + env.mass[:, None] * g_vec
    tau = (r[..., 0] * env.F_env[..., 1] - r[..., 1] * env.F_env[..., 0]).sum(1)
    gaps = np.round(foot_w[0, :, 1] * 1000, 3)
    fz = np.round(env.F_env[0, :, 1], 0)
    fx = np.round(env.F_env[0, :, 0], 0)
    print(f'{i+1:7d}: F=({F_tot[0,0]:8.1f},{F_tot[0,1]:8.1f}) tau={tau[0]:8.1f} '
          f'| {gaps} | {fz} | {fx}')
    if np.abs(env.p[0, 1] - 0.15) > 0.02:
        break
print(f'p={env.p[0]}, v={env.v[0]}, phi={env.phi[0]:.4f}, omega={env.omega[0]:.3f}')
