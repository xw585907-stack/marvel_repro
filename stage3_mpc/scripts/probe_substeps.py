"""探查 test4 中一个控制步内的子步动力学（为什么全拉命令下躯干只缓慢下降）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.mpc import SRB_MPC
from envs.ik import two_link_ik

G = 9.81


def run_to(env, mpc, n_steps):
    sched = np.ones((mpc.H, 4), dtype=bool)
    for k in range(n_steps):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        k_n = env.contact_model.k_n
        k_t = env.contact_model.k_t
        f_mag = float(env.f_max[0])
        anchor = env.contact_model.anchor[0]
        target_w = np.stack([anchor[:, 0] - u[:, 0] / k_t,
                             -(u[:, 1] + f_mag) / k_n], axis=-1)
        d = target_w - env.p[0]
        rb = np.stack([d[:, 0], d[:, 1]], axis=-1) - env.hip_offsets
        env.step(np.zeros((1, 8)), np.ones((1, 4)), two_link_ik(rb).ravel()[None, :])


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    env.p[0, 1] = 0.32
    mpc = SRB_MPC()
    run_to(env, mpc, 20)                    # 到 k=20 的状态
    snap = env.snapshot()
    print(f'k=20 起点: z={env.p[0, 1]:.4f}, v_z={env.v[0, 1]:.4f}, '
          f'gap={np.round(env.gap[0] * 1000, 3)}mm')

    # 求解第 21 步的命令，然后手动跑 100 个子步并记录
    u = mpc.solve(snap, np.ones((mpc.H, 4), dtype=bool), 0.0, (0.0, -G))
    k_n = env.contact_model.k_n
    k_t = env.contact_model.k_t
    f_mag = float(env.f_max[0])
    anchor = env.contact_model.anchor[0]
    target_w = np.stack([anchor[:, 0] - u[:, 0] / k_t,
                         -(u[:, 1] + f_mag) / k_n], axis=-1)
    d = target_w - env.p[0]
    rb = np.stack([d[:, 0], d[:, 1]], axis=-1) - env.hip_offsets
    q_des = two_link_ik(rb).ravel()[None, :]
    print(f'u = {np.round(u, 1)}, 目标 delta = {np.round((u[:, 1] + f_mag) / k_n * 1e6, 2)}um')
    print(f'q_des 目标足端 z = {target_w[:, 1]}')

    g_vec = np.stack([-G * np.sin(env.gravity_theta),
                      -G * np.cos(env.gravity_theta)], axis=-1)
    orig = env._substep
    def logged(cmd, mc, qd, gv):
        orig(cmd, mc, qd, gv)
        if env._sub_cnt % 10 == 0:
            print(f'  sub {env._sub_cnt:3d}: z={env.p[0, 1]:.5f} '
                  f'v_z={env.v[0, 1]:.4f} gap={env.gap[0, 0] * 1000:.3f}mm '
                  f'f_spring={env.contact_model.n_load[0, 0] if hasattr(env.contact_model, "n_load") else np.nan:.0f} '
                  f'F_env_z 和={env.F_env[0, :, 1].sum():.1f} '
                  f'f_mag={env.f_mag[0, 0]:.0f} q0={env.q[0, 0]:.3f} '
                  f'qd0={env.qd[0, 0]:.3f} foot_z={env.snapshot()["foot_pos"][0, 1] * 1000:.3f}mm')
        env._sub_cnt += 1
    env._sub_cnt = 0
    env._substep = logged
    for s in range(100):
        logged(np.zeros((1, 8)), np.ones((1, 4)), q_des, g_vec)
    env._substep = orig


if __name__ == '__main__':
    main()
