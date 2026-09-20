"""探查罚函数牛顿法的失速：调用真实 _solve_qp（trace=True）逐迭代打印。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController
from stage3_mpc.control.mpc import SRB_MPC

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(3):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])

    # 重建 k=3 的 QP（与 probe_qp_opt 相同的构建）
    snap = env.snapshot()
    mpc = ctrl.mpc
    t = ctrl.gait.t
    sched = ctrl.gait.contact_schedule(t, mpc.H, mpc.dt)
    g_vec = (-G * np.sin(snap['theta']), -G * np.cos(snap['theta']))
    ref = SRB_MPC()
    ref.u_prev = mpc.u_prev.copy()
    p = snap['base_pos']; v = snap['base_vel']
    s0 = np.concatenate([p, [snap['base_phi']], v, [snap['base_omega']]])
    M_u, b0 = ref._build_prediction(snap['foot_pos'], p, sched, g_vec, s0)
    H = ref.H
    S_ref = np.zeros((6 * H,))
    for kk in range(1, H + 1):
        S_ref[(kk - 1) * 6:kk * 6] = [p[0] + 0.3 * kk * ref.dt, ref.z_nom,
                                      0.0, 0.3, 0.0, 0.0]
    Qbar = np.kron(np.eye(H), ref.Q)
    Rbar = ref.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    Rdbar = ref.Rd * np.eye(8 * H)
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar + D.T @ Rdbar @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    v0 = np.zeros(8 * H); v0[:8] = ref.u_prev[:8]
    q += -2.0 * (D.T @ Rdbar @ D) @ v0
    SC = 1000.0
    Ps, qs = SC * SC * P, SC * q
    x0s = np.concatenate([ref.u_prev[8:], ref.u_prev[-8:]]) / SC

    eigs = np.linalg.eigvalsh(Ps)
    print(f'Ps 特征值范围: [{eigs.min():.2e}, {eigs.max():.2e}] '
          f'(条件数 {eigs.max() / eigs.min():.1e})')

    x_sol, nit = ref._solve_qp(Ps, qs, sched, x0s, SC, max_iter=200,
                               trace=True)
    print(f'最终 u0 = {np.round(x_sol[:8] * SC, 2)} (it={nit})')


if __name__ == '__main__':
    main()
