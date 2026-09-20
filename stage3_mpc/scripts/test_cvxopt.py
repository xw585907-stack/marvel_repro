"""测试 cvxopt coneqp 求解 k=3 QP：精度（vs SLSQP 参考）与耗时。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
from cvxopt import matrix, solvers

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController
from stage3_mpc.control.mpc import SRB_MPC

G = 9.81
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 100


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(3):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])

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

    # ---- cvxopt 约束组装（缩放问题，全部线性约束）----
    G_rows, h_rows = [], []
    A_rows, b_rows = [], []
    for kk in range(H):
        for i in range(4):
            idx = (kk * 4 + i) * 2
            if sched[kk, i]:
                # ±t ≤ μ(n + a) → t − μn ≤ μa, −t − μn ≤ μa
                r1 = np.zeros(8 * H); r1[idx] = 1.0; r1[idx + 1] = -ref.mu
                r2 = np.zeros(8 * H); r2[idx] = -1.0; r2[idx + 1] = -ref.mu
                G_rows += [r1, r2]
                h_rows += [0.98 * ref.mu * ref.f_mag / SC] * 2
                # 盒子 −f_mag ≤ n ≤ f_n_max
                r3 = np.zeros(8 * H); r3[idx + 1] = -1.0
                r4 = np.zeros(8 * H); r4[idx + 1] = 1.0
                G_rows += [r3, r4]
                h_rows += [ref.f_mag / SC, ref.f_n_max / SC]
            else:
                for j in (0, 1):
                    row = np.zeros(8 * H); row[idx + j] = 1.0
                    A_rows.append(row); b_rows.append(0.0)
    Gm = matrix(np.stack(G_rows))
    hm = matrix(np.array(h_rows))
    Am = matrix(np.stack(A_rows)) if A_rows else None
    bm = matrix(np.array(b_rows)) if A_rows else None
    Pm = matrix(Ps)
    qm = matrix(qs)

    # ---- 求解 + 计时 ----
    dims = {'l': len(G_rows), 'q': [], 's': []}
    t0 = time.perf_counter()
    res = solvers.coneqp(Pm, qm, Gm, hm, dims, Am, bm,
                         initvals={'x': matrix(x0s)})
    dt_ms = (time.perf_counter() - t0) * 1e3
    x_cvx = np.array(res['x']).ravel()
    print(f'cvxopt 状态: {res["status"]}, 耗时 {dt_ms:.1f} ms, '
          f'迭代 {res.get("iterations", "?")}')
    print(f'cvxopt u0 = {np.round(x_cvx[:8] * SC, 2)}')

    def fobj(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    print(f'f(cvxopt) = {fobj(x_cvx):.8f}  （SLSQP 参考 = -232.08）')

    # 可行性检查
    viol = (np.stack(G_rows) @ x_cvx - np.array(h_rows)).max()
    print(f'最大约束违反 = {viol:.2e} (应 ≤ 0)')

    # 连跑 20 次测稳态耗时（同一起点）
    t0 = time.perf_counter()
    for _ in range(20):
        solvers.coneqp(Pm, qm, Gm, hm, dims, Am, bm,
                       initvals={'x': matrix(x0s)})
    print(f'20 次平均 {((time.perf_counter() - t0) * 1e3) / 20:.1f} ms')


if __name__ == '__main__':
    main()
