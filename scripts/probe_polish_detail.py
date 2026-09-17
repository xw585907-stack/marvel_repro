"""polish 内部计时:每轮 KKT 维度/耗时,coneqp 解与 polish 解的差距。

判断:(1) polish 慢在哪(KKT solve 还是步长搜索);
(2) Rd=1e-4 正则下 coneqp 解是否已经足够好(polish 是否还有必要)。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
from cvxopt import matrix, solvers

from envs.env import ClimbEnv
from control.controller import MPCController

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.15)
    mpc = ctrl.mpc

    # 拿一个真实步的 QP 数据:手动组装(与 solve 相同流程)
    snap = env.snapshot()
    p = np.asarray(snap['base_pos'], float)
    v = np.asarray(snap['base_vel'], float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], float)
    t = ctrl.gait.t
    sched = ctrl.gait.contact_schedule(t, mpc.H, mpc.dt)
    g_vec = (-G * np.sin(np.pi / 2), -G * np.cos(np.pi / 2))
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, g_vec, s0)
    S_ref = mpc._ref_traj(p, 0.15, 0.15, 0.15)
    Qbar = np.kron(np.eye(mpc.H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * mpc.H)
    D = np.eye(8 * mpc.H) - np.eye(8 * mpc.H, k=8)
    E_first = np.zeros((8 * mpc.H, 8 * mpc.H)); E_first[:8, :8] = 1.0
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * mpc.Rd * (D.T @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    P += 2.0 * mpc.Rd * E_first
    q[:8] += -2.0 * mpc.Rd * mpc.u_prev[:8]
    SC = 1000.0
    Ps, qs = SC * SC * P, SC * q
    x0s = np.zeros(8 * mpc.H)
    x0s[:-8] = mpc.u_prev[8:]
    x0s[-8:] = mpc.u_prev[-8:]
    x0s = x0s / SC
    jac = snap['J_w']

    def fun(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    # coneqp
    t0 = time.perf_counter()
    G_rows, h_rows, A_rows, b_rows = mpc._constraints(sched, SC, jac)
    res = solvers.coneqp(
        matrix(Ps, tc='d'), matrix(qs, tc='d'),
        matrix(np.stack(G_rows), tc='d'), matrix(np.array(h_rows), tc='d'),
        {'l': len(G_rows), 'q': [], 's': []},
        matrix(np.stack(A_rows), tc='d') if A_rows else None,
        matrix(np.array(b_rows), tc='d') if A_rows else None,
        initvals={'x': matrix(x0s, tc='d')})
    t_qp = (time.perf_counter() - t0) * 1e3
    x_sol = np.array(res['x']).ravel()
    print(f'coneqp: {t_qp:.1f} ms  it={res["iterations"]} '
          f'status={res["status"]}')
    print(f'coneqp 解: Fz和={x_sol[1:8:2].sum()*SC:+.1f} '
          f'obj={fun(x_sol):.3f}')

    # polish 逐步计时
    nv = len(qs)
    Gm = np.stack(G_rows)
    h = np.array(h_rows)
    A = np.stack(A_rows) if A_rows else np.zeros((0, nv))
    b = np.array(b_rows) if A_rows else np.zeros(0)
    if len(A):
        x = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        x = x_sol.copy()
    f_cur = fun(x)
    eps = 1e-9
    print(f'polish 起点(投影到等式面): obj={fun(x):.3f} '
          f'Fz和={x[1:8:2].sum()*SC:+.1f}')
    for it in range(5):
        t0 = time.perf_counter()
        act = np.where(Gm @ x - h >= -eps)[0]
        W = np.vstack([Gm[act], A]) if len(act) else A
        ncon = len(W)
        g = Ps @ x + qs
        K = np.zeros((nv + ncon, nv + ncon))
        K[:nv, :nv] = Ps
        K[:nv, nv:] = W.T
        K[nv:, :nv] = W
        rhs = np.zeros(nv + ncon)
        rhs[:nv] = -g
        dx = np.linalg.solve(K, rhs)[:nv]
        t_solve = (time.perf_counter() - t0) * 1e3
        alpha = 1.0
        Gd = Gm @ dx
        tight = Gd > 1e-14
        if np.any(tight):
            alpha = min(alpha, np.min((h - Gm @ x)[tight] / Gd[tight]))
        alpha = min(max(alpha, 0.0), 1.0)
        x_new = x + alpha * dx
        f_new = fun(x_new)
        t_all = (time.perf_counter() - t0) * 1e3
        print(f'  it={it} ncon={ncon} KKT={nv+ncon}x{nv+ncon} '
              f'solve={t_solve:.2f}ms 全轮={t_all:.2f}ms alpha={alpha:.4f} '
              f'obj={f_new:.3f} Fz和={x_new[1:8:2].sum()*SC:+.1f}')
        if f_new >= f_cur - 1e-12:
            break
        x, f_cur = x_new, f_new
    print(f'polish 解: Fz和={x[1:8:2].sum()*SC:+.1f} obj={fun(x):.3f}')


if __name__ == '__main__':
    main()
