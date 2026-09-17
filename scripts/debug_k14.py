"""k=14 QP：coneqp 返回解 vs 上一步计划 vs 均匀解的目标对比，
并从不同初值重解验证 coneqp 解的唯一性。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
from cvxopt import matrix, solvers

from envs.env import ClimbEnv
from control.mpc import SRB_MPC
from scripts.debug_mpc import realize_step

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    snap = None
    u_prev_plan = None
    for k in range(15):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        if k == 13:
            u_prev_plan = mpc.u_prev.copy()   # 第 13 步计划（第 14 步的锚）
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        if k == 13:
            print(f'k=13 结束: z={env.p[0,1]:.4f} vz={env.v[0,1]:+.3f} '
                  f'u0={np.round(u.ravel(), 1)}')

    # ---- 重建 k=14 的 QP（与 solve() 完全一致）----
    H = mpc.H
    SC = 1000.0
    p = np.asarray(snap['base_pos'], float)
    v = np.asarray(snap['base_vel'], float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], float)
    A_, B_, _ = mpc._dynamics(foot, p, phi)
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, (0.0, -G), s0)
    S_ref = np.zeros((6 * H,))
    for kk in range(1, H + 1):
        S_ref[(kk - 1) * 6:(kk - 1) * 6 + 6] = [
            p[0], mpc.z_nom, 0.0, 0.0, 0.0, 0.0]
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    E_first = np.zeros((8 * H, 8 * H))
    E_first[:8, :8] = 1.0
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * mpc.Rd * (D.T @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    P += 2.0 * mpc.Rd * E_first
    q[:8] += -2.0 * mpc.Rd * u_prev_plan[:8]
    Ps = SC * SC * P
    qs = SC * q

    def fun(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    x_ret = mpc.last_plan.ravel() / SC          # coneqp 返回的 k=14 计划
    print(f'k=14 返回解: obj={fun(x_ret):.3f} '
          f'Fz0={x_ret[1::2].sum()*SC:+.1f}')
    # 上一步计划（Δu_0=0 锚点，z 计划几乎相同）
    x_hold = u_prev_plan / SC
    print(f'上一步计划:  obj={fun(x_hold):.3f} '
          f'Fz0={x_hold[1::2].sum()*SC:+.1f}')
    # 均匀解：u_x=0，u_z=Fz0/4 均摊，全程 mg/4 附近
    x_uni = np.zeros(8 * H)
    for kk in range(H):
        x_uni[kk * 8 + 1::8] = G * mpc.m / 4 / SC * 1.0
    print(f'均匀 mg/4:    obj={fun(x_uni):.3f}')
    # 目标分量拆解
    for name, xx in (('返回解', x_ret), ('上步计划', x_hold)):
        traj = (M_u @ xx * SC + b0).reshape(H, 6)
        zc = mpc.Q[1, 1] * ((traj[:, 1] - mpc.z_nom) ** 2).sum()
        vc = mpc.Q[4, 4] * (traj[:, 4] ** 2).sum()
        rc = mpc.R * (xx * SC) @ (xx * SC)
        du = np.diff(xx.reshape(H, 8), axis=0).ravel() * SC
        du0 = (xx[:8] - u_prev_plan[:8] / SC) * SC
        dc = mpc.Rd * (du0 @ du0 + du @ du)
        print(f'  {name}: z={zc:.3f} vz={vc:.3f} R={rc:.4f} Δu={dc:.3f}')

    # ---- 不同初值重解同一个 QP，检查 coneqp 解是否唯一 ----
    G_rows, h_rows, A_rows, b_rows = mpc._constraints(sched, SC)
    Gm = matrix(np.stack(G_rows), tc='d')
    hm = matrix(np.array(h_rows), tc='d')
    x_true = None
    for name, x0 in (('零初值', np.zeros(8 * H)),
                     ('上步计划', u_prev_plan),
                     ('返回解', mpc.last_plan.ravel())):
        res = solvers.coneqp(matrix(Ps, tc='d'), matrix(qs, tc='d'),
                             Gm, hm, {'l': len(G_rows), 'q': [], 's': []},
                             initvals={'x': matrix(x0 / SC, tc='d')})
        xs = np.array(res['x']).ravel()
        x_true = xs
        print(f'{name} 重解: status={res["status"]} it={res["iterations"]} '
              f'obj={fun(xs):.3f} u0={np.round(xs[:8]*SC, 1)}')

    # 真最优的目标分量拆解（含线性项）
    print('\n真最优分量拆解:')
    traj = (M_u @ x_true * SC + b0).reshape(H, 6)
    zc = mpc.Q[1, 1] * ((traj[:, 1] - mpc.z_nom) ** 2).sum()
    vc = mpc.Q[4, 4] * (traj[:, 4] ** 2).sum()
    pc = mpc.Q[2, 2] * (traj[:, 2] ** 2).sum()
    wc = mpc.Q[5, 5] * (traj[:, 5] ** 2).sum()
    rc = mpc.R * (x_true * SC) @ (x_true * SC)
    du = np.diff(x_true.reshape(H, 8), axis=0).ravel() * SC
    du0 = (x_true[:8] - u_prev_plan[:8] / SC) * SC
    dc = mpc.Rd * (du0 @ du0 + du @ du)
    lin = qs @ x_true
    print(f'  z={zc:.3f} vz={vc:.3f} phi={pc:.3f} w={wc:.3f} '
          f'R={rc:.3f} Δu={dc:.3f} 线性项={lin:.3f} obj={fun(x_true):.3f}')
    print(f'  各步 Fz和={np.round(x_true.reshape(H, 8)[:, 1::2].sum(1)*SC, 1)}')
    print(f'  各步 Fx和={np.round(x_true.reshape(H, 8)[:, 0::2].sum(1)*SC, 1)}')
    print(f'  φ轨迹={np.round(traj[:, 2], 4)}')


if __name__ == '__main__':
    main()
