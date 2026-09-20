"""静止全支撑保持稳态下的冷/热启动解对比。

若冷解 ≈ 74.5（真最优，z 反馈把体高压回 0.15）而热解 = 78.5，
则 coneqp 提前判停确认，加 polish 有效；若冷解也是 78.5，
则体高 0.1542 是其他机制（polish 无效）。
"""
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
        u_prev_before = None if mpc.u_prev is None else mpc.u_prev.copy()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        if 12 <= k <= 14:
            # 崩溃时刻：打印每腿力 + MPC 预测的 z 轨迹与各步合力
            print(f'k={k} u/腿 = {np.round(u, 1)} '
                  f'it={mpc.last_nit} obj={mpc.last_obj[0]:.2f} {mpc.last_msg}')
            traj = (mpc.last_Mu @ mpc.last_plan.ravel()
                    + mpc.last_b0).reshape(mpc.H, 6)
            print(f'  预测 z: {np.round(traj[:, 1], 4)}')
            print(f'  预测 vz: {np.round(traj[:, 4], 3)}')
            print(f'  各步 Fz和: {np.round(mpc.last_plan[:, 1::2].sum(1), 1)}')
            Jw = snap['J_w']                     # (4,2,2) 世界系雅可比
            tau_chk = np.einsum('fok,fo->fk', Jw, u)
            print(f'  |JTu| 髋/膝: {np.round(np.abs(tau_chk), 1)} (tau_max=30)')
        if k % 5 == 0 or 8 <= k <= 18:
            s2 = env.snapshot()
            print(f'k={k:2d} z={s2["base_pos"][1]:.4f} vz={s2["base_vel"][1]:+.3f} '
                  f'u_z和={u[:,1].sum():+7.1f} F_env_z和={s2["F_env"][:,1].sum():+7.1f} '
                  f'gap={np.round(env.gap[0]*1000, 1)}mm '
                  f'q0={np.round(env.q[0,:2], 3)} [{mpc.last_msg}]')
        if k == 15:
            # 同进程内重建 k=15 QP 并重解，与回路求解器结果直接对比
            from cvxopt import matrix as cvm, solvers as cvsol
            H = mpc.H
            p = np.asarray(snap['base_pos'], float)
            v = np.asarray(snap['base_vel'], float)
            phi = float(snap['base_phi'])
            omega = float(snap['base_omega'])
            foot = np.asarray(snap['foot_pos'], float)
            s0 = np.concatenate([p, [phi], v, [omega]])
            M_u, b0 = mpc._build_prediction(foot, p, sched, (0.0, -G), s0)
            S_ref = mpc._ref_traj(p, 0.0, mpc.z_target)
            Qbar = np.kron(np.eye(H), mpc.Q)
            Rbar = mpc.R * np.eye(8 * H)
            D = np.eye(8 * H) - np.eye(8 * H, k=8)
            E_first = np.zeros((8 * H, 8 * H)); E_first[:8, :8] = 1.0
            P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * mpc.Rd * (D.T @ D)
            q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
            P += 2.0 * mpc.Rd * E_first
            q[:8] += -2.0 * mpc.Rd * u_prev_before[:8]
            SC = 1000.0
            Ps, qs = SC * SC * P, SC * q
            G_rows, h_rows, A_rows, b_rows = mpc._constraints(
                sched, SC, snap['J_w'])

            def fun(xs):
                return 0.5 * xs @ Ps @ xs + qs @ xs

            x_in = mpc.last_plan.ravel() / SC
            print(f'  回路解 obj={fun(x_in):.3f} '
                  f'Fz0={x_in[1:8:2].sum()*SC:+.1f}')
            x0s = np.zeros(8 * H)
            x0s[:-8] = u_prev_before[8:]
            x0s[-8:] = u_prev_before[-8:]
            res = cvsol.coneqp(
                cvm(Ps, tc='d'), cvm(qs, tc='d'),
                cvm(np.stack(G_rows), tc='d'),
                cvm(np.array(h_rows), tc='d'),
                {'l': len(G_rows), 'q': [], 's': []},
                None, None, initvals={'x': cvm(x0s / SC, tc='d')})
            xs = np.array(res['x']).ravel()
            print(f'  同初值重解 status={res["status"]} it={res["iterations"]} '
                  f'obj={fun(xs):.3f} Fz0={xs[1:8:2].sum()*SC:+.1f}')
    snap = env.snapshot()
    print(f'稳态: z={snap["base_pos"][1]:.4f} vz={snap["base_vel"][1]:+.3f} '
          f'热解 u_z和={u[:,1].sum():+.2f} (闭环命令)')
    u_prev_save = mpc.u_prev.copy()
    mpc.u_prev = None
    u_c = mpc.solve(snap, sched, 0.0, (0.0, -G))   # 真冷启动（零初值）
    print(f'      u_prev=None 冷解: {u_c[:,1].sum():+.2f} '
          f'obj={mpc.last_obj[0]:.3f} it={mpc.last_nit}')
    # 从冷解再热启动一次（Δu 项应几乎不移动真最优），开 trace
    mpc.u_prev = u_prev_save
    mpc.polish_trace = True
    u_h = mpc.solve(snap, sched, 0.0, (0.0, -G))
    mpc.polish_trace = False
    print(f'      热解(重解): {u_h[:,1].sum():+.2f} obj={mpc.last_obj[0]:.3f} '
          f'it={mpc.last_nit}')
    # 稳态 QP 从零初值重解：真最优 vs 回路热解
    from cvxopt import matrix as cvm, solvers as cvsol
    H = mpc.H
    p = np.asarray(snap['base_pos'], float)
    v = np.asarray(snap['base_vel'], float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], float)
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, (0.0, -G), s0)
    S_ref = mpc._ref_traj(p, 0.0, mpc.z_target)
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    E_first = np.zeros((8 * H, 8 * H)); E_first[:8, :8] = 1.0
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * mpc.Rd * (D.T @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    P += 2.0 * mpc.Rd * E_first
    q[:8] += -2.0 * mpc.Rd * u_prev_save[:8]
    SC = 1000.0
    Ps, qs = SC * SC * P, SC * q
    G_rows, h_rows, A_rows, b_rows = mpc._constraints(sched, SC,
                                                      snap['J_w'])

    def fun(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    for name, x0 in (('零初值', np.zeros(8 * H)),
                     ('锚计划', u_prev_save / SC)):
        res = cvsol.coneqp(cvm(Ps, tc='d'), cvm(qs, tc='d'),
                           cvm(np.stack(G_rows), tc='d'),
                           cvm(np.array(h_rows), tc='d'),
                           {'l': len(G_rows), 'q': [], 's': []},
                           None, None,
                           initvals={'x': cvm(x0, tc='d')})
        xs = np.array(res['x']).ravel()
        print(f'      稳态重解({name}): status={res["status"]} '
              f'it={res["iterations"]} obj={fun(xs):.3f} '
              f'Fz0={xs[1:8:2].sum()*SC:+.1f}')
    # 一维线搜索：在重解最优上沿"步0 各腿 u_n 均匀 +δ"方向扰动
    xs = np.array(res['x']).ravel()
    base = xs.copy()
    print('      线搜索(步0 u_n 均匀+d, d in [-30..+60]):')
    for d in (-30, -10, 0, 10, 20, 30, 40, 60):
        xt = base.copy()
        xt[1:8:2] += d / SC
        print(f'        d={d:+3d}N: obj={fun(xt):.3f}')
    # 分量拆解：d=0 vs d=+10 时哪个代价项在动
    print('      分量拆解(d=0 / d=+10):')
    for d in (0, 10):
        xt = base.copy()
        xt[1:8:2] += d / SC
        traj = (M_u @ xt * SC + b0).reshape(H, 6)
        zc = mpc.Q[1, 1] * ((traj[:, 1] - mpc.z_nom) ** 2).sum()
        vc = mpc.Q[4, 4] * (traj[:, 4] ** 2).sum()
        pc = mpc.Q[2, 2] * (traj[:, 2] ** 2).sum()
        wc = mpc.Q[5, 5] * (traj[:, 5] ** 2).sum()
        rc = mpc.R * (xt * SC) @ (xt * SC)
        du = np.diff(xt.reshape(H, 8), axis=0).ravel() * SC
        du0 = (xt[:8] - u_prev_save[:8] / SC) * SC
        dc = mpc.Rd * (du0 @ du0 + du @ du)
        print(f'        d={d:+3d}: z={zc:8.2f} vz={vc:6.2f} phi={pc:6.2f} '
              f'w={wc:5.2f} R={rc:.3f} dU={dc:.3f} '
              f'z轨迹={np.round(traj[:, 1], 4)}')


if __name__ == '__main__':
    main()
