"""竖直墙全腿支撑静止保持（无步态/无摆动腿）：隔离步态层的原始 MPC。

若此脚本能稳住，问题在 gait/swing/controller 层；若同样倾倒，
则 env 墙平衡或 MPC 墙模型本身有误。
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
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    for k in range(120):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (-G, 0.0))
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        if k % 5 == 0 or k < 6:
            s2 = env.snapshot()
            print(f'k={k:3d} p=({s2["base_pos"][0]:+.4f},{s2["base_pos"][1]:+.4f}) '
                  f'v=({s2["base_vel"][0]:+.3f},{s2["base_vel"][1]:+.3f}) '
                  f'phi={s2["base_phi"]:+.3f} '
                  f'u_z和={u[:,1].sum():+7.1f} u_x和={u[:,0].sum():+7.1f} '
                  f'Fz={s2["F_env"][:,1].sum():+7.1f} Fx={s2["F_env"][:,0].sum():+7.1f} '
                  f'gap={np.round(env.gap[0]*1000,1)}mm '
                  f'q0={np.round(env.q[0,:2],3)} [{mpc.last_msg}]')
        if env.snapshot()['terminated']:
            print(f'终止于 k={k} ({k*0.02:.2f}s)')
            break

    # ---- 稳态 QP：切向推动线搜索 + 重解 ----
    from cvxopt import matrix as cvm, solvers as cvsol
    H = mpc.H
    snap = env.snapshot()
    p = np.asarray(snap['base_pos'], float)
    v = np.asarray(snap['base_vel'], float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], float)
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, (-G, 0.0), s0)
    S_ref = mpc._ref_traj(p, 0.0, mpc.z_target)
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    E_first = np.zeros((8 * H, 8 * H)); E_first[:8, :8] = 1.0
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * mpc.Rd * (D.T @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    P += 2.0 * mpc.Rd * E_first
    q[:8] += -2.0 * mpc.Rd * mpc.u_prev[:8]
    SC = 1000.0
    Ps, qs = SC * SC * P, SC * q
    G_rows, h_rows, A_rows, b_rows = mpc._constraints(sched, SC,
                                                      snap['J_w'])

    def fun(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    x_ret = mpc.last_plan.ravel() / SC
    print(f'  回路解 obj={fun(x_ret):.3f} u_x和0={x_ret[:8:2].sum()*SC:+.1f}')
    for name, x0 in (('零初值', np.zeros(8 * H)),
                     ('锚计划', mpc.u_prev / SC)):
        res = cvsol.coneqp(cvm(Ps, tc='d'), cvm(qs, tc='d'),
                           cvm(np.stack(G_rows), tc='d'),
                           cvm(np.array(h_rows), tc='d'),
                           {'l': len(G_rows), 'q': [], 's': []},
                           None, None, initvals={'x': cvm(x0, tc='d')})
        xs = np.array(res['x']).ravel()
        print(f'  重解({name}): status={res["status"]} it={res["iterations"]} '
              f'obj={fun(xs):.3f} u_x和0={xs[:8:2].sum()*SC:+.1f}')
    base = np.array(res['x']).ravel()
    print('  线搜索(步0 u_x 均匀+d):')
    for d in (-10, -3, 0, 3, 10, 20):
        xt = base.copy()
        xt[:8:2] += d / SC
        print(f'    d={d:+3d}N: obj={fun(xt):.3f}')


if __name__ == '__main__':
    main()
