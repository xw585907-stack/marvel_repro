"""探查：闭环某一步的 QP 返回解是否真最优（vs 高精度参考解），
以及模型对该步命令的预测（vx/φ/ω 一步预测）。"""
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
    for k in range(3):                      # 跑到 k=2 的状态
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        s = env.snapshot()
        print(f'k={k}: vx={s["base_vel"][0]:+.3f} vz={s["base_vel"][1]:+.3f} '
              f'phi={s["base_phi"]:+.4f} omega={s["base_omega"]:+.4f} '
              f'z={s["base_pos"][1]:.4f} '
              f'u={np.round(ctrl.info["u"].ravel(), 1)} '
              f'obj(res,x0)=({ctrl.mpc.last_obj[0]:.1f},{ctrl.mpc.last_obj[1]:.1f}) '
              f'it={ctrl.mpc.last_nit}')

    # 第 4 步（k=3）的 QP：常规解 vs 高精度参考解
    snap = env.snapshot()
    mpc = ctrl.mpc
    t = ctrl.gait.t
    sched = ctrl.gait.contact_schedule(t, mpc.H, mpc.dt)
    g_vec = (-G * np.sin(snap['theta']), -G * np.cos(snap['theta']))
    u_fast = mpc.solve(snap, sched, 0.3, g_vec)
    print(f'\nk=3 常规解 (it={mpc.last_nit}): u={np.round(u_fast.ravel(), 2)}')

    # 参考解：独立 QP，多迭代 + 高罚
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

    # 独立参考解：scipy SLSQP（锥为显式约束）+ 2000 次迭代
    from scipy.optimize import minimize

    lb = np.zeros(8 * H); ub = np.zeros(8 * H)
    Gc_rows, hc_rows = [], []
    for kk in range(H):
        for i in range(4):
            idx = (kk * 4 + i) * 2
            if sched[kk, i]:
                lb[idx] = -np.inf; ub[idx] = np.inf
                lb[idx + 1] = -ref.f_mag; ub[idx + 1] = ref.f_n_max
                for sgn in (-1.0, 1.0):
                    row = np.zeros(8 * H)
                    row[idx] = sgn; row[idx + 1] = ref.mu
                    Gc_rows.append(row)
                    hc_rows.append(0.98 * ref.mu * ref.f_mag)
            else:
                lb[idx:idx + 2] = 0.0; ub[idx:idx + 2] = 0.0
    Gcs = np.stack(Gc_rows) * SC
    hcs = np.array(hc_rows)

    def fobj(xs):
        return 0.5 * xs @ Ps @ xs + qs @ xs

    def jobj(xs):
        return Ps @ xs + qs

    # SLSQP 会就地改写 x0——必须传副本
    res = minimize(fobj, x0s.copy(), jac=jobj, method='SLSQP',
                   constraints={'type': 'ineq', 'fun': lambda xs: Gcs @ xs + hcs,
                                'jac': lambda xs: Gcs},
                   bounds=list(zip(lb / SC, ub / SC)),
                   options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
    x_ref = res.x
    print(f'SLSQP 参考解 (nit={res.nit}, {res.message}): '
          f'u={np.round(x_ref[:8] * SC, 2)}')
    print(f'常规解与参考解差 = {np.abs((u_fast.ravel() / SC) - x_ref[:8]).max():.2e} (缩放)')
    x_reg, _ = ref._solve_qp(Ps, qs, sched, x0s.copy(), SC, max_iter=200)
    print(f'全变量常规解 (it=200 上限): u0={np.round(x_reg[:8] * SC, 2)}')
    print(f'f(常规解) = {fobj(x_reg):.8f}, f(参考解) = {fobj(x_ref):.8f}  '
          f'Δf={fobj(x_reg) - fobj(x_ref):.2e}  ← 若 Δf≈0 则两者同为最优，'
          f'差在代价零空间（QP 退化）')

    # 模型一步预测（用常规解）
    A, B, _ = ref._dynamics(snap['foot_pos'], p, snap['base_phi'])
    g_ext = np.concatenate([np.zeros(3), ref.dt * np.array([g_vec[0], g_vec[1], 0.0])])
    s1 = A @ s0 + B @ u_fast.ravel() + g_ext
    print(f'\n模型一步预测（该步命令 u 后）: vx1={s1[3]:+.3f} (当前 {v[0]:+.3f}), '
          f'vz1={s1[4]:+.3f}, omega1={s1[5]:+.3f} (当前 {snap["base_omega"]:+.3f}), '
          f'phi1={s1[2]:+.3f} (当前 {snap["base_phi"]:+.3f})')
    print(f'u_x 和={u_fast[:, 0].sum():+.1f}, u_z 和={u_fast[:, 1].sum():+.1f}')

    # 预测时域 vx 序列（常规解整条 u 轨迹）
    U = np.concatenate([u_fast.ravel(), np.zeros(8 * H - 8)])
    traj = b0 + M_u @ U
    print('模型预测 vx 序列:', np.round(traj[3::6], 2))
    print('模型预测 phi 序列:', np.round(traj[2::6], 3))


if __name__ == '__main__':
    main()
