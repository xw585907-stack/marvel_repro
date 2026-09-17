"""A/B：旧 Δu 边界实现 vs 新实现，在 reset 状态（u_prev=None）下
冷解最优力对比。两者 k=0 的 q 相同，只有 P 差 2Rd·E_first。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.mpc import SRB_MPC

G = 9.81


def build_qp(mpc, snap, sched, old_style):
    p = np.asarray(snap['base_pos'], float)
    v = np.asarray(snap['base_vel'], float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], float)
    A_, B_, _ = mpc._dynamics(foot, p, phi)
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, (0.0, -G), s0)
    H = mpc.H
    S_ref = np.zeros((6 * H,))
    for k in range(1, H + 1):
        S_ref[(k - 1) * 6:(k - 1) * 6 + 6] = [
            p[0], mpc.z_nom, 0.0, 0.0, 0.0, 0.0]
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    E_first = np.zeros((8 * H, 8 * H))
    E_first[:8, :8] = 1.0
    if old_style:
        P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar + mpc.Rd * (D.T @ D))
    else:
        P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) \
            + 2.0 * mpc.Rd * (D.T @ D + E_first)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    SC = 1000.0
    Ps = SC * SC * P
    qs = SC * q
    return Ps, qs, SC


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    snap = env.snapshot()
    print(f'reset 状态: z={snap["base_pos"][1]:.4f} '
          f'vz={snap["base_vel"][1]:+.4f} phi={snap["base_phi"]:.4f} '
          f'脚端z={np.round(snap["foot_pos"][:,1], 4)}')
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    xs = {}
    for old_style in (True, False):
        Ps, qs, SC = build_qp(mpc, snap, sched, old_style)

        def fun(xx):
            return 0.5 * xx @ Ps @ xx + qs @ xx

        # 用 polish 求精确最优（无约束活动时为纯牛顿一步）
        x = mpc._polish_active_set(Ps, qs, sched, np.zeros_like(qs), SC)
        u = x[:8] * SC
        print(f'{"旧" if old_style else "新"}边界 P: 冷解 u_z和={u[1::2].sum():+.2f} '
              f'u_x和={u[0::2].sum():+.2f} obj={fun(x):.4f}')
        print(f'  u0 = {np.round(u.reshape(4,2), 1)}')
        # 时域各步的合力
        U = (x.reshape(mpc.H, 8) * SC)
        print(f'  各步 u_z和 = {np.round(U[:,1::2].sum(1), 1)}')
        xs[old_style] = x
    dx = xs[False] - xs[True]
    # Δx 对应的力矩轨迹（新-旧）
    p = np.asarray(snap['base_pos'], float)
    M_u, _ = mpc._build_prediction(np.asarray(snap['foot_pos'], float), p,
                                   sched, (0.0, -G),
                                   np.concatenate([p, [snap['base_phi']],
                                                   snap['base_vel'],
                                                   [snap['base_omega']]]))
    w = (M_u @ dx) * SC
    print(f'Δu(新-旧) z和={dx[1::2].sum()*SC:+.2f} '
          f'|Δu|={np.linalg.norm(dx)*SC:.1f}')
    print(f'Δ力矩轨迹 Fz={np.round(w[1::6].reshape(mpc.H), 1)}')
    print(f'Δ力矩轨迹 Ty={np.round(w[5::6].reshape(mpc.H), 1)}')


if __name__ == '__main__':
    main()
