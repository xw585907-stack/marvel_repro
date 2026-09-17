"""调试 _polish_active_set：静止保持稳态下，polish 为何没把冷/热解钉到同一点。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.mpc import SRB_MPC
from scripts.debug_mpc import realize_step

G = 9.81


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    for k in range(80):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
    snap = env.snapshot()

    # 手动重建同一 QP
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
    for kk in range(1, H + 1):
        S_ref[(kk - 1) * 6:(kk - 1) * 6 + 6] = [
            p[0], mpc.z_nom, 0.0, 0.0, 0.0, 0.0]
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    Rdbar = mpc.Rd * np.eye(8 * H)
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar + D.T @ Rdbar @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    SC = 1000.0
    Ps = SC * SC * P
    qs = SC * q

    def fun(xx):
        return 0.5 * xx @ Ps @ xx + qs @ xx

    # 冷解（零初值）与热解（u_prev）各跑一遍 coneqp，然后手动复刻 polish 循环
    mpc.u_prev = None
    u_cold = mpc.solve(snap, sched, 0.0, (0.0, -G))
    x_cold = mpc.u_prev / SC
    print(f'冷解 u_z和={u_cold[:,1].sum():+.2f} obj={fun(x_cold):.4f} '
          f'it={mpc.last_nit}')

    # 热解
    u_hot = mpc.solve(snap, sched, 0.0, (0.0, -G))
    x_hot = mpc.u_prev / SC
    print(f'热解 u_z和={u_hot[:,1].sum():+.2f} obj={fun(x_hot):.4f} '
          f'it={mpc.last_nit}')

    # 检查 Ps 是否奇异：最小特征值
    ev = np.linalg.eigvalsh(Ps)
    print(f'Ps 特征值范围 [{ev[0]:.2e}, {ev[-1]:.2e}] 条件数 {ev[-1]/ev[0]:.2e}')

    # 手动 polish（带打印）
    G_rows, h_rows, A_rows, b_rows = mpc._constraints(sched, SC)
    Gc = np.stack(G_rows)
    h = np.array(h_rows)
    A = np.stack(A_rows) if A_rows else np.zeros((0, len(qs)))
    nv = len(qs)
    for name, x0 in (('冷', x_cold), ('热', x_hot)):
        x = x0.copy()
        f_cur = fun(x)
        print(f'--- polish 从{name}解出发: f0={f_cur:.4f}')
        for it in range(5):
            act = np.where(Gc @ x - h >= -1e-9)[0]
            W = np.vstack([Gc[act], A]) if len(act) else A
            ncon = len(W)
            g = Ps @ x + qs
            if ncon == 0:
                dx = np.linalg.solve(Ps, -g)
            else:
                K = np.zeros((nv + ncon, nv + ncon))
                K[:nv, :nv] = Ps
                K[:nv, nv:] = W.T
                K[nv:, :nv] = W
                rhs = np.zeros(nv + ncon)
                rhs[:nv] = -g
                dx = np.linalg.solve(K, rhs)[:nv]
            alpha = 1.0
            Gd = Gc @ dx
            tight = Gd > 1e-14
            if np.any(tight):
                alpha = min(alpha, np.min((h - Gc @ x)[tight] / Gd[tight]))
            alpha = min(max(alpha, 0.0), 1.0)
            x_new = x + alpha * dx
            f_new = fun(x_new)
            print(f'  it{it}: ncon={ncon} |g|={np.linalg.norm(g):.3e} '
                  f'|dx|={np.linalg.norm(dx):.3e} alpha={alpha:.4f} '
                  f'f_cur={f_cur:.4f} f_new={f_new:.4f}')
            if f_new >= f_cur - 1e-12:
                print(f'  -> 目标不降，break')
                break
            x, f_cur = x_new, f_new
        print(f'  {name}解 polish 后 u_z和={x[:8].sum()*SC:.2f} '
              f'f={fun(x):.4f}')
        # 对照：直接调类内 _polish_active_set，输入完全相同
        x_pol = mpc._polish_active_set(Ps, qs, sched, x0, SC)
        print(f'  类内方法同输入: u_z和={x_pol[:8].sum()*SC:.2f} '
              f'f={fun(x_pol):.4f} 与手动差={np.linalg.norm(x_pol-x):.2e}')


if __name__ == '__main__':
    main()
