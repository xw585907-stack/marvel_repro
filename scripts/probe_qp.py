"""探查 k=1 状态（z=0.3121 悬空）QP 的目标函数：分解各状态块代价。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from control.mpc import SRB_MPC
from control.ik import two_link_ik

G = 9.81


def rebuild_qp(mpc, snap, sched, g_vec):
    """复制 solve() 的代价装配（不求解），返回可评估对象。"""
    p = np.asarray(snap['base_pos'], dtype=float)
    v = np.asarray(snap['base_vel'], dtype=float)
    phi = float(snap['base_phi'])
    omega = float(snap['base_omega'])
    foot = np.asarray(snap['foot_pos'], dtype=float)
    s0 = np.concatenate([p, [phi], v, [omega]])
    M_u, b0 = mpc._build_prediction(foot, p, sched, g_vec, s0)

    H = mpc.H
    S_ref = np.zeros((6 * H,))
    for k in range(1, H + 1):
        S_ref[(k - 1) * 6:(k - 1) * 6 + 6] = [p[0], mpc.z_nom, 0.0, 0.0, 0.0, 0.0]
    Qbar = np.kron(np.eye(H), mpc.Q)
    Rbar = mpc.R * np.eye(8 * H)
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    Rdbar = mpc.Rd * np.eye(8 * H)
    P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar + D.T @ Rdbar @ D)
    q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
    if mpc.u_prev is not None:
        v0 = np.zeros(8 * H)
        v0[:8] = mpc.u_prev[:8]
        q += -2.0 * (D.T @ Rdbar @ D) @ v0
    return M_u, b0, S_ref, P, q


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    env.p[0, 1] = 0.32
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)

    # k=0 求解（制造 u_prev = 全拉）
    u0 = mpc.solve(env.snapshot(), sched, 0.0, (0.0, -G))
    k_n = env.contact_model.k_n
    k_t = env.contact_model.k_t
    f_mag = float(env.f_max[0])
    anchor = env.contact_model.anchor[0]
    target_w = np.stack([anchor[:, 0] - u0[:, 0] / k_t,
                         -(u0[:, 1] + f_mag) / k_n], axis=-1)
    d = target_w - env.p[0]
    rb = np.stack([d[:, 0], d[:, 1]], axis=-1) - env.hip_offsets
    env.step(np.zeros((1, 8)), np.ones((1, 4)), two_link_ik(rb).ravel()[None, :])

    snap = env.snapshot()
    print(f'k=1 状态: p={snap["base_pos"]}, v={snap["base_vel"]}, '
          f'foot={snap["foot_pos"][:, 0]}')
    M_u, b0, S_ref, P, q = rebuild_qp(mpc, snap, sched, (0.0, -G))
    H = mpc.H
    SC = 1000.0
    Ps = SC * SC * P
    qs = SC * q

    def traj_cost(x):      # 只算状态跟踪部分
        S = M_u @ x + b0
        e = S - S_ref
        return np.sum(e.reshape(H, 6) ** 2 * np.diag(mpc.Q)[None, :], axis=1)

    def fun(x):
        return 0.5 * x @ Ps @ x + qs @ x

    # 候选 1：全拉 −697（约束下界，未缩放）
    x_pull = np.zeros(8 * H)
    x_pull[1::8] = -697.0
    # 候选 2：投影热启动（u_n=−0.97f_mag, 缩放后 ≈ −0.676）
    x_warm_s = np.tile(mpc.u_prev[8:] / SC, 1)
    print(f'fun(x=0)         = {fun(np.zeros(8 * H)):.3e}')
    print(f'fun(全拉−697)    = {fun(x_pull):.3e}')
    print(f'fun(热启动)      = {fun(x_warm_s):.3e}')
    for name, x in [('x=0', np.zeros(8 * H)), ('全拉−697', x_pull),
                    ('热启动', x_warm_s)]:
        tc = traj_cost(x)
        block = np.sum((M_u @ x + b0 - S_ref).reshape(H, 6) ** 2
                       * np.diag(mpc.Q)[None, :], axis=0)
        print(f'  {name}: 轨迹代价按步={np.round(tc, 2)} 总={tc.sum():.3e} '
              f'分块 x/z/phi/vx/vz/w={np.round(block, 2)}')
        rc = mpc.R * np.sum(x ** 2)
        print(f'    力代价={rc:.2e}')


if __name__ == '__main__':
    main()
