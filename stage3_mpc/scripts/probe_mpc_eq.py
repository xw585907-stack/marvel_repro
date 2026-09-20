"""探查收敛状态 MPC 的静态响应：z 扫描 / 俯仰清零 / 冷启动 / 求解质量。

目标：定位 +25mm 体高偏置。闭环稳态 z=0.1748 而 MPC 命令 u_z和=87.5
（>mg=78.5）——若 MPC 的 z 反馈符号/幅值正确，此状态不可能出现；
逐项隔离：真状态解、z 扫描曲线、俯仰贡献、热启动影响、目标值下降。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController


def solve_at(ctrl, snap, zz=None, phi0=None, u_prev=None, cold=False):
    """按修改后的状态求解一次，返回 u_z和；u_prev=None 时不恢复。"""
    if cold:
        ctrl.mpc.u_prev = None
    elif u_prev is not None:
        ctrl.mpc.u_prev = u_prev.copy()
    s2 = {k: v for k, v in snap.items()}
    s2['base_pos'] = snap['base_pos'].copy()
    if zz is not None:
        s2['base_pos'][1] = zz
    if phi0 is not None:
        s2['base_phi'] = phi0
        s2['base_omega'] = 0.0
    sched = ctrl.gait.contact_schedule(ctrl.gait.t, ctrl.mpc.H, ctrl.mpc.dt)
    g_vec = (0.0, -9.81)
    u = ctrl.mpc.solve(s2, sched, 0.3, g_vec)
    return u, ctrl.mpc


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    for k in range(160):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
    snap = env.snapshot()
    mpc = ctrl.mpc
    print(f'收敛状态: z={snap["base_pos"][1]:.4f} vz={snap["base_vel"][1]:+.3f} '
          f'phi={snap["base_phi"]*1e3:+.1f}mrad omega={snap["base_omega"]*1e3:+.2f}mrad/s '
          f't={ctrl.gait.t:.3f} vx={snap["base_vel"][0]:+.3f}')
    sched = ctrl.gait.contact_schedule(ctrl.gait.t, mpc.H, mpc.dt)
    print(f'horizon 每步支撑腿数: {sched.sum(axis=1)}')

    # 1) 真状态 + 冷启动（排除热启动钉住次优解）
    u, mpc = solve_at(ctrl, snap, cold=True)
    print(f'真状态[冷启动] u_z和={u[:,1].sum():+.2f} u_x和={u[:,0].sum():+.2f} '
          f'解={mpc.last_obj[0]:.4f} 初值={mpc.last_obj[1]:.4f} it={mpc.last_nit}')
    u, mpc = solve_at(ctrl, snap, u_prev=np.zeros_like(ctrl.mpc.u_prev))
    print(f'真状态[零初值] u_z和={u[:,1].sum():+.2f} 解={mpc.last_obj[0]:.4f} '
          f'初值={mpc.last_obj[1]:.4f} it={mpc.last_nit}')
    print(f'真状态 u/腿 = {np.round(u, 1)}')

    # 2) z 扫描：u_z和 随 z 的响应曲线（符号与斜率）
    print('z 扫描（其余状态不变, 冷启动）:')
    for zz in (0.13, 0.14, 0.15, 0.16, 0.17, 0.175, 0.18):
        u, mpc = solve_at(ctrl, snap, zz=zz, cold=True)
        print(f'  z={zz:.3f} -> u_z和={u[:,1].sum():+7.2f} '
              f'u_x和={u[:,0].sum():+7.2f} it={mpc.last_nit}')

    # 3) 俯仰清零：俯仰代价/力臂不对称对 u_z和 的贡献
    u, mpc = solve_at(ctrl, snap, phi0=0.0, cold=True)
    print(f'俯仰清零 -> u_z和={u[:,1].sum():+.2f} '
          f'(俯仰项贡献 {u[:,1].sum() - 87.5:+.1f}N 若为真状态值)')

    # 4) 同一状态下的热启动解（不碰 u_prev，与闭环完全一致）
    sched = ctrl.gait.contact_schedule(ctrl.gait.t, mpc.H, mpc.dt)
    g_vec = (0.0, -9.81)
    u_hot = ctrl.mpc.solve(snap, sched, 0.3, g_vec)
    print(f'真状态[热启动] u_z和={u_hot[:,1].sum():+.2f} '
          f'解={mpc.last_obj[0]:.4f} 初值={mpc.last_obj[1]:.4f} it={mpc.last_nit}')
    # 5) 完整闭环一步（步态推进 + 热启动，与真实控制循环完全一致）
    tau, mag, q_des = ctrl.step()
    u_cl = ctrl.info['u']
    print(f'闭环下一步命令 u_z和={u_cl[:,1].sum():+.2f} '
          f'u/腿={np.round(u_cl, 1)} '
          f'解={ctrl.mpc.last_obj[0]:.4f} 初值={ctrl.mpc.last_obj[1]:.4f} '
          f'it={ctrl.mpc.last_nit}')
    # 6) 足端力臂打印（判断俯仰-力耦合的不对称性）
    foot = np.asarray(snap['foot_pos'])
    p = np.asarray(snap['base_pos'])
    print(f'足端力臂 r = foot - COM =\n{np.round(foot - p, 3)}')

    # 7) 容差实验：收紧 coneqp 收敛判据，冷/热启动是否收敛到同一点
    from cvxopt import solvers as _solv
    for opts in ({}, {'feastol': 1e-9, 'abstol': 1e-9, 'reltol': 1e-9},
                 {'feastol': 1e-11, 'abstol': 1e-11, 'reltol': 1e-11}):
        _solv.options.update({k: 1e-7 if k == 'feastol' else 1e-6
                              for k in ('feastol', 'abstol', 'reltol')})
        _solv.options.update(opts)
        u_c = solve_at(ctrl, snap, cold=True)[0]
        u_h = solve_at(ctrl, snap, u_prev=ctrl.mpc.u_prev)[0]
        m = ctrl.mpc
        print(f'opts={opts or "默认"}: 冷 u_z和={u_c[:,1].sum():+.2f} '
              f'热 u_z和={u_h[:,1].sum():+.2f} 差={abs(u_c[:,1].sum()-u_h[:,1].sum()):.2f}N '
              f'it={m.last_nit} solve={m.solve_ms:.1f}ms')
    _solv.options.update({'feastol': 1e-7, 'abstol': 1e-7, 'reltol': 1e-6})

    # 8) QP 凝聚核算：fun 下界 = −(b0−S_ref)ᵀQ(b0−S_ref) 逐通道
    from stage3_mpc.control.mpc import SRB_MPC
    s0 = np.concatenate([np.asarray(snap['base_pos'], dtype=float),
                         [float(snap['base_phi'])],
                         np.asarray(snap['base_vel'], dtype=float),
                         [float(snap['base_omega'])]])
    M_u, b0 = mpc._build_prediction(np.asarray(snap['foot_pos'], dtype=float),
                                    np.asarray(snap['base_pos'], dtype=float),
                                    sched, (0.0, -9.81), s0)
    H = mpc.H
    S_ref = np.zeros((6 * H,))
    for k in range(1, H + 1):
        S_ref[(k - 1) * 6:(k - 1) * 6 + 6] = [
            p[0] + 0.3 * k * mpc.dt, mpc.z_nom, 0.0, 0.3, 0.0, 0.0]
    e = b0 - S_ref
    Qbar = np.kron(np.eye(H), mpc.Q)
    bound = float(e @ Qbar @ e)
    print(f'QP 下界 (b0−S_ref)ᵀQ(b0−S_ref) = {bound:.3f}')
    for name, idx, w in (('x', 0, mpc.Q[0, 0]), ('z', 1, mpc.Q[1, 1]),
                         ('phi', 2, mpc.Q[2, 2]), ('vx', 3, mpc.Q[3, 3]),
                         ('vz', 4, mpc.Q[4, 4]), ('om', 5, mpc.Q[5, 5])):
        ee = e[idx::6]
        print(f'  {name:>4}: Σe²={np.sum(ee**2):10.5f}  ×Q={w:5.0f} = '
              f'{w * np.sum(ee**2):9.3f}   e@k=0/9 = {ee[0]:+.4f}/{ee[9]:+.4f}')
    # 用闭环命令 u_cl 手工核算 fun = 0.5uᵀPu + qᵀu
    U = np.tile(u_cl.ravel(), H)
    Pfull = 2.0 * (M_u.T @ Qbar @ M_u + mpc.R * np.eye(8 * H)
                   + np.eye(8 * H) @ np.eye(8 * H) * 0)  # 略 Rd 占位
    D = np.eye(8 * H) - np.eye(8 * H, k=8)
    qfull = 2.0 * M_u.T @ Qbar @ e
    fun_manual = 0.5 * U @ Pfull @ U + qfull @ U
    full_cost = float((M_u @ U + b0 - S_ref) @ Qbar @ (M_u @ U + b0 - S_ref))
    print(f'手工 fun(闭环 u) = {fun_manual:.3f}  （solve 报告 {ctrl.mpc.last_obj[0]:.3f}）')
    print(f'手工全代价 (Mu+b0−S)ᵀQ(Mu+b0−S) = {full_cost:.3f}')


if __name__ == '__main__':
    main()
