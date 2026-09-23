"""单控制步内亚步级探查：u_z 阶梯后，足端 gap/F_env_z/膝误差/PD/前馈/载荷
力矩在各子步的演化——定位 z 通道力实现链断裂点。

躯干每子步冻结（保存/恢复 p,v,φ,ω）：只测"腿+接触"链，排除俯仰中性
失稳（MPC Q_φ 的职责）与躯干坠落的耦合。静态理论预期：
  k=0 钉扎于 reset 的 -500μm → F = 0.983·u − 0.0171·(1e6·x_pin+697)
  ≈ 15.9（81%）；逐步重钉扎收敛 → 19.6（100%）；台阶 39.2 → ≈38.2。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from envs.robot import foot_jac

G = 9.81


def frozen_substep(env, *args):
    p, v, phi, om = env.p.copy(), env.v.copy(), env.phi.copy(), env.omega.copy()
    env._substep(*args)
    env.p, env.v, env.phi, env.omega = p, v, phi, om


def run_step(env, u_cmd, print_substeps=None, body_vx=0.0, body_omega=0.0):
    """手动跑一个控制步（100 子步）。

    body_vx/body_omega: 每个子步强制的躯干平动/俯仰速率（躯干位姿由
    探针规定，_substep 积分后被覆盖——模拟闭合回路的足端扫描工况）。
    """
    snap = env.snapshot()
    target_w = snap['foot_pos'].copy()
    c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
    R = np.array([[c, -s], [s, c]])
    J = foot_jac(snap['q'], env.l1, env.l2)
    J_w = np.einsum('ij,fjk->fik', R, J)
    tau_cmd = -np.einsum('fij,fi->fj', J_w, u_cmd).ravel()
    foot_start, _ = env._foot_state()
    target_v = (target_w[None] - foot_start) / env.control_dt
    p0, phi0 = env.p.copy(), env.phi.copy()
    for s_i in range(env.substeps):
        frac = (s_i + 1) / env.substeps
        tw = foot_start + (target_w[None] - foot_start) * frac
        q_sub, qd_sub = env._world_ik(tw, target_v)
        g_vec = np.stack([-G * np.sin(env.gravity_theta),
                          -G * np.cos(env.gravity_theta)], axis=-1)
        env.p = p0.copy()
        env.p[:, 0] += body_vx * (s_i + 1) * env.dt
        env.phi = phi0 + body_omega * (s_i + 1) * env.dt
        env.v = np.full_like(env.v, [body_vx, 0.0])
        env.omega = np.full_like(env.omega, body_omega)
        env._substep(tau_cmd[None], np.ones((1, 4)), q_sub, g_vec, qd_sub)
        if print_substeps is not None and s_i in print_substeps:
            gap_um = env.gap[0, 0] * 1e6
            Fz = env.F_env[0, 0, 1]
            e = (q_sub - env.q)[0, 0:2]
            kpe = env.kp[0] * e
            damp = env.kd[0] * (qd_sub[0, 0:2] - env.qd[0, 0:2])
            pd = kpe + damp
            Jw0 = np.einsum('ij,jk->ik', R, foot_jac(env.q, env.l1, env.l2)[0, 0])
            load = Jw0.T @ env.F_env[0, 0]
            clip_act = np.clip(tau_cmd[0:2] + pd, -30, 30)
            # 法向速度与弹簧力（f_spring = clip(−k_n·gap − d_n·v_n)）
            foot_w, foot_v = env._foot_state()
            vn = foot_v[0, 0, 1]
            f_spr = np.clip(-env.contact_model.k_n * env.gap[0, 0]
                            - env.contact_model.d_n * vn, 0.0, None)
            print(f'{s_i:4d}  {gap_um:+8.1f}  {Fz:+8.1f}  '
                  f'{e[0]*1e3:+8.2f}  {e[1]*1e3:+8.2f}  '
                  f'{pd[1]:+8.2f} {tau_cmd[1]:+8.2f} {clip_act[1]:+8.2f} '
                  f'{load[1]:+8.2f}  qd_des_h={qd_sub[0, 0]:+8.3f} '
                  f'qd_h={env.qd[0, 0]:+8.3f} qd_des_k={qd_sub[0, 1]:+8.3f} '
                  f'qd_k={env.qd[0, 1]:+8.3f} J00={Jw0[0, 0]:+8.3f} '
                  f'J11={Jw0[1, 1]:+8.3f} vn={vn:+8.3f} f_spr={f_spr:+8.1f}')
    return env.snapshot()


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    print('hold（躯干冻结, u_z=19.6/腿, 每步重钉扎）:')
    for k in range(12):
        u_cmd = np.zeros((4, 2))
        u_cmd[:, 1] = 19.6
        s2 = run_step(env, u_cmd)
        if k <= 8:
            print(f'  k={k:2d} gap={np.round(s2["foot_pos"][:, 1]*1e3, 3)}mm '
                  f'Fz={np.round(s2["F_env"][:, 1], 1)} '
                  f'q[0]={np.round(env.q[0, 0:2], 3)}')
    # 台阶：u_z 19.6 → 39.2，三种躯干运动工况对比
    u_cmd = np.zeros((4, 2))
    u_cmd[:, 1] = 39.2
    cases = [('冻结', 0.0, 0.0), ('vx=0.34 扫描', 0.34, 0.0),
             ('ω=0.5rad/s 俯仰', 0.0, 0.5)]
    for name, vx, om in cases:
        s2 = run_step(env, u_cmd, body_vx=vx, body_omega=om)
        print(f'台阶 19.6→39.2 [{name}]: gap={np.round(s2["foot_pos"][:, 1]*1e3, 3)}mm '
              f'Fz={np.round(s2["F_env"][:, 1], 1)} '
              f'q[0]={np.round(env.q[0, 0:2], 3)}')
    # 瞬态 vs 稳态：先跑 10 步扫描持形（关节进入扫描稳态），再台阶
    print('先 10 步扫描持形（u_z=19.6, vx=0.34）:')
    for k in range(10):
        u_cmd = np.zeros((4, 2))
        u_cmd[:, 1] = 19.6
        s2 = run_step(env, u_cmd, body_vx=0.34)
        if k in (0, 1, 3, 5, 9):
            print(f'  sweep k={k:2d} Fz={np.round(s2["F_env"][:, 1], 1)} '
                  f'q[0]={np.round(env.q[0, 0:2], 3)}')
    u_cmd[:, 1] = 39.2
    s2 = run_step(env, u_cmd, body_vx=0.34)
    print(f'台阶 19.6→39.2 [扫描稳态后]: gap={np.round(s2["foot_pos"][:, 1]*1e3, 3)}mm '
          f'Fz={np.round(s2["F_env"][:, 1], 1)}')
    # 扫描稳态下的亚步追踪：看膝的 qd_des vs qd
    print('扫描稳态亚步追踪（腿0, u_z=39.2, vx=0.34）:')
    run_step(env, u_cmd, body_vx=0.34, print_substeps=(0, 10, 40, 99))
    print('期望（冻结）: gap −717μm → −736μm, F_env_z → ~38.2（97.5%）')


if __name__ == '__main__':
    main()
