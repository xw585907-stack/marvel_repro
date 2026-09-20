"""MPC QP 诊断：单独求解 + 预测一致性 + 力实现验证。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.mpc import SRB_MPC
from envs.ik import two_link_ik
from envs.robot import foot_jac

G = 9.81


def qp_sanity():
    print('=== 1. QP 解质量：平地全支撑保持 ===')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    snap = env.snapshot()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    t0 = time.perf_counter()
    u = mpc.solve(snap, sched, 0.0, (0.0, -G))
    print(f'  solve: {mpc.solve_ms:.2f} ms')
    print(f'  u = {np.round(u, 1)}')
    print(f'  sum u_z = {u[:, 1].sum():.1f} (期望 ~78.5 = mg)')
    print(f'  sum u_x = {u[:, 0].sum():.1f} (期望 ~0)')
    # 手动验证一步预测：s1 = A s0 + B u0 + g_ext
    p = snap['base_pos']; v = snap['base_vel']
    s0 = np.concatenate([p, [snap['base_phi']], v, [snap['base_omega']]])
    A, B, C = mpc._dynamics(snap['foot_pos'], p, snap['base_phi'])
    g_ext = np.concatenate([np.zeros(3), mpc.dt * np.array([0.0, -G, 0.0])])
    s1_manual = A @ s0 + B @ u.ravel() + g_ext
    print(f'  s1_manual = {np.round(s1_manual, 4)}')
    print(f'  (z1 应 ~= 0.1493, vz1 应 ~= (sum u_z - 78.5)/8 * 0.02 ~= 0)')


def qp_warm_speed():
    print('=== 2. 连续求解速度（热启动路径 vs 冷启动）===')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    times = []
    for k in range(20):
        snap = env.snapshot()
        t0 = time.perf_counter()
        mpc.solve(snap, sched, 0.0, (0.0, -G))
        times.append((time.perf_counter() - t0) * 1e3)
        env.step(np.zeros((1, 8)), np.ones((1, 4)), env.q_nominal.copy())
    print(f'  平均 {np.mean(times):.2f} ms, 最大 {np.max(times):.2f} ms, '
          f'各步: {np.round(times, 1)}')


def realize_step(env, u, snap):
    """控制器同款力实现：力矩前馈 τ_ff = -J_wᵀ·u + 世界 IK 原地钉扎。

    返回 (tau, q_des, target_w) 供 env.step。
    """
    c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
    R = np.array([[c, -s], [s, c]])
    J = foot_jac(snap['q'], env.l1, env.l2)
    J_w = np.einsum('ij,fjk->fik', R, J)
    tau = -np.einsum('fij,fj->fi', J_w, u).ravel()
    target_w = snap['foot_pos'][None, :, :]              # 钉在当前足端
    d = target_w[0] - snap['base_pos']
    rb = np.stack([c * d[:, 0] + s * d[:, 1],
                   -s * d[:, 0] + c * d[:, 1]], axis=-1) - env.hip_offsets
    q_des = two_link_ik(rb).ravel()
    return tau, q_des, target_w


def force_realization():
    print('=== 3. 力→位姿映射验证：u=(0, 19.6) 的静平衡 ===')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    u = np.zeros((4, 2))
    u[:, 1] = 8 * G / 4                    # 每脚 19.6 N
    for k in range(100):                   # 1s
        snap = env.snapshot()
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        if k in (0, 1, 2, 10, 99):
            s2 = env.snapshot()
            print(f'  k={k}: F_env_z/腿 = {np.round(s2["F_env"][:, 1], 1)}, '
                  f'z = {s2["base_pos"][1]:.4f}')
    snap = env.snapshot()
    print(f'  z = {snap["base_pos"][1]:.4f} (期望 0.1493), '
          f'v_z = {snap["base_vel"][1]:.5f} (期望 ~=0)')
    print(f'  F_env 净力 = {np.round(snap["F_env"].sum(0), 1)} (期望 ~= (0, 78.5))')


def z_response():
    print('=== 4. z 扰动响应：MPC 应把躯干从 +0.5m/s 扰动拉回 0.15 ===')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    env.v[0, 1] = 0.5                     # 向上速度扰动（足端保持接触）
    mpc = SRB_MPC()
    sched = np.ones((mpc.H, 4), dtype=bool)
    z_hist = []
    for k in range(100):
        snap = env.snapshot()
        u = mpc.solve(snap, sched, 0.0, (0.0, -G))
        tau, q_des, target_w = realize_step(env, u, snap)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :], target_w)
        z_hist.append(env.p[0, 1])
        if k % 20 == 0 or k < 3:
            print(f'  k={k}: z={env.p[0, 1]:.4f}, u_z 和={u[:, 1].sum():.1f}, '
                  f'v_z={env.v[0, 1]:.3f}, solve={mpc.solve_ms:.1f}ms '
                  f'({mpc.last_msg[:20]}, it={mpc.last_nit}) '
                  f'F_env_z 和={env.F_env[0, :, 1].sum():.1f}, '
                  f'gap={np.round(env.gap[0] * 1000, 2)}mm')
    print(f'  最终 z={z_hist[-1]:.4f} (期望 ~= 0.15)')


if __name__ == '__main__':
    qp_sanity()
    qp_warm_speed()
    force_realization()
    z_response()
