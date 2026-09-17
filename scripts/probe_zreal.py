"""探查 z 通道力实现链：全支撑持形，u_z 阶梯变化，逐步打印
每腿 gap/f_spring/F_env_z/knee 误差/PD 力矩（含限幅前）与载荷力矩。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from envs.robot import foot_jac


def main():
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    # 先重力补偿持形 12 步（u_z=mg/4≈19.6），稳定后再阶梯加 u_z
    u_z_plan = [19.6] * 12 + [39.2] * 6 + [78.4] * 6
    snap = env.snapshot()
    for k, u_z in enumerate(u_z_plan):
        snap = env.snapshot()
        u_cmd = np.zeros((4, 2))
        u_cmd[:, 1] = u_z
        target_w = snap['foot_pos'].copy()          # 原地钉扎
        c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
        R = np.array([[c, -s], [s, c]])
        J = foot_jac(snap['q'], env.l1, env.l2)
        J_w = np.einsum('ij,fjk->fik', R, J)
        tau = -np.einsum('fij,fj->fi', J_w, u_cmd).ravel()
        q_des = np.zeros(8)
        env.step(tau[None, :], np.ones((1, 4)), q_des[None, :],
                 target_w[None, :, :])
        s2 = env.snapshot()
        gap = s2['foot_pos'][:, 1] * 1000            # mm
        f_spring = env.contact_model.k_n * np.clip(-s2['foot_pos'][:, 1], 0, None)
        # 每腿 F_env_z = f_spring − f_mag（忽略阻尼项，稳态近似）
        f_env_z = s2['F_env'][:, 1]
        # 腿 0 的 knee 误差与限幅前 PD 力矩
        qk_err = env.q[0, 1] - env.q_nominal[0, 1]
        qh_err = env.q[0, 0] - env.q_nominal[0, 0]
        tau_pd_pre = env.kp[0] * np.array([qh_err, qk_err])
        phi_mrad = float(s2['base_phi']) * 1e3
        omega = float(s2['base_omega'])
        if k <= 14:
            print(f'k={k:2d} u_z={u_z:5.1f} gap={np.round(gap, 3)}mm '
                  f'F_env_z={np.round(f_env_z, 1)} '
                  f'z={s2["base_pos"][1]:.4f} vz={s2["base_vel"][1]:+.3f} '
                  f'phi={phi_mrad:+.2f}mrad omega={omega:+.2f} '
                  f'knee_err={qk_err * 1e3:+.2f}mrad')
    print('期望: u_z=19.6 → F_env_z≈19.5/腿；39.2→39.0；78.4→77.9')


if __name__ == '__main__':
    main()
