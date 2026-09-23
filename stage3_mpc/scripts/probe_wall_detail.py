"""墙面步态保持的逐子步探针：确认离地踢/触地拽 + 关节饱和机制。

记录前 3 个控制步内每条腿的间隙、磁力、关节施加力矩（限幅前后）
与接触负载力矩 JᵀF_env。重点看支撑腿 1、2 的髋关节是否饱和。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from envs.robot import foot_jac
from envs.magnet import magnet_force
from stage3_mpc.control.controller import MPCController

G = 9.81


class ProbeEnv(ClimbEnv):
    """包装 _substep：记录施加力矩（限幅前/后）与负载力矩。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.rec = []

    def _substep(self, torque_cmd, magnet_cmd, q_des, g_vec, qd_des=None):
        N = self.num_envs
        foot_w, _ = self._foot_state()
        gap = foot_w[..., 1]
        mag = magnet_force(gap, self.attach_ok, magnet_cmd >= 0.5,
                           self.f_max, self.gap_lam)
        tau_pd = self.kp[:, None] * (q_des - self.q) - self.kd[:, None] * self.qd
        if qd_des is not None:
            tau_pd = tau_pd + self.kd[:, None] * qd_des
        tau_unc = torque_cmd + tau_pd
        tau = np.clip(tau_unc, -self.tau_max[:, None], self.tau_max[:, None])
        # 腿对耦合（复刻 env._substep 的 tc 计算，观察谁在锁摆动腿）
        tc = np.zeros((N, 8))
        for a, b in ((0, 2), (1, 3)):
            dq = ((self.q[:, 2 * b:2 * b + 2] - self.q[:, 2 * a:2 * a + 2])
                  - (q_des[:, 2 * b:2 * b + 2] - q_des[:, 2 * a:2 * a + 2]))
            if qd_des is not None:
                dqd = ((self.qd[:, 2 * b:2 * b + 2] - self.qd[:, 2 * a:2 * a + 2])
                       - (qd_des[:, 2 * b:2 * b + 2] - qd_des[:, 2 * a:2 * a + 2]))
            else:
                dqd = self.qd[:, 2 * b:2 * b + 2] - self.qd[:, 2 * a:2 * a + 2]
            f = (self.leg_couple_k[:, None] * dq +
                 self.leg_couple_d[:, None] * dqd)
            tc[:, 2 * a:2 * a + 2] += f
            tc[:, 2 * b:2 * b + 2] -= f
        self.rec.append({'gap': gap.copy(), 'mag': mag.copy(),
                         'tau': tau.copy(), 'tau_unc': tau_unc.copy(),
                         'tc': tc.copy(), 'q': self.q.copy(),
                         'q_des': q_des.copy()})
        super()._substep(torque_cmd, magnet_cmd, q_des, g_vec, qd_des)
        self.rec[-1]['F_env'] = self.F_env.copy()
        J = foot_jac(self.q, self.l1, self.l2)
        c, s = np.cos(self.phi), np.sin(self.phi)
        R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -1)
        J_w = np.einsum('nij,nfjk->nfik', R, J)
        self.rec[-1]['tau_load'] = np.einsum(
            'nfij,nfi->nfj', J_w, self.F_env).reshape(N, 8)


def main():
    env = ProbeEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.0)
    n_show = 3                      # 记录前 3 个控制步
    show_ss = [0, 2, 5, 10, 20, 50, 99]
    for k in range(n_show):
        tau, mag, q_des = ctrl.step()
        env.rec = []
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        s = env.snapshot()
        print(f'--- 控制步 k={k} stance={ctrl.info["stance"].astype(int)} '
              f'u_z={ctrl.info["u"][:,1].sum():+.1f} '
              f'u_x={ctrl.info["u"][:,0].sum():+.1f}')
        for si in show_ss:
            r = env.rec[si]
            legs = (1, 0)           # 支撑腿 FR、摆动腿 RR
            parts = []
            for i in legs:
                parts.append(
                    f'L{i}:gap={r["gap"][0,i]*1000:+.2f}mm '
                    f'mag={r["mag"][0,i]:6.0f} '
                    f'Fz={r["F_env"][0,i,1]:+6.0f} '
                    f'tau_h={r["tau"][0,2*i]:+6.1f}'
                    f'({r["tau_unc"][0,2*i]:+6.1f}) '
                    f'tau_k={r["tau"][0,2*i+1]:+6.1f}'
                    f'({r["tau_unc"][0,2*i+1]:+6.1f}) '
                    f'tc_h={r["tc"][0,2*i]:+6.1f} '
                    f'tc_k={r["tc"][0,2*i+1]:+6.1f} '
                    f'q=({r["q"][0,2*i]:.3f},{r["q"][0,2*i+1]:.3f})'
                    f'/d({r["q_des"][0,2*i]:.3f},{r["q_des"][0,2*i+1]:.3f})')
            b = env.rec[si]
            Fz = b['F_env'][0,:,1].sum()
            print(f'  ss={si:3d} z={s["base_pos"][1]:+.4f} '
                  f'Fz总={Fz:+7.1f} | ' + ' | '.join(parts))
        print(f'  步末: p={s["base_pos"][1]:+.4f} vz={s["base_vel"][1]:+.3f} '
              f'phi={s["base_phi"]:+.3f}')


if __name__ == '__main__':
    main()
