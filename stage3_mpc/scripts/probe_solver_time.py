"""求解器分阶段计时:build/约束/coneqp/polish/回退 各占多少。

跑一段墙面步态(与 test_mpc 场景2 同配置),统计各阶段耗时与回退次数。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
from cvxopt import matrix, solvers

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController

G = 9.81

t_build = t_con = t_qp = t_pol = t_pen = 0.0
n_qp = n_pen = 0
nits = []


def main():
    global t_build, t_con, t_qp, t_pol, t_pen, n_qp, n_pen
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.15)
    mpc = ctrl.mpc

    # 包装各阶段
    orig_build = mpc._build_prediction
    orig_con = mpc._constraints
    orig_qp = mpc._solve_qp
    orig_pol = mpc._polish_active_set
    orig_pen = mpc._solve_qp_penalty

    def w_build(*a, **k):
        global t_build
        t0 = time.perf_counter()
        r = orig_build(*a, **k)
        t_build += time.perf_counter() - t0
        return r

    def w_con(*a, **k):
        global t_con
        t0 = time.perf_counter()
        r = orig_con(*a, **k)
        t_con += time.perf_counter() - t0
        return r

    def w_qp(*a, **k):
        global t_qp, n_qp
        t0 = time.perf_counter()
        r = orig_qp(*a, **k)
        t_qp += time.perf_counter() - t0
        n_qp += 1
        return r

    def w_pol(*a, **k):
        global t_pol
        t0 = time.perf_counter()
        r = orig_pol(*a, **k)
        t_pol += time.perf_counter() - t0
        return r

    def w_pen(*a, **k):
        global t_pen, n_pen
        t0 = time.perf_counter()
        r = orig_pen(*a, **k)
        t_pen += time.perf_counter() - t0
        n_pen += 1
        return r

    mpc._build_prediction = w_build
    mpc._constraints = w_con
    mpc._solve_qp = w_qp
    mpc._polish_active_set = w_pol
    mpc._solve_qp_penalty = w_pen

    n_steps = 120
    for k in range(n_steps):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        nits.append(mpc.last_nit)

    print(f'steps={n_steps} 总 solve_ms={mpc.solve_ms:.1f} (最后一步)')
    print(f'build:   {t_build*1000/n_steps:7.2f} ms/步')
    print(f'约束组装: {t_con*1000/n_steps:7.2f} ms/步')
    print(f'coneqp:  {t_qp*1000/n_steps:7.2f} ms/步 ({n_qp} 次)')
    print(f'polish:  {t_pol*1000/n_steps:7.2f} ms/步')
    print(f'penalty: {t_pen*1000/n_steps:7.2f} ms/步 ({n_pen} 次回退)')
    print(f'coneqp it 分布: min={min(nits)} max={max(nits)} '
          f'avg={np.mean(nits):.1f}')
    itv, cnt = np.unique(nits, return_counts=True)
    print(f'it 直方图: {dict(zip(itv.tolist(), cnt.tolist()))}')


if __name__ == '__main__':
    main()
