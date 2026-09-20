"""阶段3 论文1 MPC 控制器验证脚本（无头运行，不需要 pygame）。

验证项：
  1. 平地小跑（θ=0，v_cmd=0.3 m/s）：稳态速度 ≈ 0.3 m/s
  2. 垂直墙爬行（θ=90°，v_cmd=0.15 m/s）：稳态爬升速度 ≈ 0.15 m/s、
     保持吸附（体高≈0.15 m）、未终止——对应论文1 垂直爬行
  3. 垂直墙原地保持（θ=90°，v_cmd=0）：不滑落（|v_x|≈0）
  4. 天花板悬吊爬行（θ=180°，v_cmd=0.1 m/s）：磁吸承载全部体重、
     稳态速度 ≈ 0.1 m/s

运行：.venv/Scripts/python.exe stage3_mpc/scripts/test_mpc.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController

PASS = []
FAIL = []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"通过" if cond else "失败"}] {name}  {detail}')


def run_controlled(env, ctrl, seconds):
    """运行闭环仿真，返回 (最终快照, v_x 序列, z 序列, φ 序列, 求解耗时)。"""
    vx_hist, z_hist, phi_hist, solve_hist = [], [], [], []
    for _ in range(int(seconds / env.control_dt)):
        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        s = env.snapshot()
        vx_hist.append(s['base_vel'][0])
        z_hist.append(s['base_pos'][1])
        phi_hist.append(s['base_phi'])
        solve_hist.append(ctrl.mpc.solve_ms)
        if s['terminated']:
            print(f'  ! 终止于 t={s["time"]:.2f}s（φ={s["base_phi"]:.2f}, '
                  f'x={s["base_pos"][0]:.2f}, z={s["base_pos"][1]:.2f}）')
            break
    return (s, np.array(vx_hist), np.array(z_hist), np.array(phi_hist),
            np.array(solve_hist))


def steady_avg(x, frac=0.4):
    """取后 60% 段的平均值（跳过起步瞬态）。"""
    n = max(int(len(x) * frac), 1)
    return x[-n:].mean() if len(x) else np.nan


def test_flat_trot():
    print('1. 平地小跑（θ=0，v_cmd=0.3 m/s，6s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)
    s, vx, z, phi, ms = run_controlled(env, ctrl, 6.0)
    check('未终止', not s['terminated'])
    check('稳态速度 ≈ 0.3 m/s (±0.08)', abs(steady_avg(vx) - 0.3) < 0.08,
          f"vx_avg={steady_avg(vx):.3f} m/s")
    check('体高 ≈ 0.15 m (±2cm)', abs(np.median(z) - 0.15) < 0.02,
          f"z_med={np.median(z):.4f} m")
    check('俯仰有界 |φ| < 0.3 rad', np.abs(phi).max() < 0.3,
          f"max_phi={np.abs(phi).max():.3f}")
    check('MPC 平均求解 < 20 ms', ms.mean() < 20.0,
          f"{ms.mean():.2f} ms (max {ms.max():.2f})")


def test_wall_climb():
    print('2. 垂直墙爬行（θ=90°，v_cmd=0.15 m/s，6s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.15)
    s, vx, z, phi, ms = run_controlled(env, ctrl, 6.0)
    check('未终止（保持吸附）', not s['terminated'])
    check('稳态爬升速度 ≈ 0.15 m/s (±0.08)', abs(steady_avg(vx) - 0.15) < 0.08,
          f"vx_avg={steady_avg(vx):.3f} m/s")
    check('体高 ≈ 0.15 m (±2cm)', abs(np.median(z) - 0.15) < 0.02,
          f"z_med={np.median(z):.4f} m")
    check('俯仰有界 |φ| < 0.3 rad', np.abs(phi).max() < 0.3,
          f"max_phi={np.abs(phi).max():.3f}")
    check('MPC 平均求解 < 20 ms', ms.mean() < 20.0,
          f"{ms.mean():.2f} ms (max {ms.max():.2f})")


def test_wall_hold():
    print('3. 垂直墙原地保持（θ=90°，v_cmd=0，4s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.0)
    s, vx, z, phi, ms = run_controlled(env, ctrl, 4.0)
    check('未终止', not s['terminated'])
    check('不滑落（|vx_avg| < 0.05 m/s）', abs(steady_avg(vx)) < 0.05,
          f"vx_avg={steady_avg(vx):.3f} m/s")
    check('体高 ≈ 0.15 m (±2cm)', abs(np.median(z) - 0.15) < 0.02,
          f"z_med={np.median(z):.4f} m")


def test_ceiling_walk():
    print('4. 天花板悬吊爬行（θ=180°，v_cmd=0.1 m/s，6s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.1)
    s, vx, z, phi, ms = run_controlled(env, ctrl, 6.0)
    check('未终止（磁吸承载体重）', not s['terminated'])
    check('稳态速度 ≈ 0.1 m/s (±0.08)', abs(steady_avg(vx) - 0.1) < 0.08,
          f"vx_avg={steady_avg(vx):.3f} m/s")
    check('体高 ≈ 0.15 m (±2cm)', abs(np.median(z) - 0.15) < 0.02,
          f"z_med={np.median(z):.4f} m")
    check('俯仰有界 |φ| < 0.3 rad', np.abs(phi).max() < 0.3,
          f"max_phi={np.abs(phi).max():.3f}")


def main():
    t0 = time.perf_counter()
    test_flat_trot()
    test_wall_climb()
    test_wall_hold()
    test_ceiling_walk()
    print(f'\n结果：{len(PASS)} 通过 / {len(FAIL)} 失败 '
          f'（总耗时 {time.perf_counter() - t0:.1f}s）')
    if FAIL:
        print('失败项：', *FAIL, sep='\n  - ')
        sys.exit(1)


if __name__ == '__main__':
    main()
