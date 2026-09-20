"""阶段2 物理内核验证脚本（无头运行，不需要 pygame）。

验证项：
  1. 平地站立（θ=0）：体高≈0.15m、俯仰/速度≈0
  2. 垂直墙吸附（θ=90°，磁开）：保持吸附、无滑移、F_mag≈697N、
     总摩擦力≈mg（对应论文1 垂直爬行的静力平衡）
  3. 垂直墙断磁（θ=90°，磁关）：沿壁面加速滑落（无摩擦承载）
  4. 随机吸附统计（Prob_attach=0.85）：大量落地事件中吸附成功率≈85%

运行：.venv/Scripts/python.exe scripts/test_env.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np

from envs.env import ClimbEnv

PASS = []
FAIL = []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    print(f'  [{"通过" if cond else "失败"}] {name}  {detail}')


def run_sim(env, seconds, magnet_on, q_des):
    magnet_cmd = np.ones((1, 4)) if magnet_on else np.zeros((1, 4))
    for _ in range(int(seconds / env.control_dt)):
        env.step(np.zeros((1, 8)), magnet_cmd, q_des)
        if env.terminated[0]:
            break
    return env.snapshot()


def test_flat_ground():
    print('1. 平地站立（θ=0，磁开，5s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.reset()
    s = run_sim(env, 5.0, True, env.q_nominal.copy())
    check('体高 ≈ 0.15 m', abs(s['base_pos'][1] - 0.15) < 0.01,
          f"z={s['base_pos'][1]:.4f}")
    check('俯仰角 ≈ 0', abs(s['base_phi']) < 0.05, f"phi={s['base_phi']:.4f}")
    check('速度 ≈ 0', np.linalg.norm(s['base_vel']) < 0.05,
          f"v={s['base_vel']}")
    check('无 NaN', np.isfinite(np.concatenate([s['base_pos'], s['q']])).all())


def test_wall_cling():
    print('2. 垂直墙吸附（θ=90°，磁开，5s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    s = run_sim(env, 5.0, True, env.q_nominal.copy())
    check('保持吸附（未终止）', not s['terminated'])
    check('体高 ≈ 0.15 m', abs(s['base_pos'][1] - 0.15) < 0.01,
          f"z={s['base_pos'][1]:.4f}")
    check('无滑移（|v_x|≈0）', abs(s['base_vel'][0]) < 0.05,
          f"v_x={s['base_vel'][0]:.4f}")
    check('磁吸力 ≈ 697 N/脚', np.allclose(s['f_mag'], 697.1, rtol=0.05),
          f"F_mag={s['f_mag'].round(0)}")
    friction = s['F_env'][:, 0].sum()  # 切向合力（沿壁面方向）
    check('总摩擦力 ≈ mg', abs(friction - 8 * 9.81) < 5.0,
          f"ΣF_t={friction:.1f} N (mg={8 * 9.81:.1f})")


def test_magnet_off():
    print('3. 垂直墙断磁（θ=90°，磁关，1s）')
    env = ClimbEnv(num_envs=1, seed=0)
    env.set_gravity_theta(np.pi / 2)
    env.reset()
    # 先吸附 1s 稳定，再断磁
    s = run_sim(env, 1.0, True, env.q_nominal.copy())
    s = run_sim(env, 1.0, False, env.q_nominal.copy())
    check('断磁后明显滑落加速', abs(s['base_vel'][0]) > 2.0,
          f"v_x={s['base_vel'][0]:.2f} m/s")
    check('磁吸力为零', np.allclose(s['f_mag'], 0.0))


def test_attach_probability():
    print('4. 随机吸附统计（Prob_attach=0.85，200 次落地）')
    env = ClimbEnv(num_envs=1, seed=123)
    env.set_gravity_theta(np.pi / 2)
    env.prob_attach[:] = 0.85
    q_des = env.q_nominal.copy()
    landed = np.zeros(4)
    attached = np.zeros(4)
    for _ in range(200):
        env.reset()
        env.step(np.zeros((1, 8)), np.ones((1, 4)), q_des)
        s = env.snapshot()
        landed += s['contact'].astype(int)
        attached += s['attach_ok'].astype(int)
    rate = attached.sum() / max(landed.sum(), 1)
    check('吸附成功率 ≈ 85% (±5%)', abs(rate - 0.85) < 0.05,
          f"{rate * 100:.1f}% ({int(attached.sum())}/{int(landed.sum())})")


def main():
    test_flat_ground()
    test_wall_cling()
    test_magnet_off()
    test_attach_probability()
    print(f'\n结果：{len(PASS)} 通过 / {len(FAIL)} 失败')
    if FAIL:
        print('失败项：', *FAIL, sep='\n  - ')
        sys.exit(1)


if __name__ == '__main__':
    main()
