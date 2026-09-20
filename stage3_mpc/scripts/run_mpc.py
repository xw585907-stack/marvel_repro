"""阶段3 演示：Pygame 交互式仿真（SRB-MPC + 步态调度，论文1 控制器）。

键盘：
  1 / 2 / 3 / 4   表面倾角 0° / 45° / 90°（垂直墙）/ 180°（天花板）
  上/下（或 W/S）  调整沿面速度指令 v_cmd（±0.05 m/s，按表面自适应上限）
  R               复位
  Esc             退出

运行：.venv/Scripts/python.exe stage3_mpc/scripts/run_mpc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
import pygame

from envs.env import ClimbEnv
from stage3_mpc.control.controller import MPCController
from render.pygame_renderer import PygameRenderer

# 各表面的默认/上限速度（m/s），与 test_mpc 场景一致
SURFACES = {
    pygame.K_1: (0.0, 0.3, 0.4, '平地'),
    pygame.K_2: (np.pi / 4, 0.2, 0.3, '45度'),
    pygame.K_3: (np.pi / 2, 0.15, 0.25, '垂直墙'),
    pygame.K_4: (np.pi, 0.1, 0.2, '天花板'),
}


def main():
    env = ClimbEnv(num_envs=1)
    renderer = PygameRenderer()
    theta = 0.0
    env.set_gravity_theta(theta)
    env.reset()
    ctrl = MPCController(env, v_cmd=0.3)

    clock = pygame.time.Clock()
    running = True
    print('提示：请先点击 pygame 窗口使其获得焦点，再按键。')

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in SURFACES:
                    theta, v0, _, name = SURFACES[event.key]
                    env.set_gravity_theta(theta)
                    env.reset()
                    ctrl = MPCController(env, v_cmd=v0)
                    print(f'表面 → {name}（v_cmd={v0} m/s）')
                elif event.key in (pygame.K_UP, pygame.K_w):
                    ctrl.v_cmd += 0.05
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    ctrl.v_cmd -= 0.05
                elif event.key == pygame.K_r:
                    env.reset()
                    ctrl.reset()
                    print('复位')

        tau, mag, q_des = ctrl.step()
        env.step(tau[None, :], mag[None, :], q_des[None, :],
                 ctrl.info['target_w'][None, :, :])
        if env.terminated[0]:
            env.reset()
            ctrl.reset()
            print('终止条件触发，自动复位')

        snap = env.snapshot()
        renderer.draw(snap)

        # 追加 MPC HUD：状态 / 指令 / 求解耗时 / 支撑相
        lines = [
            f'v_cmd={ctrl.v_cmd:+.2f} m/s   vx={snap["base_vel"][0]:+.3f} '
            f'z={snap["base_pos"][1]:.4f} phi={np.degrees(snap["base_phi"]):+5.1f}deg',
            f'solve={ctrl.info["solve_ms"]:5.1f} ms   '
            f'stance={"".join(str(int(s)) for s in ctrl.info["stance"])}   '
            f'u=({ctrl.info["u"][:,0].sum():+5.1f},{ctrl.info["u"][:,1].sum():+5.1f})N',
            '按键: 1=地面 2=45度 3=墙面 4=天花板 | 上下=调速 | R=复位 | Esc=退出',
        ]
        for j, text in enumerate(lines):
            surf = renderer.font.render(text, True, (230, 230, 235))
            renderer.screen.blit(surf, (15, 15 + (3 + j) * 24))

        pygame.display.flip()
        clock.tick(50)  # 与控制步 20ms 对应

    renderer.close()


if __name__ == '__main__':
    main()
