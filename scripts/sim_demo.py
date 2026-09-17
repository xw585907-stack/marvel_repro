"""阶段2 演示：Pygame 交互式仿真（固定站立位型 + 磁脚开关）。

键盘：
  1 / 2 / 3   重力倾角 0° / 45° / 90°（90° = 垂直墙面，渲染自动旋转视角）
  M           磁脚开关（全开/全关）
  R           复位
  Esc         退出

运行：.venv/Scripts/python.exe scripts/sim_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
import pygame

from envs.env import ClimbEnv
from render.pygame_renderer import PygameRenderer


def main():
    env = ClimbEnv(num_envs=1)
    renderer = PygameRenderer()
    env.reset()

    q_des = env.q_nominal.copy()
    magnet_on = True
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
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    env.set_gravity_theta(0.0)
                    print('倾角 → 0°（平地）')
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    env.set_gravity_theta(np.pi / 4)
                    print('倾角 → 45°')
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    env.set_gravity_theta(np.pi / 2)
                    print('倾角 → 90°（垂直墙）')
                elif event.key == pygame.K_m:
                    magnet_on = not magnet_on
                    print(f'磁开关 → {"开" if magnet_on else "关"}')
                elif event.key == pygame.K_r:
                    env.reset()
                    print('复位')

        torque = np.zeros((1, 8))
        magnet_cmd = np.ones((1, 4)) if magnet_on else np.zeros((1, 4))
        env.step(torque, magnet_cmd, q_des)
        if env.terminated[0]:
            env.reset()

        renderer.draw(env.snapshot())
        pygame.display.flip()
        clock.tick(50)  # 与 control_dt=20ms 对应

    renderer.close()


if __name__ == '__main__':
    main()
