"""加载阶段4 PPO 权重并用 Pygame 演示。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas

import numpy as np
import pygame
import torch

from render.pygame_renderer import PygameRenderer
from stage4_rl.rl import ActorCriticEstimator, PPOTrainer, RLEnvConfig, VectorClimbRLEnv
from stage4_rl.rl.ppo import to_tensor


SURFACES = {
    pygame.K_1: (0.0, 0.30, '平地'),
    pygame.K_2: (np.pi / 4, 0.20, '45度'),
    pygame.K_3: (np.pi / 2, 0.15, '垂直墙'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='stage4_rl/checkpoints/ppo_final.pt')
    args = parser.parse_args()
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(
            f'未找到 {args.checkpoint}，请先运行 stage4_rl/scripts/train_ppo.py')

    torch.set_num_threads(1)
    cfg = RLEnvConfig(num_envs=1, domain_randomization=False,
                      observation_noise=False, standing_probability=0.0)
    env = VectorClimbRLEnv(cfg)
    env.set_iteration(35000)
    env.set_command(0.15)
    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim,
                                 env.action_dim)
    trainer = PPOTrainer(model)
    iteration, _ = trainer.load(args.checkpoint, load_optimizer=False)
    renderer = PygameRenderer()
    clock = pygame.time.Clock()
    running = True
    print(f'已加载 iteration={iteration}；1/2/3 切换表面，方向键调速。')

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in SURFACES:
                    theta, command, name = SURFACES[event.key]
                    env.env.set_gravity_theta(theta)
                    env.reset()
                    env.set_command(command)
                    print(f'表面 -> {name}, v_cmd={command:.2f}m/s')
                elif event.key in (pygame.K_UP, pygame.K_w):
                    env.set_command(np.clip(env.command + 0.05, -0.5, 0.5))
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    env.set_command(np.clip(env.command - 0.05, -0.5, 0.5))
                elif event.key == pygame.K_r:
                    env.reset()

        with torch.no_grad():
            action, _, _, _, estimate = trainer.act(
                to_tensor(env.proprio), to_tensor(env.critic_obs),
                deterministic=True)
        _, _, _, _, _, info = env.step(action.numpy(), estimate.numpy())
        snap = env.env.snapshot()
        renderer.draw(snap)
        hud = [
            f'RL iteration={iteration} v_cmd={env.command[0]:+.2f} '
            f'vx={snap["base_vel"][0]:+.3f} reward={info["reward_terms"]["total"]:+.2f}',
            f'phase={env.curriculum_state.phase} '
            f'p_attach={env.curriculum_state.prob_attach:.2f}',
            '按键: 1=平地 2=45度 3=墙面 | 上下=调速 | R=复位 | Esc=退出',
        ]
        for j, line in enumerate(hud):
            surf = renderer.font.render(line, True, (230, 230, 235))
            renderer.screen.blit(surf, (15, 15 + (3 + j) * 24))
        pygame.display.flip()
        clock.tick(50)

    renderer.close()


if __name__ == '__main__':
    main()
