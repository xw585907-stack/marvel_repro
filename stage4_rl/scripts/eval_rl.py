"""评估 PPO 检查点：速度 RMSE、提前终止、时长和吸附保持率。"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas

import numpy as np
import torch

from stage4_rl.rl import ActorCriticEstimator, PPOTrainer, RLEnvConfig, VectorClimbRLEnv
from stage4_rl.rl.ppo import to_tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='stage4_rl/checkpoints/ppo_final.pt')
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--prob-attach', type=float, default=0.85)
    p.add_argument('--command', type=float, default=0.15)
    args = p.parse_args()

    torch.set_num_threads(1)
    cfg = RLEnvConfig(num_envs=1, domain_randomization=False,
                      observation_noise=False, standing_probability=0.0)
    env = VectorClimbRLEnv(cfg)
    env.set_iteration(35000)
    env.env.prob_attach[...] = args.prob_attach
    env.set_command(args.command)
    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim,
                                 env.action_dim)
    trainer = PPOTrainer(model)
    trainer.load(args.checkpoint, load_optimizer=False)

    durations, rmses, retention, successes = [], [], [], []
    for _ in range(args.episodes):
        env.reset()
        env.set_command(args.command)
        errors, retained, stance_count = [], 0, 0
        for step in range(env.max_episode_steps):
            action, _, _, _, estimate = trainer.act(
                to_tensor(env.proprio), to_tensor(env.critic_obs), True)
            _, _, _, _, done, info = env.step(action.numpy(), estimate.numpy())
            metrics = info['transition_metrics']
            errors.append((metrics['velocity'][0] - args.command) ** 2)
            phase = metrics['stance'][0]
            retained += int(np.sum(phase & (metrics['magnetic_force'][0] > 1.0)))
            stance_count += int(np.sum(phase))
            if done[0]:
                break
        duration = (step + 1) * env.env.control_dt
        durations.append(duration)
        rmses.append(float(np.sqrt(np.mean(errors))))
        retention.append(retained / max(stance_count, 1))
        successes.append(duration >= cfg.episode_seconds - 1e-9
                         and not metrics['physical_failure'][0])

    print(f'episodes={args.episodes} p_attach={args.prob_attach:.2f}')
    print(f'速度 RMSE: {np.mean(rmses):.4f} +/- {np.std(rmses):.4f} m/s')
    print(f'提前终止率: {(1.0 - np.mean(successes)) * 100:.1f}%')
    print(f'平均时长: {np.mean(durations):.3f} s')
    print(f'吸附保持率: {np.mean(retention) * 100:.1f}%')


if __name__ == '__main__':
    main()
