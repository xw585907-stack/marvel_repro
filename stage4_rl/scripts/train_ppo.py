"""阶段4：训练论文2的二维 PPO 磁吸附攀爬策略。"""
import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas

import numpy as np
import torch

from stage4_rl.rl import (ActorCriticEstimator, PPOConfig, PPOTrainer, RLEnvConfig,
                RolloutBuffer, VectorClimbRLEnv)
from stage4_rl.rl.ppo import to_tensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--iterations', type=int, default=35000)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--rollout', type=int, default=32)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--curriculum-scale', type=float, default=1.0)
    p.add_argument('--save-dir', default='stage4_rl/checkpoints')
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--resume', default='')
    p.add_argument('--quick', action='store_true',
                   help='4次迭代的接口冒烟训练，不代表论文收敛结果')
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.iterations = 4
        args.num_envs = 8
        args.rollout = 8
        args.curriculum_scale = 0.001
        args.save_every = 2

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)

    env_cfg = RLEnvConfig(num_envs=args.num_envs, seed=args.seed,
                          curriculum_scale=args.curriculum_scale)
    env = VectorClimbRLEnv(env_cfg)
    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim,
                                 env.action_dim)
    ppo_cfg = PPOConfig(minibatch_size=min(1024,
                                           args.num_envs * args.rollout))
    trainer = PPOTrainer(model, ppo_cfg, device)
    start_iteration = 0
    if args.resume:
        loaded_iteration, _ = trainer.load(args.resume)
        start_iteration = loaded_iteration + 1
        print(f'继续训练: {args.resume}, 从 iteration={start_iteration} 开始')

    save_dir = Path(args.save_dir)
    episode_returns = []
    t_start = time.perf_counter()
    proprio, critic_obs, privileged = env.proprio, env.critic_obs, env.privileged

    for iteration in range(start_iteration, args.iterations):
        env.set_iteration(iteration)
        buffer = RolloutBuffer(
            args.rollout, args.num_envs, env.proprio_dim, env.critic_dim,
            env.privileged_dim, env.action_dim, device)
        last_info = None
        for _ in range(args.rollout):
            p_t = to_tensor(proprio, device)
            c_t = to_tensor(critic_obs, device)
            action, raw_action, logp, value, estimate = trainer.act(p_t, c_t)
            next_p, next_c, next_priv, reward, done, info = env.step(
                action.cpu().numpy(), estimate.cpu().numpy())
            buffer.add(p_t, c_t, to_tensor(privileged, device), raw_action,
                       logp, value, to_tensor(reward, device),
                       to_tensor(done, device))
            if info['episode_returns'].size:
                episode_returns.extend(info['episode_returns'].tolist())
                episode_returns = episode_returns[-100:]
            proprio, critic_obs, privileged = next_p, next_c, next_priv
            last_info = info

        with torch.no_grad():
            last_value = model.value(to_tensor(critic_obs, device))
        buffer.finish(last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
        losses = trainer.update(buffer)

        if iteration % 10 == 0 or iteration == args.iterations - 1:
            state = env.curriculum_state
            mean_ep = float(np.mean(episode_returns)) if episode_returns else np.nan
            terms = last_info['reward_terms'] if last_info else {}
            elapsed = time.perf_counter() - t_start
            print(
                f'iter={iteration:5d} phase={state.phase} '
                f'theta={np.degrees(state.theta):5.1f}deg '
                f'p_attach={state.prob_attach:.3f} '
                f'reward={terms.get("total", np.nan):+.3f} '
                f'ep100={mean_ep:+.2f} policy={losses["policy"]:+.3f} '
                f'value={losses["value"]:.3f} est={losses["estimator"]:.3f} '
                f't={elapsed:.1f}s')

        if (iteration + 1) % args.save_every == 0:
            trainer.save(
                save_dir / f'ppo_iter_{iteration + 1:05d}.pt', iteration,
                extra={'env_config': asdict(env_cfg)})

    final_path = save_dir / 'ppo_final.pt'
    trainer.save(final_path, args.iterations - 1,
                 extra={'env_config': asdict(env_cfg)})
    print(f'训练结束，模型已保存到 {final_path}')


if __name__ == '__main__':
    main()
