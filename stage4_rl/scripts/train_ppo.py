"""阶段4：训练论文2的二维 PPO 磁吸附攀爬策略。"""
import argparse
import hashlib
import json
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
    p.add_argument(
        '--device', default='auto',
        help='训练设备：auto 自动优先使用 CUDA，也可显式指定 cpu/cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--curriculum-scale', type=float, default=1.0)
    p.add_argument('--save-dir', default='stage4_rl/checkpoints')
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--resume', default='')
    p.add_argument('--bounded-policy', action='store_true', help='新实验：用 tanh 约束高斯动作均值')
    p.add_argument('--target-kl', type=float, default=None)
    p.add_argument('--learning-rate', type=float, default=None,
                   help='可选学习率；续训时覆盖检查点中的学习率')
    p.add_argument('--gait-prior', action='store_true',
                   help='二维适配实验：解析爬行轨迹加 PPO 残差动作')
    p.add_argument('--quick', action='store_true',
                   help='4次迭代的接口冒烟训练，不代表论文收敛结果')
    return p.parse_args()


def resolve_device(requested):
    """选择训练设备，并避免请求 CUDA 时静默回退到 CPU。"""
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(
            '请求了 CUDA 训练，但当前 PyTorch 无法使用显卡。请检查 CUDA 版 '
            'PyTorch、NVIDIA 驱动和 torch.cuda.is_available()。')
    return device


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
    device = resolve_device(args.device)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
        gpu_name = torch.cuda.get_device_name(device)
        print(f'训练设备: {device} ({gpu_name}), PyTorch={torch.__version__}, '
              f'CUDA={torch.version.cuda}', flush=True)
    else:
        print(f'训练设备: {device}, PyTorch={torch.__version__}', flush=True)

    env_cfg = RLEnvConfig(num_envs=args.num_envs, seed=args.seed,
                          curriculum_scale=args.curriculum_scale,
                          gait_prior=args.gait_prior)
    env = VectorClimbRLEnv(env_cfg)
    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim,
                                 env.action_dim)
    ppo_cfg = PPOConfig(minibatch_size=min(1024,
                                           args.num_envs * args.rollout), target_kl=args.target_kl or 0.0)
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError('learning-rate 必须为正数')
        ppo_cfg.learning_rate = args.learning_rate
    model.bounded_policy = args.bounded_policy
    trainer = PPOTrainer(model, ppo_cfg, device)
    start_iteration = 0
    if args.resume:
        loaded_iteration, _ = trainer.load(args.resume)
        if model.bounded_policy and trainer.policy_transform_version < 2:
            raise ValueError('旧 bounded-policy 检查点使用硬裁剪采样，不能继续训练；请创建新实验。')
        if args.bounded_policy and not model.bounded_policy:
            raise ValueError('旧检查点不使用 bounded-policy；请创建新实验，不能静默改变策略分布。')
        if args.target_kl is not None:
            trainer.config.target_kl = args.target_kl
        if args.learning_rate is not None:
            trainer.config.learning_rate = args.learning_rate
            for group in trainer.optimizer.param_groups:
                group['lr'] = args.learning_rate
        ppo_cfg = trainer.config
        start_iteration = loaded_iteration + 1
        print(f'继续训练: {args.resume}, 从 iteration={start_iteration} 开始')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / 'ppo_final.pt').exists() and not args.resume:
        raise FileExistsError(f'{save_dir} 已有最终模型，请为新实验选择新目录。')
    source_root = Path(__file__).resolve().parents[2]
    source_paths = ['envs/env.py', 'stage4_rl/rl/env_adapter.py',
                    'stage4_rl/rl/networks.py', 'stage4_rl/rl/ppo.py',
                    'stage4_rl/scripts/train_ppo.py']
    provenance = {p: hashlib.sha256((source_root / p).read_bytes()).hexdigest()
                  for p in source_paths}
    with (save_dir / 'runs.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(dict(arguments=vars(args), source_sha256=provenance)) + '\n')
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
            if info['terminal_critic_obs'] is not None:
                with torch.no_grad():
                    terminal_value = model.value(to_tensor(info['terminal_critic_obs'], device))
                reward = reward + ppo_cfg.gamma * terminal_value.cpu().numpy() * info['time_outs']
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
            completed = iteration - start_iteration + 1
            remaining = max(args.iterations - iteration - 1, 0)
            eta_seconds = elapsed / completed * remaining
            with torch.no_grad():
                distribution, _ = model.distribution(buffer.proprio[0])
                policy_mean = distribution.mean.tanh() if model.bounded_policy else distribution.mean
                mean_max = policy_mean.abs().max().item()
                mean_saturation = (policy_mean.abs() >= 0.99).float().mean().item()
            metrics = dict(iteration=iteration, elapsed_s=elapsed,
                           phase=state.phase, ep100=mean_ep if np.isfinite(mean_ep) else None,
                           mean_max=mean_max, mean_saturation=mean_saturation,
                           reward_terms=terms,
                           **losses)
            with (save_dir / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, allow_nan=False) + '\n')
            print(
                f'iter={iteration:5d} phase={state.phase} '
                f'theta={np.degrees(state.theta):5.1f}deg '
                f'p_attach={state.prob_attach:.3f} '
                f'reward={terms.get("total", np.nan):+.3f} '
                f'ep100={mean_ep:+.2f} policy={losses["policy"]:+.3f} '
                f'value={losses["value"]:.3f} est={losses["estimator"]:.3f} '
                f'kl={losses["kl"]:.4f} mean_max={mean_max:.2f} '
                f'mean_sat={mean_saturation:.1%} '
                f't={elapsed:.1f}s eta={eta_seconds / 3600.0:.2f}h',
                flush=True)

        if (iteration + 1) % args.save_every == 0:
            trainer.save(
                save_dir / f'ppo_iter_{iteration + 1:05d}.pt', iteration,
                extra={'env_config': asdict(env_cfg), 'source_sha256': provenance})

    final_path = save_dir / 'ppo_final.pt'
    trainer.save(final_path, args.iterations - 1,
                 extra={'env_config': asdict(env_cfg), 'source_sha256': provenance})
    print(f'训练结束，模型已保存到 {final_path}', flush=True)


if __name__ == '__main__':
    main()
