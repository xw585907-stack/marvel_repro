"""可重复的批量评估：保存每回合数据和汇总，不把存活当成速度跟踪成功。"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas
import numpy as np
import torch
from stage4_rl.rl import ActorCriticEstimator, PPOTrainer, RLEnvConfig, VectorClimbRLEnv
from stage4_rl.rl.ppo import to_tensor


def evaluate(checkpoint, episodes, theta, probability, command, seed, oracle=False,
             iteration=35000, stochastic=False, randomized=False):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved_env = saved.get('extra', {}).get('env_config', {})
    cfg = RLEnvConfig(num_envs=episodes, seed=seed, domain_randomization=randomized,
                      observation_noise=randomized, standing_probability=0.0,
                      command_min=command, command_max=command,
                      gait_prior=bool(saved_env.get('gait_prior', False)),
                      residual_action_scale=float(saved_env.get('residual_action_scale', .25)),
                      curriculum_scale=float(saved_env.get('curriculum_scale', 1.0)))
    env = VectorClimbRLEnv(cfg)
    env.set_iteration(iteration)
    env.env.set_gravity_theta(np.radians(theta))
    env.env.prob_attach[:] = probability
    env.reset()
    env.set_command(command)
    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim, env.action_dim)
    trainer = PPOTrainer(model)
    trainer.load(checkpoint, load_optimizer=False)
    alive = np.ones(episodes, dtype=bool)
    steps = np.zeros(episodes, dtype=int)
    errors = np.zeros(episodes)
    retained = np.zeros(episodes)
    stance_count = np.zeros(episodes)
    displacement = np.zeros(episodes)
    failures = np.zeros(episodes, dtype=bool)
    saturation_count = 0
    action_count = 0
    mean_abs_max = 0.0
    for _ in range(env.max_episode_steps):
        action, raw, _, _, estimate = trainer.act(to_tensor(env.proprio),
                                               to_tensor(env.critic_obs), not stochastic)
        means = action.numpy()[alive]
        mean_abs_max = max(mean_abs_max, float(np.abs(means).max()))
        saturation_count += int((np.abs(means) >= 0.99).sum())
        action_count += means.size
        _, _, _, _, done, info = env.step(action.numpy(), None if oracle else estimate.numpy())
        m = info['transition_metrics']
        steps[alive] += 1
        errors[alive] += (m['velocity'][alive] - command) ** 2
        retained[alive] += np.sum(m['stance'][alive] & (m['magnetic_force'][alive] > 1), axis=1)
        stance_count[alive] += np.sum(m['stance'][alive], axis=1)
        displacement[alive] = m['position'][alive, 0]
        failures[alive] = m['physical_failure'][alive]
        alive &= ~done.astype(bool)
        if not alive.any():
            break
    rows = []
    for i in range(episodes):
        rows.append(dict(episode=i, duration_s=float(steps[i] * env.env.control_dt),
                         velocity_rmse=float(np.sqrt(errors[i] / steps[i])),
                         retention=float(retained[i] / max(stance_count[i], 1)),
                         displacement_m=float(displacement[i]),
                         survived=bool(steps[i] == env.max_episode_steps and not failures[i])))
    summary = dict(checkpoint=str(checkpoint), sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   episodes=episodes, seed=seed, theta_deg=theta, prob_attach=probability,
                   command=command, oracle_contact=oracle, stochastic_policy=stochastic,
                   gait_prior=cfg.gait_prior, randomized=randomized,
                   curriculum_iteration=iteration,
                   adhesion_enabled=env.curriculum_state.adhesion_enabled,
                   policy_mean_abs_max=mean_abs_max,
                   policy_mean_saturation=saturation_count / action_count,
                   mean_duration_s=float(np.mean([r['duration_s'] for r in rows])),
                   mean_velocity_rmse=float(np.mean([r['velocity_rmse'] for r in rows])),
                   mean_retention=float(np.mean([r['retention'] for r in rows])),
                   mean_displacement_m=float(np.mean(displacement)),
                   survival_rate=float(np.mean([r['survived'] for r in rows])))
    return summary, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--theta', type=float, default=90)
    p.add_argument('--prob-attach', type=float, default=0.85)
    p.add_argument('--command', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--iteration', type=int, default=35000,
                   help='评估课程阶段；平地阶段一用 0，关闭磁吸，与训练一致')
    p.add_argument('--oracle-contact', action='store_true', help='仅用于诊断，不是正式策略指标')
    p.add_argument('--stochastic', action='store_true', help='诊断训练时的随机策略；正式评估默认确定性')
    p.add_argument('--randomized', action='store_true', help='评估时开启训练使用的域随机化和观测噪声')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.episodes <= 0 or not 0 <= args.prob_attach <= 1:
        p.error('episodes 必须为正，prob-attach 必须在 [0,1]')
    torch.set_num_threads(1)
    summary, rows = evaluate(args.checkpoint, args.episodes, args.theta,
                             args.prob_attach, args.command, args.seed, args.oracle_contact,
                             args.iteration, args.stochastic, args.randomized)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
