"""轻量 PPO 实现，供论文2二维复现使用。"""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    update_epochs: int = 5
    minibatch_size: int = 1024
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    estimator_coef: float = 1.0
    max_grad_norm: float = 1.0
    target_kl: float = 0.0


class RolloutBuffer:
    def __init__(self, horizon, num_envs, proprio_dim, critic_dim,
                 privileged_dim, action_dim, device='cpu'):
        shape = (horizon, num_envs)
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.proprio = torch.zeros(*shape, proprio_dim, device=self.device)
        self.critic_obs = torch.zeros(*shape, critic_dim, device=self.device)
        self.privileged = torch.zeros(*shape, privileged_dim, device=self.device)
        self.raw_action = torch.zeros(*shape, action_dim, device=self.device)
        self.log_prob = torch.zeros(*shape, device=self.device)
        self.value = torch.zeros(*shape, device=self.device)
        self.reward = torch.zeros(*shape, device=self.device)
        self.done = torch.zeros(*shape, device=self.device)
        self.advantage = torch.zeros(*shape, device=self.device)
        self.returns = torch.zeros(*shape, device=self.device)
        self.pos = 0

    def add(self, proprio, critic_obs, privileged, raw_action,
            log_prob, value, reward, done):
        i = self.pos
        self.proprio[i].copy_(proprio)
        self.critic_obs[i].copy_(critic_obs)
        self.privileged[i].copy_(privileged)
        self.raw_action[i].copy_(raw_action)
        self.log_prob[i].copy_(log_prob)
        self.value[i].copy_(value)
        self.reward[i].copy_(reward)
        self.done[i].copy_(done)
        self.pos += 1

    def finish(self, last_value, gamma, gae_lambda):
        gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.horizon)):
            next_value = last_value if t == self.horizon - 1 else self.value[t + 1]
            nonterminal = 1.0 - self.done[t]
            delta = self.reward[t] + gamma * next_value * nonterminal - self.value[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            self.advantage[t] = gae
        self.returns.copy_(self.advantage + self.value)

    def flatten(self):
        n = self.horizon * self.num_envs
        return {
            'proprio': self.proprio.reshape(n, -1),
            'critic_obs': self.critic_obs.reshape(n, -1),
            'privileged': self.privileged.reshape(n, -1),
            'raw_action': self.raw_action.reshape(n, -1),
            'old_log_prob': self.log_prob.reshape(n),
            'old_value': self.value.reshape(n),
            'advantage': self.advantage.reshape(n),
            'returns': self.returns.reshape(n),
        }


class PPOTrainer:
    def __init__(self, model, config=None, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config or PPOConfig()
        self.policy_transform_version = 0
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate)

    @torch.no_grad()
    def act(self, proprio, critic_obs, deterministic=False):
        return self.model.act(proprio, critic_obs, deterministic)

    def update(self, buffer):
        data = buffer.flatten()
        adv = data['advantage']
        data['advantage'] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        count = adv.numel()
        cfg = self.config
        totals = dict(policy=0.0, value=0.0, entropy=0.0,
                      estimator=0.0, kl=0.0, batches=0)

        for _ in range(cfg.update_epochs):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                idx = order[start:start + cfg.minibatch_size]
                logp, entropy, value, estimate = self.model.evaluate_actions(
                    data['proprio'][idx], data['critic_obs'][idx],
                    data['raw_action'][idx])
                ratio = (logp - data['old_log_prob'][idx]).exp()
                # 非负的采样 KL 估计；更新过大时结束本轮，避免策略崩溃。
                with torch.no_grad():
                    sampled_kl = (ratio - 1.0 - (logp - data['old_log_prob'][idx])).mean()
                if cfg.target_kl > 0 and sampled_kl.item() > 1.5 * cfg.target_kl:
                    batches = max(totals.pop('batches'), 1)
                    return {k: v / batches for k, v in totals.items()}
                unclipped = ratio * data['advantage'][idx]
                clipped = ratio.clamp(1.0 - cfg.clip_ratio,
                                      1.0 + cfg.clip_ratio) * data['advantage'][idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value - data['returns'][idx]).square().mean()
                estimator_loss = nn.functional.mse_loss(
                    estimate, data['privileged'][idx])
                entropy_mean = entropy.mean()
                loss = (policy_loss + cfg.value_coef * value_loss
                        + cfg.estimator_coef * estimator_loss
                        - cfg.entropy_coef * entropy_mean)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                totals['policy'] += policy_loss.detach().item()
                totals['value'] += value_loss.detach().item()
                totals['entropy'] += entropy_mean.detach().item()
                totals['estimator'] += estimator_loss.detach().item()
                totals['kl'] += sampled_kl.item()
                totals['batches'] += 1

        batches = max(totals.pop('batches'), 1)
        return {k: v / batches for k, v in totals.items()}

    def save(self, path, iteration, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'iteration': int(iteration),
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ppo_config': asdict(self.config),
            'extra': extra or {},
            'bounded_policy': self.model.bounded_policy,
            'policy_transform_version': 2 if self.model.bounded_policy else 0,
        }, path)

    def load(self, path, load_optimizer=True):
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(data['model'])
        self.model.bounded_policy = data.get('bounded_policy', False)
        self.policy_transform_version = data.get('policy_transform_version', 0)
        if load_optimizer and 'optimizer' in data:
            self.optimizer.load_state_dict(data['optimizer'])
            self.config = PPOConfig(**data.get('ppo_config', {}))
        return int(data.get('iteration', 0)), data.get('extra', {})


def to_tensor(array, device='cpu'):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device)
