"""论文2 的 Actor/Critic/状态估计器 MLP。"""
import torch
from torch import nn
from torch.distributions import Normal


def _mlp(in_dim, hidden, out_dim, out_gain=1.0):
    layers = []
    last = in_dim
    for width in hidden:
        linear = nn.Linear(last, width)
        nn.init.orthogonal_(linear.weight, gain=2.0 ** 0.5)
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.ELU()]
        last = width
    output = nn.Linear(last, out_dim)
    nn.init.orthogonal_(output.weight, gain=out_gain)
    nn.init.zeros_(output.bias)
    layers.append(output)
    return nn.Sequential(*layers)


class ActorCriticEstimator(nn.Module):
    """二维适配网络。

    论文使用 12 个关节动作 + 4 个磁铁动作；二维模型缩减为
    8 个关节动作 + 4 个磁铁动作。隐藏层尺寸保持论文设置。
    """

    def __init__(self, proprio_dim, privileged_dim, action_dim=12):
        super().__init__()
        self.proprio_dim = int(proprio_dim)
        self.privileged_dim = int(privileged_dim)
        self.action_dim = int(action_dim)

        self.estimator = _mlp(self.proprio_dim, [256, 128],
                              self.privileged_dim, out_gain=0.1)
        self.actor = _mlp(self.proprio_dim + self.privileged_dim,
                          [256, 128, 64], self.action_dim, out_gain=0.01)
        self.critic = _mlp(self.proprio_dim + self.privileged_dim,
                           [256, 128, 64], 1, out_gain=1.0)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -1.0))

    def estimate(self, proprio):
        return self.estimator(proprio)

    def distribution(self, proprio):
        estimate = self.estimate(proprio)
        mean = self.actor(torch.cat([proprio, estimate], dim=-1))
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std), estimate

    def value(self, critic_obs):
        return self.critic(critic_obs).squeeze(-1)

    @torch.no_grad()
    def act(self, proprio, critic_obs, deterministic=False):
        dist, estimate = self.distribution(proprio)
        raw_action = dist.mean if deterministic else dist.sample()
        action = raw_action.clamp(-1.0, 1.0)
        log_prob = dist.log_prob(raw_action).sum(-1)
        value = self.value(critic_obs)
        return action, raw_action, log_prob, value, estimate

    def evaluate_actions(self, proprio, critic_obs, raw_action):
        dist, estimate = self.distribution(proprio)
        log_prob = dist.log_prob(raw_action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(critic_obs)
        return log_prob, entropy, value, estimate
