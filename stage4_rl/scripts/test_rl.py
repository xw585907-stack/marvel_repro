"""阶段4 PPO 结构、课程、环境接口与一次更新的快速回归测试。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas

import numpy as np
import torch

from stage4_rl.rl import (ActorCriticEstimator, ClimbingCurriculum, PPOConfig,
                PPOTrainer, RLEnvConfig, RolloutBuffer, VectorClimbRLEnv)
from stage4_rl.rl.ppo import to_tensor

PASS, FAIL = [], []


def check(name, condition, detail=''):
    (PASS if condition else FAIL).append(name)
    print(f'  [{"通过" if condition else "失败"}] {name} {detail}')


def test_curriculum():
    print('1. 三阶段课程 Eq. (5)-(7)')
    cur = ClimbingCurriculum()
    s0, s1 = cur.state(0), cur.state(1200)
    s2, s3 = cur.state(21200), cur.state(35000)
    check('Phase 1 平地且关闭物理吸附',
          s0.phase == 1 and s0.theta == 0 and not s0.adhesion_enabled)
    check('Phase 2 从 1200 开始', s1.phase == 2 and s1.prob_attach == 1.0)
    check('21200 次达到 90 度', abs(s2.theta - np.pi / 2) < 1e-12)
    check('35000 次吸附成功率降至 0.85',
          abs(s3.prob_attach - 0.85) < 1e-12)


def test_env_and_update():
    print('2. 向量环境、网络和 PPO 更新')
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = RLEnvConfig(num_envs=4, domain_randomization=True,
                      observation_noise=True, curriculum_scale=0.001, seed=0)
    env = VectorClimbRLEnv(cfg)
    check('观测维度', env.proprio.shape == (4, 51)
          and env.critic_obs.shape == (4, 61))
    check('随机化范围', np.all((env.env.mu >= 0.3) & (env.env.mu <= 0.5))
          and np.all((env.action_delay >= 0.0) & (env.action_delay <= 0.008)))

    model = ActorCriticEstimator(env.proprio_dim, env.privileged_dim,
                                 env.action_dim)
    trainer = PPOTrainer(model, PPOConfig(update_epochs=2, minibatch_size=8))
    horizon = 4
    buf = RolloutBuffer(horizon, 4, env.proprio_dim, env.critic_dim,
                        env.privileged_dim, env.action_dim)
    p, c, priv = env.proprio, env.critic_obs, env.privileged
    for _ in range(horizon):
        p_t, c_t = to_tensor(p), to_tensor(c)
        action, raw, logp, value, estimate = trainer.act(p_t, c_t)
        p2, c2, priv2, reward, done, _ = env.step(
            action.numpy(), estimate.numpy())
        buf.add(p_t, c_t, to_tensor(priv), raw, logp, value,
                to_tensor(reward), to_tensor(done))
        p, c, priv = p2, c2, priv2
    with torch.no_grad():
        last_value = model.value(to_tensor(c))
    buf.finish(last_value, 0.99, 0.95)
    losses = trainer.update(buf)
    check('动作与估计器维度', action.shape == (4, 12)
          and estimate.shape == (4, 10))
    check('奖励和 PPO 损失有限', np.isfinite(reward).all()
          and all(np.isfinite(v) for v in losses.values()), str(losses))

    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / 'smoke.pt'
        trainer.save(path, 3, extra={'test': True})
        iteration, extra = trainer.load(path)
        check('模型保存与加载', path.exists() and iteration == 3
              and extra.get('test') is True)

    # 验证局部复位不会改动其余环境。
    before = env.env.p[1:].copy()
    env.env.p[0] = [9.0, 9.0]
    env._reset_indices([0])
    check('并行环境局部复位', env.env.p[0, 0] == 0.0
          and np.allclose(env.env.p[1:], before))


def main():
    test_curriculum()
    test_env_and_update()
    print(f'\n结果：{len(PASS)} 通过 / {len(FAIL)} 失败')
    if FAIL:
        print('失败项：', *FAIL, sep='\n  - ')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
