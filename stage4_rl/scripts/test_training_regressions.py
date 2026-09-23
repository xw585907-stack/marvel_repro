"""覆盖自动复位、超时回报、动作范围及检查点兼容性的回归测试。"""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _blas
import numpy as np
import torch
from stage4_rl.rl import ActorCriticEstimator, PPOConfig, PPOTrainer, RLEnvConfig, RolloutBuffer, VectorClimbRLEnv


class RegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_command_visible_immediately(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=2))
        env.set_command(0.15)
        np.testing.assert_allclose(env.proprio[:, -1], 0.3)
        np.testing.assert_allclose(env.critic_obs[:, 50], 0.3)

    def test_gait_prior_moves_with_command_and_stands(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=3, domain_randomization=False,
            observation_noise=False, standing_probability=0.0, gait_prior=True))
        env.set_command(np.array([.15, -.15, 0.0]))
        for _ in range(100):
            env.step(np.zeros((3, 12)), None)
        x = env.env.p[:, 0]
        self.assertGreater(x[0], .1)
        self.assertLess(x[1], -.1)
        self.assertLess(abs(x[2]), .05)

    def test_contact_estimate_does_not_block_magnet_action(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=1, domain_randomization=False,
            observation_noise=False, standing_probability=0.0, gait_prior=True))
        env.set_iteration(1200)
        env.set_command(0.0)
        env.step(np.zeros((1, 12)), np.zeros((1, env.privileged_dim)))
        self.assertTrue(env.env.magnet_on.all())

    def test_timeout_metrics_before_reset(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=1, episode_seconds=0.02,
            domain_randomization=False, observation_noise=False))
        _, _, _, _, done, info = env.step(np.zeros((1, 12)))
        self.assertTrue(done[0])
        self.assertTrue(info['time_outs'][0])
        self.assertEqual(info['terminal_critic_obs'].shape, (1, 61))
        terminal = info['transition_metrics']['position'].copy()
        env.env.p[:] = 123
        np.testing.assert_array_equal(info['transition_metrics']['position'], terminal)

    def test_failure_not_timeout(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=1, episode_seconds=0.02))
        env.env.p[:, 1] = .4
        env._reward = lambda *args: np.zeros(1, dtype=np.float32)
        _, _, _, reward, done, info = env.step(np.zeros((1, 12)))
        self.assertTrue(done[0])
        self.assertTrue(info['transition_metrics']['physical_failure'][0])
        self.assertFalse(info['time_outs'][0])
        self.assertEqual(reward[0], -2.0)

    def test_physical_penalties_use_documented_2d_scales(self):
        env = VectorClimbRLEnv(RLEnvConfig(num_envs=1, domain_randomization=False,
            observation_noise=False))
        obs = env.env.get_obs()
        obs['joint_tau'] = np.full((1, 8), 2.0)
        obs['qd'] = np.full((1, 8), 3.0)
        env._reward(obs, np.zeros((1, 12)), np.zeros((1, 4)), np.full((1, 8), 4.0))
        self.assertAlmostEqual(env.last_reward_terms['torque_penalty'], .003 * 8 * (2/30)**2)
        self.assertAlmostEqual(env.last_reward_terms['joint_speed_penalty'], .003 * 8 * (3/10)**2)
        self.assertAlmostEqual(env.last_reward_terms['joint_acceleration_penalty'], .003 * 8 * (4/100)**2)

    def test_bootstrap_stops_at_reset(self):
        buf = RolloutBuffer(2, 1, 1, 1, 1, 1)
        buf.reward[:, 0] = torch.tensor([2.0 + 0.99 * 10.0, 100.0])
        buf.value[:, 0] = torch.tensor([3.0, 1.0])
        buf.done[:, 0] = torch.tensor([1.0, 0.0])
        buf.finish(torch.tensor([1.0]), 0.99, 0.95)
        self.assertAlmostEqual(buf.returns[0, 0].item(), 11.9, places=4)

    def test_bounded_roundtrip_and_legacy(self):
        model = ActorCriticEstimator(51, 10, 12)
        model.bounded_policy = True
        model.actor[-1].bias.data.fill_(100)
        action, raw, logp, _, _ = model.act(torch.zeros(2, 51), torch.zeros(2, 61))
        self.assertLessEqual(action.abs().max().item(), 1)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertGreater(raw.abs().max().item(), 1)
        trainer = PPOTrainer(model, PPOConfig(target_kl=0.01))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.pt'
            trainer.save(path, 4)
            other = PPOTrainer(ActorCriticEstimator(51, 10, 12))
            other.load(path)
            self.assertTrue(other.model.bounded_policy)
            self.assertEqual(other.policy_transform_version, 2)
            self.assertEqual(other.config.target_kl, 0.01)
            data = torch.load(path, weights_only=False)
            del data['bounded_policy']
            torch.save(data, path)
            other.load(path)
            self.assertFalse(other.model.bounded_policy)

    def test_squashed_entropy_pushes_saturated_mean_inward(self):
        model = ActorCriticEstimator(51, 10, 12)
        model.bounded_policy = True
        model.actor[-1].bias.data.fill_(3.0)
        _, entropy, _, _ = model.evaluate_actions(
            torch.zeros(64, 51), torch.zeros(64, 61), torch.zeros(64, 12))
        (-entropy.mean()).backward()
        self.assertGreater(model.actor[-1].bias.grad.mean().item(), 0.0)

    def test_kl_guard_stops_before_update(self):
        model = ActorCriticEstimator(51, 10, 12)
        trainer = PPOTrainer(model, PPOConfig(target_kl=0.01))
        buf = RolloutBuffer(1, 2, 51, 61, 10, 12)
        buf.log_prob.fill_(100)
        buf.advantage[:] = torch.tensor([[1.0, -1.0]])
        before = [p.detach().clone() for p in model.parameters()]
        trainer.update(buf)
        for p, old in zip(model.parameters(), before):
            torch.testing.assert_close(p, old)


if __name__ == '__main__':
    unittest.main()
