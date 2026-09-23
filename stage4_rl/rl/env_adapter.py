"""将自研 2D 物理核心包装为论文2所需的向量化 PPO 环境。"""
from dataclasses import dataclass

import numpy as np

from envs.env import ClimbEnv
from envs.ik import two_link_ik
from envs.robot import RobotParams
from .curriculum import ClimbingCurriculum


@dataclass
class RLEnvConfig:
    num_envs: int = 64
    episode_seconds: float = 10.0
    joint_action_scale: float = 0.45
    obs_filter_alpha: float = 0.35
    gait_period: float = 1.2
    command_min: float = -0.5
    command_max: float = 0.5
    standing_probability: float = 0.1
    gait_prior: bool = False
    residual_action_scale: float = 0.25
    domain_randomization: bool = True
    observation_noise: bool = True
    curriculum_scale: float = 1.0
    seed: int = 0


class VectorClimbRLEnv:
    """论文2训练接口的二维适配版。

    原论文动作是 12 关节 + 4 磁铁；本项目的矢状面模型使用
    8 关节 + 4 磁铁。观测结构、8维时钟、低通滤波、课程及随机化保留。
    """

    proprio_dim = 51
    privileged_dim = 10
    critic_dim = proprio_dim + privileged_dim
    action_dim = 12

    def __init__(self, config=None):
        self.cfg = config or RLEnvConfig()
        self.num_envs = self.cfg.num_envs
        self.rng = np.random.default_rng(self.cfg.seed)
        self.env = ClimbEnv(num_envs=self.num_envs, seed=self.cfg.seed)
        self.curriculum = ClimbingCurriculum(self.cfg.curriculum_scale)
        self.curriculum_state = self.curriculum.state(0)
        self.max_episode_steps = int(round(
            self.cfg.episode_seconds / self.env.control_dt))

        self.base_kp = float(RobotParams.kp)
        self.base_kd = float(RobotParams.kd)
        self.q_nominal = self.env.q_nominal.copy()
        self.prev_targets = np.repeat(
            self.q_nominal[:, None, :], 2, axis=1)
        self.prev_action = np.zeros((self.num_envs, self.action_dim))
        self.prev_prev_action = np.zeros_like(self.prev_action)
        self.prev_exec_action = np.zeros_like(self.prev_action)
        self.prev_qd = np.zeros((self.num_envs, 8))
        self.action_delay = np.zeros(self.num_envs)
        self.orientation_bias = np.zeros(self.num_envs)
        self.command = np.zeros(self.num_envs)
        self.episode_steps = np.zeros(self.num_envs, dtype=np.int64)
        self.episode_return = np.zeros(self.num_envs)
        self.filtered_proprio = np.zeros((self.num_envs, self.proprio_dim))
        self._filter_ready = np.zeros(self.num_envs, dtype=bool)
        self.last_reward_terms = {}

        self.set_iteration(0)
        self._reset_indices(np.arange(self.num_envs))
        self.proprio, self.critic_obs, self.privileged = self._observe(
            np.ones(self.num_envs, dtype=bool))

    def set_iteration(self, iteration):
        self.curriculum_state = self.curriculum.state(iteration)
        self.env.set_gravity_theta(self.curriculum_state.theta)
        self.env.prob_attach[...] = self.curriculum_state.prob_attach

    def set_command(self, command):
        self.command[...] = np.asarray(command, dtype=float)
        # 指令不经过传感器滤波；在下一次策略调用前立即更新缓存观测。
        self.filtered_proprio[:, -1] = self.command / 0.5
        self.proprio[:, -1] = self.command / 0.5
        self.critic_obs[:, self.proprio_dim - 1] = self.command / 0.5

    def _sample_commands(self, idx):
        n = len(idx)
        cmd = self.rng.uniform(self.cfg.command_min,
                               self.cfg.command_max, size=n)
        stand = self.rng.random(n) < self.cfg.standing_probability
        cmd[stand] = 0.0
        self.command[idx] = cmd

    def _gait_prior_action(self):
        phase = np.mod(self.env.time[:, None] / self.cfg.gait_period
                       + 0.25 * np.arange(4)[None, :], 1.0)
        swing = phase < 0.25
        progress = np.where(swing, phase / 0.25, (phase - 0.25) / 0.75)
        stride = np.abs(self.command[:, None]) * self.cfg.gait_period * 0.75
        x = np.where(swing, -stride / 2 + stride * progress,
                     stride / 2 - stride * progress) * np.sign(self.command[:, None])
        z = -0.15 + np.where(swing, 0.08 * np.sin(np.pi * progress), 0.0)
        q_des = two_link_ik(np.stack([x, z], axis=-1)).reshape(self.num_envs, 8)
        joints = np.clip((q_des - self.q_nominal) / self.cfg.joint_action_scale,
                         -1.0, 1.0)
        magnets = np.where(swing, -0.8, 0.8)
        standing = np.abs(self.command) < 0.03
        joints[standing] = 0.0
        magnets[standing] = 0.8
        return np.concatenate([joints, magnets], axis=1)

    def _randomize(self, idx):
        n = len(idx)
        if self.cfg.domain_randomization:
            # 论文给出的 Kp/Kd 区间换算为相对标称值的 0.8~1.2 倍。
            self.env.kp[idx] = self.base_kp * self.rng.uniform(0.8, 1.2, n)
            self.env.kd[idx] = self.base_kd * self.rng.uniform(0.8, 1.2, n)
            self.env.mu[idx] = self.rng.uniform(0.3, 0.5, n)
            self.action_delay[idx] = self.rng.uniform(0.0, 0.008, n)
            self.orientation_bias[idx] = self.rng.uniform(-0.05, 0.05, n)
        else:
            self.env.kp[idx] = self.base_kp
            self.env.kd[idx] = self.base_kd
            self.env.mu[idx] = 0.4
            self.action_delay[idx] = 0.0
            self.orientation_bias[idx] = 0.0

    def _reset_indices(self, indices):
        idx = np.asarray(indices, dtype=int).ravel()
        if idx.size == 0:
            return
        self._randomize(idx)
        self._sample_commands(idx)
        self.env.reset_indices(idx)
        self.prev_targets[idx, 0] = self.q_nominal[idx]
        self.prev_targets[idx, 1] = self.q_nominal[idx]
        self.prev_action[idx] = 0.0
        self.prev_prev_action[idx] = 0.0
        self.prev_exec_action[idx] = 0.0
        self.prev_qd[idx] = 0.0
        self.episode_steps[idx] = 0
        self.episode_return[idx] = 0.0
        self._filter_ready[idx] = False

    def reset(self):
        self._reset_indices(np.arange(self.num_envs))
        self.proprio, self.critic_obs, self.privileged = self._observe(
            np.ones(self.num_envs, dtype=bool))
        return self.proprio, self.critic_obs, self.privileged

    def _clock(self):
        phase = (2.0 * np.pi * self.env.time[:, None] / self.cfg.gait_period
                 + 0.5 * np.pi * np.arange(4)[None, :])
        return np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(
            self.num_envs, 8)

    def _phase_masks(self):
        phase = np.mod(
            2.0 * np.pi * self.env.time[:, None] / self.cfg.gait_period
            + 0.5 * np.pi * np.arange(4)[None, :], 2.0 * np.pi)
        swing = (phase > 0.0) & (phase < 0.5 * np.pi)
        return swing, ~swing

    def _foot_relative_body(self, obs):
        d = obs['foot_pos'] - obs['base_pos'][:, None, :]
        c, s = np.cos(obs['base_phi'])[:, None], np.sin(obs['base_phi'])[:, None]
        return np.stack([c * d[..., 0] + s * d[..., 1],
                         -s * d[..., 0] + c * d[..., 1]], axis=-1)

    def _proprio_features(self, obs):
        q = obs['q'].copy()
        qd = obs['qd'].copy()
        hist = self.prev_targets.copy()
        phi = obs['base_phi'].copy()
        omega = obs['base_omega'].copy()
        foot = self._foot_relative_body(obs)

        if self.cfg.observation_noise:
            q += self.rng.uniform(-0.1, 0.1, q.shape)
            qd += self.rng.uniform(-0.5, 0.5, qd.shape)
            hist += self.rng.uniform(-0.1, 0.1, hist.shape)
            phi += self.orientation_bias + self.rng.uniform(
                -0.05, 0.05, phi.shape)
            omega += self.rng.uniform(-0.1, 0.1, omega.shape)
            foot += self.rng.uniform(-0.015, 0.015, foot.shape)

        features = np.concatenate([
            q - self.q_nominal,
            qd / 10.0,
            (hist[:, 0] - self.q_nominal),
            (hist[:, 1] - self.q_nominal),
            phi[:, None],
            (omega / 5.0)[:, None],
            (foot / 0.3).reshape(self.num_envs, 8),
            self._clock(),
            (self.command / 0.5)[:, None],
        ], axis=1)
        if features.shape[1] != self.proprio_dim:
            raise RuntimeError(f'proprio 维度错误: {features.shape[1]}')
        return features.astype(np.float32)

    def _privileged_features(self, obs):
        return np.concatenate([
            obs['base_vel'] / 1.0,
            obs['foot_pos'][..., 1] / 0.1,
            obs['contact'].astype(float),
        ], axis=1).astype(np.float32)

    def _observe(self, reset_mask=None):
        obs = self.env.get_obs()
        raw = self._proprio_features(obs)
        alpha = self.cfg.obs_filter_alpha
        update = self._filter_ready
        self.filtered_proprio[update] = (
            (1.0 - alpha) * self.filtered_proprio[update]
            + alpha * raw[update])
        fresh = ~self._filter_ready
        self.filtered_proprio[fresh] = raw[fresh]
        self._filter_ready[:] = True
        if reset_mask is not None:
            reset_mask = np.asarray(reset_mask, dtype=bool)
            self.filtered_proprio[reset_mask] = raw[reset_mask]

        privileged = self._privileged_features(obs)
        critic = np.concatenate([self.filtered_proprio, privileged], axis=1)
        return (self.filtered_proprio.copy(), critic.astype(np.float32),
                privileged)

    def _reward(self, obs, action, magnet_action, qdd):
        state = self.curriculum_state
        vel_scale = 1.5 - 0.5 * state.kappa
        penalty_scale = 0.5 + 0.5 * state.kappa
        swing, stance = self._phase_masks()
        contact = obs['contact']

        r_lv = vel_scale * 3.0 * np.exp(
            -5.0 * (self.command - obs['base_vel'][:, 0]) ** 2)
        r_av = vel_scale * 3.0 * np.exp(-5.0 * obs['base_omega'] ** 2)
        gait_ok = np.where(swing, ~contact, contact)
        r_gait = 0.5 * np.where(gait_ok, 1.0, -1.0).sum(axis=1)
        desired_h = np.where(swing, 0.08, 0.0)
        r_foot_h = 0.5 * np.exp(
            -np.sum(swing * (desired_h - obs['foot_pos'][..., 1]) ** 2,
                    axis=1))

        standing = np.abs(self.command) < 0.03
        r_stand = 0.5 * np.where(
            standing, np.where(contact, 1.0, -1.0).sum(axis=1), 0.0)
        positive = r_lv + r_av + r_gait + r_foot_h + r_stand

        foot_v = obs['foot_vel']
        p_slip = penalty_scale * 0.5 * np.sum(
            contact * foot_v[..., 0] ** 2, axis=1)
        p_clear = 140.0 * np.sum(
            (~contact) * (desired_h - obs['foot_pos'][..., 1]) ** 2
            * np.sqrt(np.abs(foot_v[..., 1]) + 1e-6), axis=1)
        p_orient = 3.0 * np.abs(obs['base_phi'])
        # 二维虚拟关节量纲映射：各项先按持续力矩、典型速度和典型加速度归一化。
        p_tau = penalty_scale * 0.003 * np.sum(
            (obs['joint_tau'] / 30.0) ** 2, axis=1)
        alpha_jp = np.where(standing, 3.0, 0.75)
        p_joint = alpha_jp * np.sum(
            (obs['q'] - self.q_nominal) ** 2, axis=1)
        p_joint_speed = 0.003 * np.sum((obs['qd'] / 10.0) ** 2, axis=1)
        p_joint_acc = 0.003 * np.sum((qdd / 100.0) ** 2, axis=1)

        if state.phase == 1 and state.paper_iteration < 1000.0:
            p_smooth1 = np.zeros(self.num_envs)
            p_smooth2 = np.zeros(self.num_envs)
        else:
            p_smooth1 = 2.5 * np.sum((action - self.prev_action) ** 2, axis=1)
            p_smooth2 = 1.2 * np.sum(
                (action - 2.0 * self.prev_action
                 + self.prev_prev_action) ** 2, axis=1)
        p_base = (3.0 * (1.0 - np.exp(-0.5 * obs['base_omega'] ** 2))
                  + 0.2 * np.abs(obs['base_vel'][:, 1]))
        p_magnet = 0.15 * np.sum(
            (contact.astype(float) - magnet_action) ** 2, axis=1)

        penalty = (p_slip + p_clear + p_orient + p_tau + p_joint
                   + p_joint_speed + p_joint_acc + p_smooth1 + p_smooth2
                   + p_base + p_magnet)
        reward = positive * np.exp(-0.2 * np.clip(penalty, 0.0, 80.0))
        self.last_reward_terms = {
            'total': float(np.mean(reward)),
            'velocity': float(np.mean(r_lv)),
            'gait': float(np.mean(r_gait)),
            'penalty': float(np.mean(penalty)),
            'torque_penalty': float(np.mean(p_tau)),
            'joint_speed_penalty': float(np.mean(p_joint_speed)),
            'joint_acceleration_penalty': float(np.mean(p_joint_acc)),
            'action_smoothness_penalty': float(np.mean(p_smooth1 + p_smooth2)),
            'magnet': float(np.mean(p_magnet)),
        }
        return reward.astype(np.float32)

    def step(self, action, contact_estimate=None):
        policy_action = np.asarray(action, dtype=float).reshape(
            self.num_envs, self.action_dim)
        policy_action = np.clip(policy_action, -1.0, 1.0)
        action = (np.clip(self._gait_prior_action()
                          + self.cfg.residual_action_scale * policy_action, -1.0, 1.0)
                  if self.cfg.gait_prior else policy_action)
        delay = (self.action_delay / self.env.control_dt)[:, None]
        executed = (1.0 - delay) * action + delay * self.prev_exec_action
        joint_action = executed[:, :8]
        magnet_action = 0.5 * (executed[:, 8:] + 1.0)

        q_des = self.q_nominal + self.cfg.joint_action_scale * joint_action
        # 磁铁是策略直接控制的执行器。接触估计器只补充本体感知；若用它
        # 阻断磁铁命令，会形成“尚未估计接触 -> 无法吸附 -> 永远无接触”的
        # 自锁，尤其会让墙面复位立即坠落。
        magnet_cmd = (magnet_action >= 0.5).astype(float)
        if not self.curriculum_state.adhesion_enabled:
            magnet_cmd[:] = 0.0

        obs, _ = self.env.step(
            np.zeros((self.num_envs, 8)), magnet_cmd, q_des)
        self.episode_steps += 1
        qdd = (obs['qd'] - self.prev_qd) / self.env.control_dt
        reward = self._reward(obs, action, magnet_action, qdd)

        physical_failure = (obs['terminated'].copy()
                            | (obs['base_pos'][:, 1] < 0.03)
                            | (obs['base_pos'][:, 1] > 0.35))
        done = physical_failure | (self.episode_steps >= self.max_episode_steps)
        reward = reward - 2.0 * physical_failure.astype(np.float32)
        self.episode_return += reward
        completed_returns = self.episode_return[done].copy()
        completed_lengths = self.episode_steps[done].copy()

        self.prev_targets[:, 1] = self.prev_targets[:, 0]
        self.prev_targets[:, 0] = q_des
        self.prev_prev_action[:] = self.prev_action
        self.prev_action[:] = action
        self.prev_exec_action[:] = action
        self.prev_qd[:] = obs['qd']

        # 必须在自动复位前保存本次转移指标，避免评估读取复位后的零速度/满磁力。
        transition_metrics = {
            'velocity': obs['base_vel'][:, 0].copy(),
            'stance': self._phase_masks()[1].copy(),
            'magnetic_force': obs['f_mag'].copy(),
            'position': obs['base_pos'].copy(),
            'physical_failure': physical_failure.copy(),
        }
        time_outs = (self.episode_steps >= self.max_episode_steps) & ~transition_metrics['physical_failure']
        terminal_critic_obs = None
        if np.any(time_outs):
            raw = self._proprio_features(obs)
            filtered = ((1.0 - self.cfg.obs_filter_alpha) * self.filtered_proprio
                        + self.cfg.obs_filter_alpha * raw)
            terminal_critic_obs = np.concatenate(
                [filtered, self._privileged_features(obs)], axis=1).astype(np.float32)
        reset_mask = done.copy()
        if np.any(done):
            self._reset_indices(np.flatnonzero(done))
        self.proprio, self.critic_obs, self.privileged = self._observe(reset_mask)
        info = {
            'episode_returns': completed_returns,
            'episode_lengths': completed_lengths,
            'reward_terms': self.last_reward_terms.copy(),
            'curriculum': self.curriculum_state,
            'transition_metrics': transition_metrics,
            'time_outs': time_outs,
            'terminal_critic_obs': terminal_critic_obs,
        }
        return (self.proprio, self.critic_obs, self.privileged,
                reward, done.astype(np.float32), info)
