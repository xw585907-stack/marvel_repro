"""2D 磁吸附四足爬壁环境（numpy 向量化，Gym 风格 API）。

物理方案：
- 躯干：单刚体动力学（论文1 Fig.5B 的 SRB 假设，腿质量忽略）
- 腿：无质量连杆，关节空间 PD + 虚拟惯量积分。前馈力矩 τ_des 经
  "几何变形 → 接触弹簧"链路转化为足端力：力控时把目标落足点
  压入壁面（论文1 的做法），MPC 的期望 GRF 以 τ_des = J^T f_des
  前馈实现，RL 的力矩动作直接叠加（论文2 的 a_joint）。
- 吸附：EPM 气隙力模型 + 落地掷骰（论文2 Eq.1-4）
- 课程：重力矢量旋转 g = (−g·sinθ, −g·cosθ)（论文2 Eq.5-6）。
  地形始终是水平地面，"爬墙"由 θ=90° + 渲染旋转呈现。

控制频率 50Hz（control_dt=20ms），每控制步 100 个物理子步（dt=0.2ms）。

dt 取 0.2ms 而非 1ms 的原因：罚函数接触的刚度 1e6 N/m 与磁吸力在
近表面气隙处的等效刚度 F_max/λ≈1.85e6 N/m 使俯仰模态频率达
~1.1–1.4e3 rad/s，dt=1ms 时 ω·dt≈1.1–1.4，半隐式欧拉在刚度切换
（接触↔气隙）处把 O(dt²) 误差整流成持续能量注入，俯仰每半周期
放大 ~3 倍（能量审计实测 +2.7 J/子步）。dt=0.2ms 后 ω·dt≈0.27，
垂直墙平衡态收敛到磁吸穿透 δ=F_max/k_n≈0.7mm，俯仰稳定。
"""
import numpy as np

from .robot import RobotParams, foot_pos_body, foot_jac, NUM_LEGS, NUM_JOINTS
from .contact import ContactModel
from .magnet import magnet_force, roll_attach
from .ik import two_link_ik

G = 9.81


class ClimbEnv:
    def __init__(self, num_envs=1, dt=2e-4, control_dt=2e-2, seed=0,
                 **overrides):
        self.num_envs = num_envs
        self.dt = dt
        self.control_dt = control_dt
        self.substeps = int(round(control_dt / dt))
        self.rng = np.random.default_rng(seed)

        P = RobotParams()
        self.hip_offsets = P.hip_offsets          # (4,2) 所有环境共享
        self.l1 = P.l1
        self.l2 = P.l2
        self.q_nominal = np.tile(P.q_nominal, (num_envs, NUM_LEGS))  # (N,8)

        # 可随机化的标量参数（(N,) 数组，阶段4 域随机化直接改这些数组）
        self.mass = np.full(num_envs, P.mass)
        self.inertia = np.full(num_envs, P.inertia)
        self.kp = np.full(num_envs, P.kp)
        self.kd = np.full(num_envs, P.kd)
        self.tau_max = np.full(num_envs, P.tau_max)
        # 执行器限幅用峰值（见 RobotParams.tau_peak 注释）
        self.tau_peak = np.full(num_envs, P.tau_peak)
        self.joint_inertia = np.full(num_envs, P.joint_inertia)
        self.mu = np.full(num_envs, 0.185)        # 论文1 无 MRE 实测 μs
        self.f_max = np.full(num_envs, 697.1)     # 论文2 EPM 最大吸力 [N]
        self.gap_lam = np.full(num_envs, 0.3765e-3)  # 气隙衰减系数 λ [m]
        self.mag_tau = np.full(num_envs, 5e-3)    # EPM 磁场建立/消退滞后 [s]
        self.prob_attach = np.ones(num_envs)      # 课程：1.0 → 0.85
        self.gravity_theta = np.zeros(num_envs)   # 课程：0 → 90°
        # 同髋腿对耦合（矢状面投影副本合并）：刚簧 [N·m/rad] + 阻尼。
        # 只绑定"差分误差"（q_a−q_b 跟踪 q_des_a−q_des_b），不绑定
        # 绝对姿态差：小跑步态下耦合对 (0,2)/(1,3) 相位恒相反（一条
        # 钉扎一条摆动），绝对耦合 k=1e4（PD 的 16 倍）会把摆动腿锁
        # 在钉扎腿姿态上（实测摆动足抬不起、钉扎腿 gap 漂移 0.16mm）。
        # 绑定差分误差后：摆动相命令差≠0，腿自由摆动；双支撑命令差
        # =0，仍抑制病态发散模态。k 取与 PD 同量级：摆动跟踪滞后
        # e_anti≈0.02rad 时对钉扎腿的扰动 tc=k·e_anti≈12N·m，钉扎
        # 足漂移 tc/(JJᵀk_n)≈0.07mm（可接受）。
        self.leg_couple_k = np.full(num_envs, 600.0)
        self.leg_couple_d = np.full(num_envs, 30.0)

        for key, val in overrides.items():
            if hasattr(self, key):
                arr = getattr(self, key)
                arr[...] = val
            else:
                raise KeyError(f'未知环境参数: {key}')

        # k_n=5e6（原 1e6）：穿透 140μm（原 700μm）。无质量足端模型里
        # 弹簧储能 0.5·k_n·pen² 在落地/离地时全部倒进躯干（真实机器人
        # 有足端质量+弹性体吸收），1e6 时每次事件踢 Δv≈0.35m/s（实测
        # 墙面步态被打翻）；5e6 时能量 5 倍小。ω=√(4k_n/m)≈1581rad/s
        # 与 0.2ms 子步（ω·dt≈0.32）仍在显式积分稳定域内。
        self.contact_model = ContactModel(num_envs, k_n=5e6, damp_cap=150.0)
        self.reset()

    # ---------------- 状态 ----------------
    def reset(self):
        """复位：足端直接放在静力平衡穿透 δ=(F_mag+mg/4)/k_n≈0.72mm。

        课程 θ>0 时重力没有指向壁面的分量，足端悬空会永远不落地，
        因此直接以平衡穿透放置。取平衡深度而非任意小穿透：k=0 第一步
        即力平衡（F_env=mg/4/腿），避免穿透欠量的瞬态把机器人逐步
        压进深折叠区（q2≈-2.4 处重合腿对的发散模态不稳定）。
        """
        N = self.num_envs
        self.p = np.zeros((N, 2))
        self.v = np.zeros((N, 2))
        self.phi = np.zeros(N)
        self.omega = np.zeros(N)
        self.q = self.q_nominal.copy()
        foot_b, _ = self._fk()
        # 静力平衡穿透：f_spring = f_mag + m·g_n/4，g_n 为重力压向表面
        # 的分量（以压入为正）。θ=0 平地 g_n=+G（体重压向表面，弹簧
        # 多承 mg/4）；θ=90° 墙面 g_n=0（磁铁自平衡）；
        # θ=180° 天花板 g_n=−G（重力拉离表面，弹簧少承 mg/4）。
        g_n = G * np.cos(self.gravity_theta)
        pen = np.ravel((self.f_max + self.mass * g_n / 4.0)
                       / self.contact_model.k_n)
        self.p[:, 1] = -foot_b[:, :, 1].min(axis=1) - pen
        self.qd = np.zeros((N, NUM_JOINTS))
        self.last_joint_tau = np.zeros((N, NUM_JOINTS))
        self.time = np.zeros(N)
        self.terminated = np.zeros(N, dtype=bool)

        self.contact = np.zeros((N, NUM_LEGS), dtype=bool)
        self.prev_contact = np.zeros((N, NUM_LEGS), dtype=bool)
        self.attach_ok = np.zeros((N, NUM_LEGS), dtype=bool)
        self.magnet_on = np.zeros((N, NUM_LEGS), dtype=bool)
        # 复位时足端已按静力平衡穿透放置，磁吸力直接给满（f_max）：
        # 避免首个子步磁力从 0 爬升时弹簧净推躯干产生复位踢
        self.f_mag = np.tile(self.f_max[:, None], (1, NUM_LEGS))
        self.F_env = np.zeros((N, NUM_LEGS, 2))
        self.gap = np.zeros((N, NUM_LEGS))
        self.contact_model.reset()
        # 摩擦锚点初始化为足端位置：足端"出生"在壁面上（微穿透），
        # 若不初始化锚点，切向弹簧初始拉伸 165mm → 33kN 瞬态 → 俯仰振荡发散
        foot_w, _ = self._foot_state()
        self.contact_model.anchor[...] = foot_w

    def reset_indices(self, indices):
        """仅复位指定并行环境，供 PPO 的异步 episode 管理使用。"""
        idx = np.asarray(indices)
        if idx.dtype == bool:
            idx = np.flatnonzero(idx)
        idx = idx.astype(int, copy=False).ravel()
        if idx.size == 0:
            return self.get_obs()

        self.p[idx] = 0.0
        self.v[idx] = 0.0
        self.phi[idx] = 0.0
        self.omega[idx] = 0.0
        self.q[idx] = self.q_nominal[idx]
        self.qd[idx] = 0.0
        self.last_joint_tau[idx] = 0.0
        self.time[idx] = 0.0
        self.terminated[idx] = False

        foot_b, _ = self._fk()
        g_n = G * np.cos(self.gravity_theta[idx])
        pen = ((self.f_max[idx] + self.mass[idx] * g_n / 4.0)
               / self.contact_model.k_n)
        self.p[idx, 1] = -foot_b[idx, :, 1].min(axis=1) - pen

        self.contact[idx] = False
        self.prev_contact[idx] = False
        self.attach_ok[idx] = False
        self.magnet_on[idx] = False
        self.f_mag[idx] = self.f_max[idx, None]
        self.F_env[idx] = 0.0
        self.gap[idx] = 0.0

        foot_w, _ = self._foot_state()
        self.contact_model.anchor[idx] = foot_w[idx]
        return self.get_obs()

    def set_gravity_theta(self, theta):
        """设置重力倾角 [rad]（0=平地，π/2=垂直墙）。"""
        self.gravity_theta[...] = theta

    # ---------------- 运动学 ----------------
    def _fk(self):
        """足端正运动学：返回足端体坐标 (N,4,2) 与旋转矩阵 R (N,2,2)。"""
        foot_b = foot_pos_body(self.q, self.hip_offsets, self.l1, self.l2)
        c, s = np.cos(self.phi), np.sin(self.phi)
        R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -1)
        return foot_b, R

    def _foot_state(self):
        """足端世界坐标 (N,4,2) 与世界速度 (N,4,2)。"""
        foot_b, R = self._fk()
        foot_w = self.p[:, None, :] + np.einsum('nij,nkj->nki', R, foot_b)
        r = foot_w - self.p[:, None, :]
        # 体坐标关节速度 → 世界
        qd_legs = self.qd.reshape(self.num_envs, NUM_LEGS, 2)
        J = foot_jac(self.q, self.l1, self.l2)
        leg_vel = np.einsum('nfok,nfk->nfo', J, qd_legs)
        leg_vel_w = np.einsum('nij,nkj->nki', R, leg_vel)
        # 躯干平动 + 转动贡献。R = [[c,−s],[s,c]] 时 d(R·r)/dt = ω·(−rz, +rx)
        # （之前写成 (ω·rz, −ω·rx)，方向反了——阻尼项变成负阻尼，俯仰发散）
        omega_cross = np.stack([-self.omega[:, None] * r[..., 1],
                                self.omega[:, None] * r[..., 0]], axis=-1)
        foot_v = self.v[:, None, :] + omega_cross + leg_vel_w
        return foot_w, foot_v

    # ---------------- 仿真 ----------------
    def _world_ik(self, target_w, target_v=0.0):
        """按当前躯干位姿，把世界系足端目标逐子步解算成 (关节目标, 关节目标速度)。

        支撑足必须"钉"在世界系目标点上。若只用步首的体坐标 q_des，
        躯干在 20ms 内移动时足端随体漂移，锚点弹簧的指令变形
        (u_t/k_t ≈ 0.1-0.3 mm) 无法建立，期望切向力落空——实测躯干
        每步漂移 ~0.6 mm 就足以把命令变形吞掉（实现率 <10%，机器人
        原地不动）。每子步按当前 p/φ 重新 IK 可让目标跟随躯干运动，
        但 PD 对"随体运动的目标"有速度追赶滞后 e = q̇_des·kd/kp：
        q̇_des ≈ v_body/0.15 ≈ 0.24 rad/s 时滞后 ~2e-3 rad ≈ 300 µm
        （实测 kp=600），又把命令变形抵消了。因此同时给出钉扎所需
        关节速度 v_foot = v + ω×r + R·J·q̇ = target_v 的解
        J·q̇ = Rᵀ(target_v − v − ω×r)，PD 阻尼项改为追踪该速度
        （前馈），稳态滞后精确消除，弹簧拉伸 k_t·(anchor−target) = u_t。
        target_v: 足端世界系目标速度（控制步内斜坡插值时的钉速度）。
        """
        c, s = np.cos(self.phi), np.sin(self.phi)
        d = target_w - self.p[:, None, :]
        rb = np.stack([c[:, None] * d[..., 0] + s[:, None] * d[..., 1],
                       -s[:, None] * d[..., 0] + c[:, None] * d[..., 1]],
                      axis=-1)                       # Rᵀ·d
        rb = rb - self.hip_offsets                    # 相对髋
        q_des = two_link_ik(rb).reshape(self.num_envs, NUM_JOINTS)

        # 目标关节速度：v_foot = target_v → J·q̇ = Rᵀ(target_v − v − ω×r)
        foot_b, R = self._fk()
        r_w = np.einsum('nij,nkj->nki', R, foot_b)    # 足端相对躯干（世界）
        omega_cross = np.stack([-self.omega[:, None] * r_w[..., 1],
                                self.omega[:, None] * r_w[..., 0]], axis=-1)
        w = target_v - self.v[:, None, :] - omega_cross   # 所需足端世界速度
        w_b = np.stack([c[:, None] * w[..., 0] + s[:, None] * w[..., 1],
                        -s[:, None] * w[..., 0] + c[:, None] * w[..., 1]],
                       axis=-1)                       # Rᵀ·w
        J = foot_jac(self.q, self.l1, self.l2)        # (N,4,2,2)
        # 正则化解 J·q̇ = w_b（奇异位型附近也稳定）：法方程 JᵀJ·q̇ = Jᵀw_b。
        # 旧装配 einsum 求和轴写错，算的是 J·Jᵀ 与 J·w_b，解出 J⁻ᵀ·w_b
        # 而非 J⁻¹·w_b——躯干静止时 w=0 无影响；平动/俯仰时速度前馈
        # 给出错误关节速度，阻尼项把关节拖向错速，膝以余速漂移（实测
        # qd_k−qd_des_k≈+0.74rad/s），扭矩失配被速度松弛吸收，位置
        # 偏转建立不起来 → 移动工况 z 力实现率崩到 ~30-50%。numpy 2.5
        # 的 solve 不支持批量，2x2 直接闭式求逆：
        JtJ = np.einsum('nfji,nfjk->nfik', J, J) + 1e-8 * np.eye(2)
        Jtw = np.einsum('nfji,nfj->nfi', J, w_b)
        a, b2 = JtJ[..., 0, 0], JtJ[..., 0, 1]
        c2, d2 = JtJ[..., 1, 0], JtJ[..., 1, 1]
        det = a * d2 - b2 * c2
        qd_legs = np.stack([(d2 * Jtw[..., 0] - b2 * Jtw[..., 1]) / det,
                            (-c2 * Jtw[..., 0] + a * Jtw[..., 1]) / det],
                           axis=-1)
        qd_des = qd_legs.reshape(self.num_envs, NUM_JOINTS)
        return q_des, qd_des

    def step(self, torque_cmd, magnet_cmd, q_des, target_w=None):
        """执行一个控制步。

        torque_cmd: (N,8) 关节前馈力矩 [N·m]（MPC 的 J^T f_des 或 RL 动作）
        magnet_cmd: (N,4) 磁开关命令 [0,1]（≥0.5 视为 ON）
        q_des: (N,8) 关节目标位置（PD 目标；target_w 给定时仅作回退）
        target_w: (N,4,2) 世界系足端目标（可选）。给定时每个物理子步按
            当前躯干位姿重新 IK 并加关节速度前馈（足端世界系钉扎，
            见 _world_ik）；不给定时退回旧行为（固定体坐标 q_des，
            用于静态测试）。
        返回 (obs, info)
        """
        N = self.num_envs
        torque_cmd = np.asarray(torque_cmd, dtype=float).reshape(N, NUM_JOINTS)
        magnet_cmd = np.asarray(magnet_cmd, dtype=float).reshape(N, NUM_LEGS)
        q_des = np.asarray(q_des, dtype=float).reshape(N, NUM_JOINTS)
        if target_w is not None:
            target_w = np.asarray(target_w, dtype=float).reshape(N, NUM_LEGS, 2)

        g_vec = np.stack([-G * np.sin(self.gravity_theta),
                          -G * np.cos(self.gravity_theta)], axis=-1)  # (N,2)

        # 目标在控制步内从当前足端位置线性斜坡过渡到目标点：力命令以
        # 斜坡实现，弹簧力从当前值连续变化到 u（实测逐步钉扎跳变会使
        # 弹簧力尖峰 + 躯干振荡发散）
        if target_w is not None:
            foot_start, _ = self._foot_state()
            target_v = (target_w - foot_start) / self.control_dt
        else:
            target_v = None

        for s_i in range(self.substeps):
            if target_w is not None:
                frac = (s_i + 1) / self.substeps
                tw = foot_start + (target_w - foot_start) * frac
                q_sub, qd_sub = self._world_ik(tw, target_v)
            else:
                q_sub, qd_sub = q_des, None
            self._substep(torque_cmd, magnet_cmd, q_sub, g_vec, qd_sub)

        self.time += self.control_dt
        self.terminated = (np.abs(self.phi) > 1.2) | \
                          (np.abs(self.p[:, 0]) > 5.0) | \
                          (self.p[:, 1] > 1.5)
        return self.get_obs(), self.get_info()

    def _substep(self, torque_cmd, magnet_cmd, q_des, g_vec, qd_des=None):
        N = self.num_envs
        foot_w, foot_v = self._foot_state()

        # 接触与磁吸（地面 z=0，法线 (0,1)）
        gap = foot_w[..., 1]
        normal = np.zeros_like(foot_w)
        normal[..., 1] = 1.0
        contact = gap < 0.0

        landing = contact & ~self.prev_contact
        roll = roll_attach(landing, self.prob_attach, self.rng)
        self.attach_ok = np.where(landing, roll, self.attach_ok)
        self.magnet_on = magnet_cmd >= 0.5
        # 磁力一阶滞后：EPM 开关非瞬时（论文1 的 5ms 提前量正是为此）。
        # 瞬时开关会让磁力阶跃——落地时 0→697N 把无质量足端猛拉撞墙
        # （阻尼脉冲 + 关节饱和 + 钉扎弹开），离地时 697→0 阶跃踢躯干。
        # τ=3ms 与 0.2ms 子步配合把阶跃铺开成平滑斜坡。
        f_mag_tgt = magnet_force(gap, self.attach_ok, self.magnet_on,
                                 self.f_max, self.gap_lam)
        alpha = np.clip(self.dt / self.mag_tau[:, None], 0.0, 1.0)
        self.f_mag = self.f_mag + alpha * (f_mag_tgt - self.f_mag)
        self.F_env = self.contact_model.step(gap, foot_w, foot_v, normal,
                                             self.mu, self.f_mag)

        # 躯干单刚体动力学
        r = foot_w - self.p[:, None, :]
        F_tot = self.F_env.sum(axis=1) + self.mass[:, None] * g_vec
        self.v += F_tot / self.mass[:, None] * self.dt
        self.p += self.v * self.dt
        # 力矩：R = [[c, −s], [s, c]]（绕 y 轴转 −φ 的约定）下，
        # 动力学为 I·φ̈ = Σ(r_x·F_z − r_z·F_x)
        tau_tot = (r[..., 0] * self.F_env[..., 1] -
                   r[..., 1] * self.F_env[..., 0]).sum(axis=1)
        self.omega += tau_tot / self.inertia * self.dt
        self.phi += self.omega * self.dt

        # 关节：PD + 前馈力矩 + 接触反力矩 JᵀF_env，限幅后按虚拟惯量积分。
        # 接触反力矩是无质量腿力平衡的核心（牛顿定律补齐）：足端环境力
        # 通过腿传到关节，关节必须承受负载力矩 JᵀF_env。缺少它时力矩
        # 前馈 τ_ff=−Jᵀu 的静态平衡是 kp(q_des−q)=−τ_ff，足端偏离钉扎
        # 目标 δ=u/k_eff，锚点弹簧实现力 = k_t·δ = (k_t/k_eff)·u ≈ 38×
        # 增益（实测闭环 F_env_x≈38·u_x、z 通道 188× 瞬态增益、躯干
        # 发散）；加上反力矩后平衡为 k_eff(x_des−x)=u−F_env ⇒
        # F_env ≈ u·k/(k+k_eff) ≈ 97.5%·u（论文1 的无质量腿静力
        # F_env=−J⁻ᵀτ 正是这一机制）。
        # qd_des 给定时阻尼项追踪目标速度 kd·(qd_des − qd)——世界系钉扎
        # 的速度前馈，消除 PD 对随体运动目标的速度追赶滞后
        tau_pd = self.kp[:, None] * (q_des - self.q) - self.kd[:, None] * self.qd
        if qd_des is not None:
            tau_pd = tau_pd + self.kd[:, None] * qd_des
        tau = np.clip(torque_cmd + tau_pd,
                      -self.tau_peak[:, None], self.tau_peak[:, None])
        self.last_joint_tau = tau.copy()
        J = foot_jac(self.q, self.l1, self.l2)          # (N,4,2,2) 体坐标
        c, s = np.cos(self.phi), np.sin(self.phi)
        R = np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -1)
        J_w = np.einsum('nij,nfjk->nfik', R, J)         # 世界系雅可比
        tau_load = np.einsum('nfij,nfj->nfi', J_w, self.F_env)  # (N,4,2)
        tau_load = tau_load.reshape(N, NUM_JOINTS)

        # 2D 矢状面投影：同髋两腿（0/2、1/3）是同一自由度的两个副本。
        # 独立仿真的副本之间有一个无物理的病态内部模态（实测折叠位型
        # 下两腿对拉 ±190N，净实现力崩到 66% 并坠落）；用刚簧+阻尼把
        # 副本耦合为同一自由度（模型内力，不限幅）。
        tc = np.zeros((N, NUM_JOINTS))
        if np.any(self.leg_couple_k > 0):
            for a, b in ((0, 2), (1, 3)):
                dq = ((self.q[:, 2 * b:2 * b + 2] - self.q[:, 2 * a:2 * a + 2])
                      - (q_des[:, 2 * b:2 * b + 2] - q_des[:, 2 * a:2 * a + 2]))
                if qd_des is not None:
                    dqd = ((self.qd[:, 2 * b:2 * b + 2] - self.qd[:, 2 * a:2 * a + 2])
                           - (qd_des[:, 2 * b:2 * b + 2] - qd_des[:, 2 * a:2 * a + 2]))
                else:
                    dqd = self.qd[:, 2 * b:2 * b + 2] - self.qd[:, 2 * a:2 * a + 2]
                f = (self.leg_couple_k[:, None] * dq +
                     self.leg_couple_d[:, None] * dqd)
                tc[:, 2 * a:2 * a + 2] += f
                tc[:, 2 * b:2 * b + 2] -= f
        self.qd += (tau + tau_load + tc) / self.joint_inertia[:, None] * self.dt
        self.q += self.qd * self.dt

        self.contact = contact
        self.prev_contact = contact.copy()
        self.gap = gap

    # ---------------- 观测 ----------------
    def get_obs(self):
        """原始状态观测字典（阶段4 再组装论文2 的完整观测向量）。"""
        foot_w, foot_v = self._foot_state()
        # 世界系足端雅可比 J_w = R·J_body：MPC 力矩约束 |J_wᵀ·u|≤τ_max 用
        Jb = foot_jac(self.q, self.l1, self.l2)          # (N,4,2,2)
        _, R = self._fk()
        J_w = np.einsum('nij,nfjk->nfik', R, Jb)
        return {
            'base_pos': self.p,          # (N,2)
            'base_vel': self.v,          # (N,2)
            'base_phi': self.phi,        # (N,)
            'base_omega': self.omega,    # (N,)
            'q': self.q,                 # (N,8)
            'qd': self.qd,               # (N,8)
            'joint_tau': self.last_joint_tau,  # (N,8) 执行器实际力矩
            'foot_pos': foot_w,          # (N,4,2) 世界坐标
            'foot_vel': foot_v,          # (N,4,2)
            'contact': self.contact,     # (N,4)
            'magnet_on': self.magnet_on, # (N,4)
            'attach_ok': self.attach_ok, # (N,4)
            'f_mag': self.f_mag,         # (N,4)
            'F_env': self.F_env,         # (N,4,2)
            'J_w': J_w,                  # (N,4,2,2) 世界系足端雅可比
            'theta': self.gravity_theta, # (N,)
            'time': self.time,           # (N,)
            'terminated': self.terminated,
        }

    def get_info(self):
        return {}

    def snapshot(self):
        """渲染快照（取第 0 个环境的状态）。"""
        obs = self.get_obs()
        return {k: v[0] for k, v in obs.items()}
