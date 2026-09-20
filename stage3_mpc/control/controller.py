"""论文1 完整控制器：步态调度 + Raibert 落足点 + SRB-MPC + 足端力实现。

架构（对照论文1 Fig.5）：
1. 步态调度器给出支撑/摆动相与 EPM 开关命令（提前 5ms）；
2. SRB-MPC 以 50Hz 求解未来 0.2s 的足端目标力 u_i（满足滑移锥、
   倾覆锥与踝关节力矩上限）；
3. 力实现（论文1 的机制）：支撑腿关节前馈力矩 τ_ff = −J_wᵀ·u——
   无质量腿的静力平衡 F_env = −J⁻ᵀτ 把期望 GRF 直接写进接触力。
   位置通道只负责"原地钉扎"：支撑足目标 = 本步起点的足端位置
   （保持受力下沉变形，绝不拉回锚点——拉回会把锚点弹簧的已
   实现力每步清零），逐子步世界系 IK + 速度前馈钉住。
   实测教训：旧方案把 u_t/k_t（~0.1-0.3mm 偏移）写进锚点弹簧变形，
   但关节 PD 的等效刚度 kp/|JJᵀ|≈5.3kN/m 使足端在切向力下下沉
   F·|JJᵀ|/kp，力实现率只有 k_eff/(k_t+k_eff)≈2.6%（躯干被锚点
   弹簧"拖着"不动）；力矩前馈的实现率为 k_t/(k_t+k_eff)≈97.5%，
   残余 2.5% 由 MPC 反馈消除；
4. 摆动腿三次 Bézier 轨迹，落足点按 Raibert 法则在离地时刻计算。

坐标系：u 为足端-壁面接触产生的净体受力（弹簧力 − 磁吸力），
u_n 指向壁面为正——与 env 中"足端力原样传至躯干"的 SRB 假设一致，
也与 MPC 线性动力学 Σu/m 的符号一致。
"""
import numpy as np

from .gait import GaitScheduler
from .mpc import SRB_MPC
from .swing import bezier_swing
from envs.ik import two_link_ik
from envs.robot import foot_jac

G = 9.81


class MPCController:
    def __init__(self, env, v_cmd=0.15, k_raibert=0.08, **mpc_kwargs):
        self.env = env
        self.control_dt = env.control_dt
        self.v_cmd = v_cmd
        self.k_raibert = k_raibert
        # MPC 模型必须与 env 的接触阻尼一致（否则阻尼反力成为未建模干扰）
        mpc_kwargs.setdefault('d_n', env.contact_model.d_n)
        mpc_kwargs.setdefault('c_t', env.contact_model.c_t)
        self.mpc = SRB_MPC(**mpc_kwargs)
        self.gait = GaitScheduler()
        self.hip_offsets = env.hip_offsets          # (4,2) 体坐标
        self.mode = None                            # (4,) bool: True=支撑
        self.p0 = np.zeros((4, 2))                  # 离地点（世界）
        self.p3 = np.zeros((4, 2))                  # 落足点（世界）
        self.info = {}                              # 每步调试信息
        self.reset()

    # ---------------- 复位 ----------------
    def reset(self):
        self.gait.reset(0.0)
        snap = self.env.snapshot()
        stance_now = self.gait.stance(0.0)
        self.mode = stance_now.copy()
        for i in range(4):
            if not stance_now[i]:
                # 起步时处于摆动相的腿：从站立位姿开始第一摆
                self.p0[i] = snap['foot_pos'][i]
                self.p3[i] = self._foothold(snap, i)
        # 热启动：重力补偿均分给当前支撑腿，供 MPC 首个 QP 使用
        theta = float(snap['theta'])
        u0 = np.zeros((4, 2))
        n_st = int(stance_now.sum())
        if n_st > 0:
            g_vec = (-G * np.sin(theta), -G * np.cos(theta))
            u0[stance_now] = (-self.env.mass[0] * g_vec[0] / n_st,
                              -self.env.mass[0] * g_vec[1] / n_st)
        self.mpc.u_prev = np.tile(u0.ravel(), self.mpc.H)

    # ---------------- 辅助 ----------------
    def _foothold(self, snap, i):
        """Raibert 落足点（论文1）：p_land = p_hip + T_st·v/2 + k_v(v−v_des)。"""
        c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
        hip_b = self.hip_offsets[i]
        hip_wx = snap['base_pos'][0] + c * hip_b[0] - s * hip_b[1]
        vx = snap['base_vel'][0]
        T_st = self.gait.period * (1.0 - self.gait.swing_fraction)
        x = hip_wx + 0.5 * T_st * self.v_cmd + self.k_raibert * (vx - self.v_cmd)
        return np.array([x, 0.0])                   # 壁面 z=0，落点压至表面

    # ---------------- 控制步 ----------------
    def step(self):
        """一个控制步（env.step 前调用）。

        返回 (torque_cmd (8,), magnet_cmd (4,), q_des (8,))。
        """
        self.gait.advance(self.control_dt)
        t = self.gait.t
        snap = self.env.snapshot()

        # 1) SRB-MPC 求解足端目标力
        theta = float(snap['theta'])
        g_vec = (-G * np.sin(theta), -G * np.cos(theta))
        sched = self.gait.contact_schedule(t, self.mpc.H, self.mpc.dt)
        u = self.mpc.solve(snap, sched, self.v_cmd, g_vec)     # (4,2)

        # 2) 力实现：支撑腿力矩前馈 τ_ff = −J_wᵀ·u（论文1 机制），
        #    位置通道原地钉扎（目标 = 本步起点足端位置，保持受力下沉）
        ph = self.gait.phase(t)
        stance = ph >= self.gait.swing_fraction
        target_w = np.zeros((4, 2))
        for i in range(4):
            if stance[i]:
                if not self.mode[i]:
                    self.mode[i] = True
                    # 落地事件：钉扎点 z 压到壁面。摆动末端追踪滞后
                    # ~0.1mm 会让钉扎悬空——无弹簧约束时腿对耦合
                    # 可把悬空足拖高 4mm 且永不落地（实测墙面步态
                    # 落地腿 gap 反升到 +4.4mm）。
                    target_w[i] = snap['foot_pos'][i].copy()
                    target_w[i, 1] = 0.0
                else:
                    target_w[i] = snap['foot_pos'][i]           # 原地钉扎
            else:
                if self.mode[i]:                               # 离地事件
                    self.p0[i] = snap['foot_pos'][i]
                    self.p3[i] = self._foothold(snap, i)
                    self.mode[i] = False
                s = ph[i] / self.gait.swing_fraction
                target_w[i] = bezier_swing(s, self.p0[i], self.p3[i])

        # 3) IK：世界目标 → 体坐标（Rᵀ，R=[[c,−s],[s,c]]）→ 关节角
        c, s = np.cos(snap['base_phi']), np.sin(snap['base_phi'])
        d = target_w - snap['base_pos']
        rb = np.stack([c * d[:, 0] + s * d[:, 1],
                       -s * d[:, 0] + c * d[:, 1]], axis=-1)   # Rᵀ·d
        rb = rb - self.hip_offsets                              # 相对髋
        q_des = two_link_ik(rb).ravel()                         # (8,)

        # 4) 力矩前馈：足端力平衡 F_env = −J_w⁻ᵀ·τ_total，令 F_env = u
        #    得 τ_ff = −J_wᵀ·u（负号经静态验证：F_env_x = +0.97·u_x）
        J = foot_jac(snap['q'], self.env.l1, self.env.l2)      # (4,2,2) 体坐标
        R = np.array([[c, -s], [s, c]])
        J_w = np.einsum('ij,fjk->fik', R, J)
        tau_legs = -np.einsum('fij,fj->fi', J_w, u)             # (4,2)
        torque_cmd = np.where(stance[:, None], tau_legs, 0.0).ravel()

        # 5) EPM 命令（5ms 提前量，见 GaitScheduler）
        magnet_cmd = self.gait.magnet_cmd().astype(float)

        self.info = {'u': u, 'target_w': target_w, 'stance': stance,
                     'solve_ms': getattr(self.mpc, 'solve_ms', 0.0)}
        return torque_cmd, magnet_cmd, q_des
