"""SRB-MPC（论文1 的模型预测控制）：单刚体线性动力学 + 可容许反力约束 QP。

公式（论文1 Eq.1-10 的 2D 矢状面投影）：
- 动力学：m·p̈ = Σf_i + m·g，I·φ̈ = Σ(r_i,x·f_i,z − r_i,z·f_i,x)（线性化在
  当前足端位置 r_i 附近，求解时按当前状态重建）
- 决策变量 u_i ∈ R² = 足端对壁面的作用力（GRF，u_n 指向壁面为正）
- 滑移锥（Eq.4）：|u_t| ≤ μs·(f_mag + u_n)
- 倾覆锥（Eq.10）：|u_t| ≤ μt·(f_mag + u_n)（取 μ_eff = min(μs, μt)）
- 吸附/关节约束：u_n ∈ [−f_mag, f_n_max]（−f_mag = 完全靠磁吸拉、
  弹簧力归零的脱附极限；f_n_max = 250 N 踝关节力矩上限，论文 Fig.4F）
- 摆动腿：u = 0

QP 求解：把线性动力学代入代价函数凝聚成仅含 U 的二次规划；
cvxopt coneqp 内点法求解（盒+锥全部转线性约束，实测 4 次迭代
/ 4.3ms，与 SLSQP 高精度参考解一致），失败回退自研罚函数牛顿。
历程：SLSQP 44-67 次迭代（80-350ms）带数值噪声；自研半光滑
牛顿 + 二次罚在 Ps 条件数 ~1e6 下冻结于锥墙前（离最优 Δf≈41）；
内点法对病态鲁棒且确定性，已取代两者。
"""
import time

import numpy as np
from scipy import linalg as sla
from cvxopt import matrix, solvers

# 静默 + 迭代上限（实测典型 4-10 次迭代，60 是安全余量）
solvers.options['show_progress'] = False
solvers.options['maxiters'] = 60


class SRB_MPC:
    def __init__(self, mass=8.0, inertia=0.084, horizon=10, dt=0.02,
                 mu_s=0.185, mu_t=0.795, f_mag=697.1, f_n_max=250.0,
                 z_nom=0.15, d_n=1500.0, c_t=20.0,
                 q_diag=(0.0, 3000.0, 100.0, 200.0, 10.0, 5.0),
                 r_weight=1e-6, rd_weight=1e-4, tau_max=30.0):
        self.m = mass
        self.I = inertia
        self.H = horizon
        self.dt = dt
        self.mu = min(mu_s, mu_t)      # 可容许区域 = 两锥交集（阶段1 结论）
        self.f_mag = f_mag
        self.f_n_max = f_n_max
        self.z_nom = z_nom
        # 接触阻尼（与 env 一致）：支撑腿法向阻尼 d_n·v_z、切向粘性 c_t·v_x。
        # 必须建模——d_n=1500 在 v_z=0.2 m/s 时每脚产生 300 N 反力，
        # 不建模时 MPC 命令的下压/上升会被阻尼完全抵消（实测躯干被
        # "顶"到 0.32 m 且下不来）。
        self.d_n = d_n
        self.c_t = c_t
        # Q_z/Q_vx 取值教训：力矩前馈的力实现率 ~97.5-99%（残余 ~2N），
        # 若 z 反馈太弱（Q_z=300 时相位平均斜率仅 ~-16 N/m，被 Q_vz 的
        # 重力补偿项主导），平衡高度 = 0.15 + 缺口/斜率 漂到 ~0.175。
        # Q_z=3000/Q_vx=200 后闭环体高回到 0.156（平地）/0.149（墙）/
        # 0.144（天花板），墙保持 vx_avg 0.019（详见 probe_mpc_phase）。
        self.Q = np.diag(q_diag)       # 状态 [x, z, φ, vx, vz, ω]
        # 力代价是 min-norm 正则（论文1 的力分配正则）：纯双积分器模型
        # 下 z 对持久力偏置 δ 的灵敏度 Σ_k ∂z_k/∂u_0 ≈ 9.6e-3 m/N，
        # R=1e-6 时稳态 z 平衡偏置 e = R·mg/(Q_z·S) ≈ 3 μm，可忽略；
        # 但 R 不能太小——R=1e-8 时零空间（净力矩为零的力再分配，
        # 如腿(0,1)vs(2,3)对角劈裂）在 QP 里几乎零代价，内点法把
        # 劈裂推到盒极限 ±220/250N，深折叠位型下膝力矩超 τ_max=30
        # 饱和，实现率掉到 54%，躯干坠落（实测 k=13 崩溃）。
        # 注意：旧模型（预测里带 375/s 接触阻尼项）的分析结论
        # "R 取 1e-8"已随模型修正失效。
        self.R = r_weight              # 力代价（8 分量共享）
        self.Rd = rd_weight            # 力变化率平滑（实际的力正则来源）
        # Rd 承担真正的正则化：MᵀQM 秩 ≤ 6H < 8H，代价在力分布的
        # 零空间方向（对模型状态无影响的力组合）上完全平坦，仅靠
        # Rbar=1e-8·I 的正则化使牛顿步在这些方向上放大 ~10⁵ 倍、
        # 线搜索被罚墙全拒（实测 40 迭代冻结在离最优 Δf≈41 处）。
        # Δu 罚在稳态无偏（稳态 Δu=0），且步态交替（FR/RL↔RR/FL）
        # 使 DᵀD 覆盖整个零空间；Rd=1e-4 时零空间特征值 ~4e2，
        # 代价敏感方向 ~2.5e3，条件数 ~6，牛顿步恢复良性。
        self.u_prev = None             # 热启动
        self.polish_trace = False      # 调试：打印 polish 每步内部状态
        self.tau_max = tau_max         # 关节力矩极限（|Jᵀu|≤τ_max 约束）
        self.z_target = None           # 参考调速器目标（solve 里由积分偏置更新）
        self.e_int = 0.0               # ∫(z_nom − z)dt 体高积分误差
        self.k_b = 2.0                 # z 积分偏置增益（ω≈4.5 rad/s, ζ≈1.1 过阻尼）
        self.e_int_vx = 0.0            # ∫(v_cmd − vx)dt 切向速度积分误差
        self.k_bv = 1.0                # vx 积分偏置增益
        self.solve_ms = 0.0
        self.n_fallback = 0            # 冷启动重试次数（诊断用）

    # ---------------- 线性化 SRB 离散动力学 ----------------
    def _dynamics(self, foot_pos, p, phi):
        """由当前状态重建 A(6,6), B(6,8), d(6,)，以及 s0(6,)。"""
        r = foot_pos - p[None, :]                       # (4,2) 当前力臂
        C = np.zeros((3, 8))
        C[0, 0::2] = 1.0 / self.m                       # p̈x = Σfx/m
        C[1, 1::2] = 1.0 / self.m                       # p̈z = Σfz/m
        # φ̈ = Σ(r_x·f_z − r_z·f_x)/I
        C[2, 0::2] = -r[:, 1] / self.I                  # f_x 系数 −r_z/I
        C[2, 1::2] = r[:, 0] / self.I                   # f_z 系数 +r_x/I
        A = np.zeros((6, 6))
        A[:3, :3] = np.eye(3)
        A[:3, 3:] = self.dt * np.eye(3)
        A[3:, 3:] = np.eye(3)
        B = np.zeros((6, 8))
        B[3:, :] = self.dt * C
        return A, B, C

    def _build_prediction(self, foot, p, sched, gravity, s0):
        """构建凝聚预测：s_k = P_k·s0 + Σ_j (Π A)·(Γ_u,j·u_j + g_j)。

        模型是纯双积分器（a_x = a_z = b = c = 0），无接触阻尼项：
        env 的法向阻尼 d_n 与切向粘性 c_t 都作用在足端-地面相对速度
        上（contact.py：f = −k·d − d_n·v_n），而支撑足由每子步世界系
        IK 钉在世界目标点（env._world_ik），v_foot ≈ 0 → 阻尼 ≈ 0，
        故模型中不出现。旧模型 a_z = d_n·n_s/m 对应"足端随体运动"的
        错误实现（v_foot = v_body），修复世界系钉扎后必须去掉——
        否则 MPC 预测 z 通道一个控制步内被阻尼抹平（e^{−a·dt} ≈ 3e−7），
        既与 env 不符，也让 z 代价失去作用。
        切向同理：env 的切向接触是锚点弹簧 + 库仑上限（stick 时 v=0
        静摩擦传力，无需滑移）。若把 c_t·v_x 建进模型，静力保持会被
        预测成"必须持续滑移 v_x ≈ 1 m/s 才能传力"，再经 r_z 力臂变成
        巨大俯仰力矩，10 步预测内发散 ~10⁷ 倍。
        精确离散化保留为通用形式：
        E = e^{−a·dt}，I = (1−E)/a，I2 = (dt−I)/a（a=0 时取 dt、dt²/2）。

        返回 (M_u (6H,8H), b0 (6H,))。
        """
        H = self.H
        m, Ii, dt = self.m, self.I, self.dt
        A_k, Gu_k, g_k = [], [], []
        for k in range(H):
            idx = np.where(sched[k])[0]
            n_s = idx.shape[0]
            a_x = a_z = b = c = 0.0
            Ex, Ez = np.exp(-a_x * dt), np.exp(-a_z * dt)
            Ix = dt if a_x == 0.0 else (1.0 - Ex) / a_x
            Iz = dt if a_z == 0.0 else (1.0 - Ez) / a_z
            I2x = dt * dt / 2 if a_x == 0.0 else (dt - Ix) / a_x
            I2z = dt * dt / 2 if a_z == 0.0 else (dt - Iz) / a_z
            # 状态转移（解析积分 v̇=−a·v、ω̇=b·vx+c·vz+f_ω、φ̇=ω）
            Ad = np.eye(6)
            Ad[0, 3] = Ix
            Ad[1, 4] = Iz
            Ad[2, 3] = b * Ix
            Ad[2, 4] = c * Iz
            Ad[2, 5] = dt
            Ad[3, 3] = Ex
            Ad[4, 4] = Ez
            Ad[5, 3] = b * Ix
            Ad[5, 4] = c * Iz
            # 输入响应 Γ_u (6,8)：每腿 [u_x, u_z] 两列
            Gu = np.zeros((6, 8))
            for i in range(4):
                rx, rz = foot[i] - p
                jx, jz = 2 * i, 2 * i + 1
                Gu[0, jx] = I2x / m                              # x
                Gu[3, jx] = Ix / m                               # vx
                Gu[2, jx] = -rz / Ii * dt * dt / 2 + b * I2x / m  # φ
                Gu[5, jx] = -rz / Ii * dt + b * Ix / m            # ω
                Gu[1, jz] = I2z / m                              # z
                Gu[4, jz] = Iz / m                               # vz
                Gu[2, jz] = rx / Ii * dt * dt / 2 + c * I2z / m   # φ
                Gu[5, jz] = rx / Ii * dt + c * Iz / m             # ω
            # 重力响应 g (6,)
            gx, gz = gravity
            g = np.zeros(6)
            g[0] = I2x * gx
            g[1] = I2z * gz
            g[2] = b * I2x * gx + c * I2z * gz
            g[3] = Ix * gx
            g[4] = Iz * gz
            g[5] = b * Ix * gx + c * Iz * gz
            A_k.append(Ad)
            Gu_k.append(Gu)
            g_k.append(g)

        # 凝聚：s_k = P_k·s0 + Σ_j (A_{k-1}···A_{j+1})·(Γ_u,j·u_j + g_j)
        P = np.eye(6)
        P_k = []
        for k in range(H):
            P = A_k[k] @ P
            P_k.append(P)                                  # s0 → s_{k+1}
        M_u = np.zeros((6 * H, 8 * H))
        b0 = np.concatenate([Pk @ s0 for Pk in P_k])       # (6H,)
        for j in range(H):
            G = np.eye(6)
            M_u[j * 6:(j + 1) * 6, j * 8:(j + 1) * 8] = Gu_k[j]
            b0[j * 6:(j + 1) * 6] += g_k[j]
            for k in range(j + 1, H):
                G = A_k[k - 1] @ G
                M_u[k * 6:(k + 1) * 6, j * 8:(j + 1) * 8] = G @ Gu_k[j]
                b0[k * 6:(k + 1) * 6] += G @ g_k[j]
        return M_u, b0

    # ---------------- 求解 ----------------
    def _ref_traj(self, p, v_cmd, z_target=None, vx_target=None):
        """参考轨迹：x 沿 vx_target 匀速，z 沿可达斜坡到 z_target（vz 跟斜坡斜率）。

        为什么不用恒值 z_nom / 从当前 z 重新爬坡的斜坡：恒值参考下任何
        步0上推力都会在预测里制造上行速度、被 Q_vz 重罚，计划最优选择
        "无速度恢复"，步0命令=mg → 闭环钉在偏低不动点；从当前 z 重新
        爬坡的斜坡则被"推迟到步1-2（参考更高处）再推"的等价计划绕开，
        步0 仍是 mg，不动点不变。因此 z_target 由参考调速器（见 solve）
        供给：带记忆、独立于当前 z 演化，身体卡住时误差持续累积，
        QP 被迫在步0 就推。vx_target 同理：切向推动被俯仰耦合通道压制
        （墙面稳态真最优 u_x≈mg，±3N 目标变差 11.3），速度积分偏置
        累积到 QP 愿意推为止。
        """
        if z_target is None:
            z_target = self.z_nom
        if vx_target is None:
            vx_target = v_cmd
        H_ramp = 5
        dz = z_target - p[1]
        vz_ramp = dz / (H_ramp * self.dt)
        S_ref = np.zeros((6 * self.H,))
        for k in range(1, self.H + 1):
            frac = min(k, H_ramp) / H_ramp
            S_ref[(k - 1) * 6:(k - 1) * 6 + 6] = [
                p[0] + vx_target * k * self.dt, p[1] + dz * frac, 0.0,
                vx_target, vz_ramp if k <= H_ramp else 0.0, 0.0]
        return S_ref

    def solve(self, state, contact_schedule, v_cmd, gravity):
        """求解一个控制步的足端力。

        state: dict（env.snapshot()）：base_pos/base_vel/base_phi/base_omega/
               foot_pos
        contact_schedule: (H,4) bool 预测时域支撑相表（来自步态调度器）
        v_cmd: 期望沿面速度 [m/s]（+x = 沿表面向前/向上爬）
        gravity: (2,) 世界系重力加速度 [g_x, g_z]
        返回 u0 (4,2)：当前步足端目标 GRF（世界系，u_n 指向壁面为正）
        """
        p = np.asarray(state['base_pos'], dtype=float)
        v = np.asarray(state['base_vel'], dtype=float)
        phi = float(state['base_phi'])
        omega = float(state['base_omega'])
        foot = np.asarray(state['foot_pos'], dtype=float)      # (4,2)
        sched = np.asarray(contact_schedule, dtype=bool)       # (H,4)

        A, B, _ = self._dynamics(foot, p, phi)
        s0 = np.concatenate([p, [phi], v, [omega]])            # (6,)
        M_u, b0 = self._build_prediction(foot, p, sched, gravity, s0)

        # 参考轨迹（见 _ref_traj 的说明）。
        # 积分偏置调速器：z_target = z_nom + k_b·∫(z_nom−z)dt。身体卡在
        # 偏低处时积分持续累积、目标持续上移（带限幅防饱和），滞后不
        # 会被"每步重新锚定的斜坡"原谅，QP 被迫在步0 就推，破坏
        # "推迟到预测尾部的等价计划"不动点（z 钉在 0.125、命令恒为 mg）。
        # 稳态 z=z_nom 时积分不再增长，z_target 回落 z_nom，与原行为一致。
        if self.z_target is None:
            self.z_target = self.z_nom
        self.e_int += (self.z_nom - p[1]) * self.dt
        self.e_int = float(np.clip(self.e_int, -0.05, 0.05))
        self.z_target = float(np.clip(
            self.z_nom + self.k_b * self.e_int,
            self.z_nom - 0.05, self.z_nom + 0.05))
        # 切向速度积分偏置（见 _ref_traj 说明）
        self.e_int_vx += (v_cmd - v[0]) * self.dt
        self.e_int_vx = float(np.clip(self.e_int_vx, -0.3, 0.3))
        vx_target = v_cmd + self.k_bv * self.e_int_vx
        self.vx_target = vx_target        # 调试：probe 重建 QP 用
        H = self.H
        S_ref = self._ref_traj(p, v_cmd, self.z_target, vx_target)

        Qbar = np.kron(np.eye(H), self.Q)
        Rbar = self.R * np.eye(8 * H)
        # 力变化率平滑：代价 Σ_{k=1}^{H-1} Rd‖u_k−u_{k−1}‖² + Rd‖u_0−u_prev‖²。
        # DᵀD（D = I − I_{k=8}，前向差分）给出内部块：对角 1/2/1、
        # 相邻块交叉 −1；但边界块不完整：
        #   - u_0 的二次项少了一半（‖u_0−u_prev‖² 还贡献一个 Rd·u_0²），
        #     补 E_first；
        #   - 线性项 −2Rd·u_prev 只该落在 u_0 块，不能写成
        #     −2(DᵀRdD)v0——那会把虚假的 +2Rd·v0 泄漏进 u_1 块，
        #     实测把热启动 QP 的最优钉在 u_prev 自身上（静止保持
        #     78.5N 自洽不动点），z 反馈被锚定项抵消、体高钉在
        #     0.1542 回不到 0.15。
        D = np.eye(8 * H) - np.eye(8 * H, k=8)
        E_first = np.zeros((8 * H, 8 * H))
        E_first[:8, :8] = 1.0

        P = 2.0 * (M_u.T @ Qbar @ M_u + Rbar) + 2.0 * self.Rd * (D.T @ D)
        q = 2.0 * M_u.T @ Qbar @ (b0 - S_ref)
        if self.u_prev is not None:
            # ‖u_0−u_prev‖² 的完整二次+线性项（只有热启动才存在；
            # u_prev=None 时 u_0 二次只来自 ‖u_1−u_0‖²，不可加 E_first——
            # 否则会把 u_0 的力挤到步 1-2（实测 80.4→54.4，闭环坠落））
            P += 2.0 * self.Rd * E_first
            q[:8] += -2.0 * self.Rd * self.u_prev[:8]

        # 约束（盒 + 锥）在 _solve_qp 的 _project 里按 (u_t,u_n) 逐腿
        # 2D 投影实现；锥容量内缩 2%（0.98·μ·f_mag），避免最优解恰好
        # 压在锥顶点（u_n = −f_mag, c=0）的退化点上。
        SC = 1000.0
        Ps = SC * SC * P
        qs = SC * q

        def fun(xs):
            return 0.5 * xs @ Ps @ xs + qs @ xs

        x0s = np.zeros(8 * H)
        if self.u_prev is not None:
            x0s[:-8] = self.u_prev[8:]
            x0s[-8:] = self.u_prev[-8:]
        x0s = x0s / SC

        jac = state.get('J_w')                     # (4,2,2) 或 None
        t0 = time.perf_counter()
        x_sol, nit = self._solve_qp(Ps, qs, sched, x0s, SC, jac=jac)
        x_pol = self._polish_active_set(Ps, qs, sched, x_sol, SC,
                                        trace=self.polish_trace, jac=jac)
        if fun(x_pol) < fun(x_sol) - 1e-12:
            x_sol = x_pol
        self.solve_ms = (time.perf_counter() - t0) * 1e3
        self.last_obj = (float(fun(x_sol)), float(fun(x0s)))
        self.last_msg = f'coneqp it={nit}'
        self.last_nit = nit
        self.last_plan = (x_sol * SC).reshape(H, 8)   # 调试用：整条力规划
        self.last_Mu = M_u
        self.last_b0 = b0
        self.u_prev = x_sol * SC
        return (x_sol[:8] * SC).reshape(4, 2)

    # ---------------- QP 求解（cvxopt 内点法 + 罚函数牛顿回退） ----------------
    def _constraints(self, sched, SC, jac=None):
        """组装锥+盒+关节力矩不等式（G_rows,h_rows）与摆动腿等式
        （A_rows,b_rows）。

        G/h 按行返回列表，由 _solve_qp 与 _polish_active_set 共用，
        避免两处组装漂移。

        jac: (4,2,2) 世界系足端雅可比（可选）。命令力由力矩前馈
        τ_ff=−J_wᵀ·u 实现，|τ_ff| 超过 τ_max 即被环境截断、力落空
        （深折叠位型 ∂z/∂q2→0.186 时对角劈裂 ±220/250N 的膝力矩
        实测 34-47 N·m，实现率掉到 54% 导致坠落）。加上 |J_wᵀu|≤τ_max
        后，QP 只能命令关节可实现的力——SRB 模型补上执行器极限。
        """
        stance = sched.ravel()
        mu = self.mu
        G_rows, h_rows, A_rows, b_rows = [], [], [], []
        for blk in range(len(stance)):
            idx = 2 * blk
            if stance[blk]:
                # ±t ≤ μ(n + a)：t − μn ≤ μa 与 −t − μn ≤ μa
                r1 = np.zeros(2 * len(stance)); r1[idx] = 1.0; r1[idx + 1] = -mu
                r2 = np.zeros(2 * len(stance)); r2[idx] = -1.0; r2[idx + 1] = -mu
                G_rows += [r1, r2]
                h_rows += [mu * 0.98 * self.f_mag / SC] * 2
                # 盒子 −f_mag ≤ n ≤ f_n_max
                r3 = np.zeros(2 * len(stance)); r3[idx + 1] = -1.0
                r4 = np.zeros(2 * len(stance)); r4[idx + 1] = 1.0
                G_rows += [r3, r4]
                h_rows += [self.f_mag / SC, self.f_n_max / SC]
                # 关节力矩 |J_wᵀ·u| ≤ τ_max（每腿髋/膝两条，±两个方向）
                if jac is not None:
                    for j in (0, 1):
                        r5 = np.zeros(2 * len(stance))
                        r5[idx] = jac[blk % 4, 0, j]
                        r5[idx + 1] = jac[blk % 4, 1, j]
                        r6 = -r5
                        G_rows += [r5, r6]
                        h_rows += [self.tau_max / SC] * 2
            else:
                for j in (0, 1):
                    row = np.zeros(2 * len(stance))
                    row[idx + j] = 1.0
                    A_rows.append(row)
                    b_rows.append(0.0)
        return G_rows, h_rows, A_rows, b_rows

    def _polish_active_set(self, Ps, qs, sched, x_sol, SC, trace=False,
                           jac=None):
        """coneqp 解的精确 Newton 精修。

        coneqp 内点法在近零曲率方向（MᵀQM 的力再分配零空间，只被
        Rbar=1e-8 弱正则）上提前判停，落点随初值漂移（实测静止保持
        冷解 60.9N vs 热解 78.5N，目标差仅 0.018）。这里从 coneqp 解
        出发做活动集 Newton：固定活动约束为等式精确解 KKT，沿步方向
        在可行域内做整步/截断线搜索。QP 凸，1-3 次迭代即可把活动面
        内的点钉到真最优；任何一步不可行或目标不降即放弃，退回
        coneqp 解（保证不劣化）。
        """
        nv = len(qs)
        G_rows, h_rows, A_rows, b_rows = self._constraints(sched, SC, jac)
        G = np.stack(G_rows)
        h = np.array(h_rows)
        A = np.stack(A_rows) if A_rows else np.zeros((0, nv))
        b = np.array(b_rows) if A_rows else np.zeros(0)
        if len(A):
            # 摆动腿 u=0 等式：先投影到等式面（投影点仍在可行域内）
            x = np.linalg.lstsq(A, b, rcond=None)[0]
        else:
            x = x_sol.copy()

        def fun(xx):
            return 0.5 * xx @ Ps @ xx + qs @ xx

        f_cur = fun(x)
        eps = 1e-9
        for _ in range(5):
            act = np.where(G @ x - h >= -eps)[0]
            W = np.vstack([G[act], A]) if len(act) else A
            ncon = len(W)
            g = Ps @ x + qs
            if ncon == 0:
                dx = np.linalg.solve(Ps, -g)
            else:
                K = np.zeros((nv + ncon, nv + ncon))
                K[:nv, :nv] = Ps
                K[:nv, nv:] = W.T
                K[nv:, :nv] = W
                rhs = np.zeros(nv + ncon)
                rhs[:nv] = -g
                try:
                    dx = np.linalg.solve(K, rhs)[:nv]
                except np.linalg.LinAlgError:
                    return x_sol
            # 沿 dx 找最大可行步长（不越出任何锥/盒面）
            alpha = 1.0
            Gd = G @ dx
            tight = Gd > 1e-14
            if np.any(tight):
                alpha = min(alpha, np.min((h - G @ x)[tight] / Gd[tight]))
            alpha = min(max(alpha, 0.0), 1.0)
            x_new = x + alpha * dx
            f_new = fun(x_new)
            if trace:
                print(f'    [polish] it={_} ncon={ncon} |g|={np.linalg.norm(g):.3e} '
                      f'|dx|={np.linalg.norm(dx):.3e} alpha={alpha:.6f} '
                      f'f_cur={f_cur:.6f} f_new={f_new:.6f} '
                      f'min_slack={(G @ x - h).min():.3e}')
            if f_new >= f_cur - 1e-12:
                if trace:
                    print('    [polish] 目标不降，break')
                break
            x, f_cur = x_new, f_new
        return x

    def _solve_qp(self, Ps, qs, sched, x0s, SC, max_iter=40, tol=1e-9,
                  trace=False, jac=None):
        """cvxopt coneqp 内点法求解缩放 QP：min 0.5·xᵀPs·x + qsᵀx，
        盒+锥约束全部转成线性不等式（锥 |t| ≤ μ(n+a) 逐腿两条
        ±t ≤ μ(n+a)），摆动腿 u=0 为等式。实测 4 次迭代 / 4.3ms、
        与 SLSQP 高精度参考解一致（Δf≈0.1）。

        为何不用自研牛顿求解器：Ps 条件数 ~1e6（MᵀQM 的常时零空间
        方向只被 Rbar=1e-8 正则），罚函数牛顿步在这些方向上放大
        ~1e5 倍、任何 α 都跨进锥墙，线搜索冻结在离最优 Δf≈41 处；
        延拓/LM/精确投影三种变体均未能脱离（详见 git 历史）。
        内点法对病态矩阵天然鲁棒，且解法确定性、4ms 级耗时在
        20ms 控制预算内。失败时回退到罚函数牛顿（_solve_qp_penalty）。
        """
        nv = len(qs)
        # ---- 约束组装（缩放问题，与 _polish_active_set 共用）----
        G_rows, h_rows, A_rows, b_rows = self._constraints(sched, SC, jac)
        try:
            res = solvers.coneqp(
                matrix(Ps, tc='d'), matrix(qs, tc='d'),
                matrix(np.stack(G_rows), tc='d'), matrix(np.array(h_rows), tc='d'),
                {'l': len(G_rows), 'q': [], 's': []},
                matrix(np.stack(A_rows), tc='d') if A_rows else None,
                matrix(np.array(b_rows), tc='d') if A_rows else None,
                initvals={'x': matrix(x0s, tc='d')})
            if res['status'] == 'optimal':
                return np.array(res['x']).ravel(), int(res['iterations'])
        except Exception:
            pass
        return self._solve_qp_penalty(Ps, qs, sched, x0s, SC, max_iter, tol,
                                      trace)

    def _solve_qp_penalty(self, Ps, qs, sched, x0s, SC, max_iter=40,
                          tol=1e-9, trace=False):
        """回退求解器：二次罚锥 + 罚系数延拓 + LM 阻尼牛顿 + 单调回溯。

        - 盒子：摆动腿 u=0、支撑腿 u_n ∈ [−f_mag, f_n_max]，投影就是
          截断（无切面突变）；
        - 摩擦锥 |u_t| ≤ μ·(u_n + 0.98·f_mag)：二次罚 ρ/2·c²。罚函数
          C¹、分段二次（分段常 Hessian）。
        Ps 病态时（条件数 ~1e6）会在锥墙前冻结——仅作 cvxopt 回退。
        """
        stance = sched.ravel()
        mu = self.mu
        a = 0.98 * self.f_mag / SC
        n_lo = -self.f_mag / SC
        n_hi = self.f_n_max / SC

        def clip_box(v):
            v = v.copy()
            t = np.where(stance, v[0::2], 0.0)
            n = np.where(stance, np.clip(v[1::2], n_lo, n_hi), 0.0)
            v[0::2] = t
            v[1::2] = n
            return v

        def cone_viol(v):
            """锥违反 c = μ(n+a) − s·t < 0 的掩码、符号 s、违反量。"""
            t, n = v[0::2], v[1::2]
            s = np.sign(t)
            c = mu * (n + a) - s * t
            return c < 0.0, s, c

        def fpen(v, rho):
            act, _, c = cone_viol(v)
            return 0.5 * v @ Ps @ v + qs @ v + 0.5 * rho * (c[act] @ c[act])

        x = clip_box(x0s)
        best_x = x.copy()
        alpha = 1.0
        lam = 1e-10                    # Levenberg–Marquardt 阻尼
        nit = 0
        # 罚系数延拓：ρ 从小到大逐层升温。单一大 ρ 会使牛顿步在锥边界
        # 上过冲、线搜索爬行；延拓让每一层都从接近解的位置出发。
        # 阻尼 λ：Ps 条件数 ~1e6（MᵀQM 的常时零空间方向只被
        # Rbar=1e-8 正则），纯牛顿步在这些方向上放大 ~1e5 倍，
        # 任意非零 α 都跨进锥墙（实测 α 阈值 ~2e-15，25 次折半
        # 全部被拒、解冻结在离最优 Δf≈41 处）。(H+λI)⁻¹ 把步长
        # 在平坦方向上限幅为 g/λ，收受即降 λ（恢复牛顿二次收敛）、
        # 被拒则升 λ。
        for lvl, rho in enumerate((10.0, 100.0, 1e3, 1e4)):
            best_f = fpen(x, rho)
            for it in range(max_iter // 4):
                nit += 1
                act, sgn, c = cone_viol(x)
                # 罚项梯度与分段 Hessian（仅活动块：∇c=(−s, μ)，
                # H_pen = ρ·∇c∇cᵀ = ρ·[[1, −sμ],[−sμ, μ²]]）
                g_pen = np.zeros_like(x)
                g_pen[0::2] = np.where(act, rho * c * (-sgn), 0.0)
                g_pen[1::2] = np.where(act, rho * c * mu, 0.0)
                H_pen = np.zeros_like(Ps)
                for blk in np.where(act)[0]:
                    b0 = 2 * blk
                    sm = sgn[blk] * mu
                    H_pen[b0, b0] += rho
                    H_pen[b0, b0 + 1] += -rho * sm
                    H_pen[b0 + 1, b0] += -rho * sm
                    H_pen[b0 + 1, b0 + 1] += rho * mu * mu
                H = Ps + H_pen
                g = Ps @ x + qs + g_pen
                # 收敛判据：罚目标的投影梯度（盒子投影）
                d = np.maximum(np.diag(H), 1e-12)
                if np.abs(clip_box(x - g / d) - x).max() < tol:
                    break
                fx = fpen(x, rho)
                step = None
                for _ in range(10):               # λ 自适应（牛顿→梯度插值）
                    Hd = H + lam * np.eye(H.shape[0])
                    cho = sla.cho_factor(Hd, lower=True, check_finite=False)
                    step = sla.cho_solve(cho, g)
                    alpha = min(alpha * 2.0, 1.0)
                    for _ in range(25):           # 单调性回溯
                        x_new = clip_box(x - alpha * step)
                        if fpen(x_new, rho) <= fx:
                            lam = max(lam / 3.0, 1e-12)
                            break
                        alpha *= 0.5
                    else:
                        x_new = None
                        lam = min(lam * 10.0, 1e8)
                        continue
                    break
                if x_new is None:
                    x_new = clip_box(x - g / d)   # λ 全拒 → PG 一步
                    if fpen(x_new, rho) > fx:
                        break                     # 已收敛（数值噪声）
                    lam = min(lam * 10.0, 1e8)
                x = x_new
                f_new = fpen(x, rho)
                if f_new < best_f:
                    best_f = f_new
                    best_x = x.copy()
                if trace:
                    cc = cone_viol(x)[2]
                    cpos = cc[cc > 0]
                    print(f'tr rho={rho:5.0f} it={nit:3d} f={f_new:10.4f} '
                          f'PG={np.abs(clip_box(x - g / d) - x).max():9.2e} '
                          f'viol={int((cc < 0).sum())} '
                          f'cmin+={cpos.min() if cpos.size else 0:9.2e} '
                          f'lam={lam:.1e} alpha={alpha:.1e}')
        return best_x, nit
