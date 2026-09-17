"""接触模型：法向弹簧-阻尼 + 锚点库仑摩擦（静摩擦用切向弹簧近似）。

方案说明（无质量腿 + 惩罚接触）：
- 足端环境力（弹簧/阻尼/摩擦/磁吸）直接施加在躯干上——无质量腿把
  足端力原样传到躯干，与论文1 的 SRB 假设一致；
- 静摩擦用锚点弹簧实现：接触建立时锚点固定在接触点，|F| ≤ μN 内
  足端粘滞（微量弹性变形 ≈0.1mm 量级），超过 μN 后锚点沿力方向
  滑移（动摩擦 μN），实现 stick-slip。
"""
import numpy as np


class ContactModel:
    def __init__(self, num_envs, k_n=1e6, d_n=1500.0, k_t=2e5, c_t=200.0,
                 damp_cap=250.0):
        self.k_n = k_n      # 法向接触刚度 [N/m]
        self.d_n = d_n      # 法向阻尼 [N·s/m]
        self.damp_cap = damp_cap  # 法向阻尼限幅 [N]：落地冲击保护
        self.k_t = k_t      # 切向锚点刚度 [N/m]
        # 切向粘性阻尼 [N·s/m]：锚点弹簧 + 关节 PD 等效质量（≈0.44kg）
        # 构成的振子 ω≈√(k_t/m_eff)≈107Hz，被 50Hz 力矩前馈逐拍踢振，
        # c_t=20 时 ζ≈0.03 振荡几十个周期（实测 F_env 摆 ±300N、闭环
        # 发散）；c_t=200 时 ζ≈0.34，10-20ms 内平息。物理上对应 MRE 橡胶
        # 垫的剪切粘弹性（tanδ~0.3）。粘滞只在粘着期足端相对锚点振动时
        # 作用（v_tan≈0），不产生宏观阻力；旧值 900 在滑移期 0.3m/s 时
        # 每脚 270N 阻力且 MPC 未建模（实测平地跑不动），故取 200。
        self.c_t = c_t
        self.anchor = np.zeros((num_envs, 4, 2))  # 摩擦锚点（世界坐标）

    def reset(self):
        self.anchor[...] = 0.0

    def step(self, gap, foot_pos, foot_vel, normal, mu, magnet_force):
        """计算每只脚上的环境接触力。

        gap: (N,4) 带符号距离（>0 悬空，<0 穿透）
        foot_pos/foot_vel: (N,4,2) 足端世界坐标/速度
        normal: (N,4,2) 表面外法线
        mu: (N,) 摩擦系数
        magnet_force: (N,4) 磁吸力（法向、指向表面，正值）
        返回 force: (N,4,2)
        """
        in_contact = gap < 0.0

        # 法向弹簧-阻尼（仅在穿透时产生：gap<0 才有力，
        # 否则快速接近但未接触的脚会被阻尼项"凭空"弹开，注入能量）。
        # 阻尼限幅：磁铁提前 5ms 开启后，无质量足端被 697N 吸力猛拉
        # 撞墙（v_n 达 1-2m/s），未限幅时 d_n·v_n 产生 1500-3000N 脉冲
        # 把躯干踹离墙面、关节饱和、钉扎弹开（实测墙面步态 0.5s 处
        # 落地腿 gap 弹到 +7.9mm）；限幅把冲击脉冲封在 damp_cap 内，
        # 稳态穿透速度极小不受影响。
        v_n = np.sum(foot_vel * normal, axis=-1)                     # (N,4)
        f_damp = np.clip(-self.d_n * v_n, -self.damp_cap, self.damp_cap)
        f_spring = np.where(in_contact,
                            np.clip(-self.k_n * gap + f_damp, 0.0, None),
                            0.0)

        # 界面法向载荷 = 弹簧力本身（磁吸力已含在弹簧平衡里）：
        # 论文1 Eq.4 的容量 μs(f_mag − f_An)，f_An(踝反力) = f_mag − f_spring
        # → 容量 = μs·f_spring。验证：零踝力时 f_spring = f_mag = 697.1，
        # μs·697.1 = 129.3 N = 论文 table S1 实测 F_shear ✓
        # （旧公式 f_spring + magnet_force 会给出 258 N，与实测不符）
        n_load = f_spring

        # 切向：锚点静摩擦
        d_tan = foot_pos - self.anchor
        d_tan -= np.sum(d_tan * normal, axis=-1, keepdims=True) * normal
        v_tan = foot_vel - np.sum(foot_vel * normal, axis=-1, keepdims=True) * normal

        f_stick = -self.k_t * d_tan                                  # 弹簧力（粘滞）
        f_visc = -self.c_t * v_tan                                   # 粘性阻尼（不封顶）
        f_stick_mag = np.linalg.norm(f_stick, axis=-1)               # (N,4)
        cap = mu[..., None] * n_load                                 # μ·(F_spring+F_mag)

        # 滑移判定只基于弹簧力：阻尼项不参与容量判定。否则 c_t·v_tan 会把
        # 合力顶过 μN 提前触发滑移，锚点重置时弹簧能量被"重置"到 cap²/2k_t
        # 而实际只有 (k_t·d)²/2k_t，每次都凭空注入能量（能量审计实测）。
        slip = f_stick_mag > cap
        safe_mag = np.where(f_stick_mag > 1e-9, f_stick_mag, 1.0)
        f_dir = f_stick / safe_mag[..., None]
        f_fric = np.where(slip[..., None], cap[..., None] * f_dir, f_stick) + f_visc
        # 滑移时把锚点前移，使残余拉伸 = cap/k_t（滑动摩擦）
        self.anchor = self.anchor + np.where(
            slip[..., None], d_tan + (cap / self.k_t)[..., None] * f_dir, 0.0)

        # 悬空：无摩擦，锚点跟随足端（落地时从落点开始粘滞）
        f_fric = np.where(in_contact[..., None], f_fric, 0.0)
        self.anchor = np.where(in_contact[..., None], self.anchor, foot_pos)

        # 法向合力 = 弹簧（推开，+normal）− 磁吸（拉向表面，−normal）
        # 注意：摩擦容量仍用 n_load = f_spring + magnet_force（总法向载荷）
        return f_fric + (f_spring - magnet_force)[..., None] * normal
