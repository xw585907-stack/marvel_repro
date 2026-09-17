"""MARVEL 复现项目的共享物理参数。

数据来源：
- 论文1（MARVEL, Science Robotics 2022, eadd1017）：Eq.1-10、Fig.3B/Fig.4、table S1
- 论文2（arXiv:2510.20174）：Fig.3、II-B 节
"""
import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
import matplotlib.pyplot as plt

# ---------- 磁脚实测吸持力（论文1 table S1，五次试验平均） ----------
F_MAG_NO_MRE = 697.1    # 无 MRE 脚垫：法向吸持力 [N]
F_SHEAR_NO_MRE = 129.3  # 无 MRE 脚垫：切向吸持力 [N]
F_MAG_MRE = 535.4       # 带 0.25mm MRE 脚垫：法向吸持力 [N]
F_SHEAR_MRE = 444.6     # 带 0.25mm MRE 脚垫：切向吸持力 [N]

# ---------- 摩擦系数 μs = F_shear / F_mag（库仑摩擦模型） ----------
MU_S_NO_MRE = F_SHEAR_NO_MRE / F_MAG_NO_MRE  # ≈ 0.185（钢-钢滑动面）
MU_S_MRE = F_SHEAR_MRE / F_MAG_MRE            # ≈ 0.830（MRE 高摩擦脚垫）

# ---------- 踝关节与磁脚几何（论文1 Fig.3B） ----------
ANKLE_HEIGHT = 13e-3    # 踝关节距接触面高度 h [m]
FOOT_LENGTH = 62e-3     # 磁脚接触面边长 l [m]（方形 S-EPM 结构；论文图中有标注，
                        # 文字未给出精确值，此值可调整，仅影响 μt 数值）
MU_T = FOOT_LENGTH / (6.0 * ANKLE_HEIGHT)     # 倾覆系数 μt = l/(6h)（论文1 Eq.10）

# ---------- 控制约束（论文1 Fig.4F 的水平截断线） ----------
F_ANKLE_NORMAL_MAX = 250.0  # 标称腿构型下踝部法向反力上限 [N]（关节力矩限制）

# ---------- EPM 气隙-力模型（论文2 Fig.3） ----------
EPM_F_MAX = 697.1   # 零气隙最大法向吸持力 [N]（论文2 使用无 MRE 磁脚）
# 指数衰减系数 λ [mm]：由"1mm 气隙时吸力降至约 7%"标定
# F(1) = F_max·exp(−1/λ) = 0.07·F_max  →  λ = 1/ln(1/0.07) ≈ 0.3765 mm
EPM_GAP_LAMBDA = 1.0 / np.log(1.0 / 0.07)  # ≈ 0.3765 [mm]


def mpl_style():
    """matplotlib 中文显示设置（Windows）。"""
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 110
