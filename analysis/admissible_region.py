"""复现论文1 Fig.4F：可容许踝部反力区域（防滑移摩擦锥 ∩ 防倾覆锥）。

论文1 Eq.1-10 的推导（2D 矢状面投影）：
- 滑移失效（Eq.4，摩擦锥）：  f_An ≤ f_mag，|f_At| ≤ μs·(f_mag − f_An)
- 倾覆失效（Eq.10，倾覆锥）：f_An ≤ f_mag，|f_At| ≤ μt·(f_mag − f_An)，μt = l/(6h)
- 可容许区域 = 两锥交集 ∩ f_An ≤ 250 N（关节力矩上限，论文 Fig.4F 的水平线）

其中 f_mag 为磁吸力，f_An/f_At 为踝部反力的法向/切向分量。
若 μt < μs，越界时先发生倾覆；若 μs < μt，越界时先发生滑移（论文 Discussion 与 Fig.4F 说明）。

运行：python analysis/admissible_region.py
输出：analysis/figures/fig4f_admissible_region.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根(import _blas 用)

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
import matplotlib.pyplot as plt

from common import (
    F_MAG_MRE, F_MAG_NO_MRE, MU_S_MRE, MU_S_NO_MRE, MU_T,
    F_SHEAR_MRE, F_SHEAR_NO_MRE, F_ANKLE_NORMAL_MAX, mpl_style,
)

FIG_DIR = Path(__file__).resolve().parent / 'figures'


def cone_boundaries(f_mag, mu, n=300):
    """锥体边界线：y = 法向反力 f_An ∈ [0, f_mag]，x = ±mu·(f_mag − y)。"""
    y = np.linspace(0.0, f_mag, n)
    return y, mu * (f_mag - y), -mu * (f_mag - y)


def admissible_region(f_mag, mu_lim, n=300):
    """可容许区域：|f_At| ≤ mu_lim·(f_mag − f_An)，且 f_An ≤ 250 N。"""
    y = np.linspace(0.0, min(f_mag, F_ANKLE_NORMAL_MAX), n)
    x = mu_lim * (f_mag - y)
    return y, x


def plot_case(ax, f_mag, mu_s, color, name):
    """画一个磁脚构型的双锥与可容许区域，返回主导失效模式说明。"""
    y, x_s_pos, x_s_neg = cone_boundaries(f_mag, mu_s)
    _, x_t_pos, x_t_neg = cone_boundaries(f_mag, MU_T)

    # 摩擦锥（虚线，防滑移）与倾覆锥（实线，防倾覆）
    ax.plot(x_s_pos, y, ls='--', c=color, lw=1.6)
    ax.plot(x_s_neg, y, ls='--', c=color, lw=1.6,
            label=f'{name}：摩擦锥 μs={mu_s:.3f}')
    ax.plot(x_t_pos, y, ls='-', c=color, lw=1.6)
    ax.plot(x_t_neg, y, ls='-', c=color, lw=1.6,
            label=f'{name}：倾覆锥 μt={MU_T:.3f}')

    # 可容许区域 = 较陡一支锥体下方的区域（两锥交集）
    mu_lim = min(mu_s, MU_T)
    y_r, x_r = admissible_region(f_mag, mu_lim)
    ax.fill_betweenx(y_r, -x_r, x_r, color=color, alpha=0.22,
                     label=f'{name}：可容许区域')

    mode = '倾覆主导（μt < μs）' if MU_T < mu_s else '滑移主导（μs < μt）'
    print(f'[{name}] μs={mu_s:.3f}, μt={MU_T:.3f} → {mode}，'
          f'锥顶点 f_mag={f_mag:.1f} N')


def main():
    mpl_style()
    fig, ax = plt.subplots(figsize=(7.5, 6.0))

    # 有 MRE 脚垫（论文图例：蓝色区域更大）
    plot_case(ax, F_MAG_MRE, MU_S_MRE, color='tab:blue', name='带 MRE 脚垫')
    # 无 MRE 脚垫（红色区域）
    plot_case(ax, F_MAG_NO_MRE, MU_S_NO_MRE, color='tab:red', name='无 MRE 脚垫')

    # 关节力矩上限：f_An ≤ 250 N（论文 Fig.4F 的水平线）
    ax.axhline(F_ANKLE_NORMAL_MAX, c='k', ls=':', lw=1.6,
               label=f'踝部法向力上限 {F_ANKLE_NORMAL_MAX:.0f} N')

    # 实测吸持力点（切向极限载荷在 f_An=0 处测得）
    ax.scatter([F_SHEAR_MRE], [0], c='tab:blue', s=55, zorder=5)
    ax.scatter([F_SHEAR_NO_MRE], [0], c='tab:red', s=55, zorder=5)
    ax.annotate(f'实测 ({F_SHEAR_MRE:.0f} N, 0)', (F_SHEAR_MRE, 0),
                textcoords='offset points', xytext=(8, 14), color='tab:blue')
    ax.annotate(f'实测 ({F_SHEAR_NO_MRE:.0f} N, 0)', (F_SHEAR_NO_MRE, 0),
                textcoords='offset points', xytext=(-10, -24), color='tab:red')

    ax.set_xlabel('切向踝部反力 f_At [N]')
    ax.set_ylabel('法向踝部反力 f_An [N]')
    ax.set_title('可容许踝部反力区域（复现论文1 Fig.4F）')
    ax.set_xlim(-320, 320)
    ax.set_ylim(0, 760)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8.5)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / 'fig4f_admissible_region.png'
    fig.savefig(out)
    print(f'已保存：{out}')


if __name__ == '__main__':
    main()
