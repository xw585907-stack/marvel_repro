"""复现论文2 Fig.3：EPM 磁吸力-气隙曲线。

论文2 II-B：EPM 必须与钢板形成闭合磁路才能产生吸持力；
磁化时若存在气隙或接触不完整，吸力会急剧下降。
Fig.3 显示 1mm 气隙时吸力降至最大值的约 7%。

模型（指数衰减，气隙-力曲线常用形式）：
    F(gap) = F_max · exp(−gap/λ)，λ ≈ 0.3765 mm（由 7%@1mm 标定）

运行：python analysis/magnetic_force.py
输出：analysis/figures/fig3_magnetic_force_gap.png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根(import _blas 用)

import _blas  # 必须在 numpy 前：OPENBLAS_NUM_THREADS=1（见 _blas.py）
import numpy as np
import matplotlib.pyplot as plt

from common import EPM_F_MAX, EPM_GAP_LAMBDA, mpl_style

FIG_DIR = Path(__file__).resolve().parent / 'figures'

# 标注点：真实场景下的典型气隙（来自论文1）
ANNOTATIONS = {
    0.143: ('0.143 mm\n储罐曲面边缘气隙', 'tab:green'),
    0.300: ('0.3 mm\n储罐表面油漆厚度', 'tab:orange'),
    1.000: ('1 mm\n吸力降至 7%', 'tab:red'),
}


def magnetic_force(gap_mm):
    """气隙-吸力模型：gap_mm [mm] → 吸力 [N]。"""
    return EPM_F_MAX * np.exp(-np.asarray(gap_mm) / EPM_GAP_LAMBDA)


def main():
    mpl_style()
    gap = np.linspace(0.0, 1.0, 400)
    force = magnetic_force(gap)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(gap, force, c='tab:blue', lw=2.4, label='EPM 磁吸力模型')

    # 7% 参考线
    f7 = 0.07 * EPM_F_MAX
    ax.axhline(f7, c='tab:red', ls='--', lw=1.2, alpha=0.8)

    for g, (text, color) in ANNOTATIONS.items():
        f = magnetic_force(g)
        ax.scatter([g], [f], c=color, s=45, zorder=5)
        ax.annotate(f'{text}\n({f:.0f} N)', (g, f),
                    textcoords='offset points', xytext=(10, 12), color=color,
                    fontsize=9)

    ax.annotate(f'零气隙：{EPM_F_MAX:.0f} N (100%)', (0, EPM_F_MAX),
                textcoords='offset points', xytext=(8, -18), color='tab:blue',
                fontsize=9)

    ax.set_xlabel('气隙 [mm]')
    ax.set_ylabel('吸力 [N]')
    ax.set_title(f'EPM 磁吸力-气隙曲线（复现论文2 Fig.3，λ={EPM_GAP_LAMBDA:.3f} mm）')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 780)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / 'fig3_magnetic_force_gap.png'
    fig.savefig(out)
    print(f'已保存：{out}')
    print(f'关键数值：F(0)={magnetic_force(0):.1f} N, '
          f'F(0.3mm)={magnetic_force(0.3):.1f} N, '
          f'F(1mm)={magnetic_force(1.0):.1f} N (7% 目标: {f7:.1f} N)')


if __name__ == '__main__':
    main()
