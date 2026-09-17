"""地形：2D 线段地面。

阶段2 只有平地。论文2 的课程学习是"旋转重力矢量"而非旋转地形
（Eq.5-6：g(t) = Ry(θ)g0），因此地形始终是水平地面，
θ=90° 时配合渲染旋转即可呈现"爬垂直墙"的视角。
后续阶段（障碍/缝隙）在此扩展。
"""
import numpy as np


class FlatFloor:
    """平面地面 z = 0，法线 (0, 1)。世界坐标：x 前，z 上。"""

    normal = np.array([0.0, 1.0])

    def height(self, x):
        """地面高度 z(x)。"""
        return np.zeros_like(np.asarray(x))

    def surface_normal(self, x):
        """表面法线 (..., 2)。"""
        return np.broadcast_to(self.normal, np.asarray(x).shape + (2,))
