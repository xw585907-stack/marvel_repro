"""论文2 Eq. (5)-(7) 的三阶段课程调度。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CurriculumState:
    iteration: int
    paper_iteration: float
    phase: int
    theta: float
    prob_attach: float
    adhesion_enabled: bool
    kappa: float


class ClimbingCurriculum:
    """三阶段课程。

    默认 ``scale=1`` 严格使用论文节点 1200/21200/35000。
    调试时可用较小 scale 压缩时间轴，但状态计算仍按论文等效迭代数进行。
    """

    def __init__(self, scale=1.0):
        if scale <= 0:
            raise ValueError('curriculum scale 必须为正数')
        self.scale = float(scale)
        self.phase1_end = max(1, int(round(1200 * self.scale)))
        self.phase2_end = max(self.phase1_end + 1,
                              int(round(21200 * self.scale)))
        self.phase3_end = max(self.phase2_end + 1,
                              int(round(35000 * self.scale)))

    def state(self, iteration):
        iteration = max(int(iteration), 0)
        t = iteration / self.scale

        theta_frac = min(max((t - 1200.0) / 20000.0, 0.0), 1.0)
        theta = 0.5 * math.pi * theta_frac
        prob_frac = min(max((t - 21200.0) / 13800.0, 0.0), 1.0)
        prob_attach = 1.0 - 0.15 * prob_frac

        if t < 1200.0:
            phase = 1
        elif t <= 21200.0:
            phase = 2
        else:
            phase = 3

        kappa = 0.99975 ** max(t - 1200.0, 0.0)
        return CurriculumState(
            iteration=iteration,
            paper_iteration=t,
            phase=phase,
            theta=theta,
            prob_attach=prob_attach,
            adhesion_enabled=phase >= 2,
            kappa=kappa,
        )
