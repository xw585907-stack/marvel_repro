"""Pygame 渲染器：只负责可视化，不参与物理。

视图变换：屏幕 = Ry(−θ)·世界（论文2 的课程是旋转重力矢量，
地形保持水平；渲染时反向旋转 θ 即可把"地面"显示为倾角
(90°−θ) 的墙面——θ=90° 时地面显示为竖直墙，重力始终朝屏幕下方）。
"""
import numpy as np
import pygame

COLOR_BG = (25, 28, 35)
COLOR_FLOOR = (120, 130, 145)
COLOR_TORSO = (230, 150, 60)
COLOR_LEG = (200, 200, 210)
COLOR_FOOT_ON = (80, 220, 120)      # 磁吸生效
COLOR_FOOT_OFF = (150, 155, 165)    # 磁吸关闭
COLOR_CONTACT = (255, 230, 90)
COLOR_GRAVITY = (235, 90, 90)
COLOR_TEXT = (230, 230, 235)


class PygameRenderer:
    def __init__(self, scale=500.0, size=(1000, 700)):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption('MARVEL 2D 复现 - 磁吸附爬壁四足机器人')
        self.scale = scale
        self.cx, self.cy = size[0] / 2.0, size[1] / 2.0
        # pygame 2.6.1 的 SysFont 在 Windows 注册表扫描时崩溃（已知 bug），
        # 直接按文件路径加载中文字体，失败则退回默认字体
        self.font = None
        for path in (r'C:/Windows/Fonts/msyh.ttc', r'C:/Windows/Fonts/msyh.ttf',
                     r'C:/Windows/Fonts/simhei.ttf', r'C:/Windows/Fonts/simsun.ttc'):
            try:
                self.font = pygame.font.Font(path, 17)
                break
            except (OSError, pygame.error):
                continue
        if self.font is None:
            self.font = pygame.font.Font(None, 17)

    # ---------------- 坐标变换 ----------------
    def to_screen(self, points_world, theta):
        """世界坐标 (..., 2) → 屏幕坐标 (..., 2)，视图旋转 Ry(−θ)。"""
        c, s = np.cos(-theta), np.sin(-theta)
        Rv = np.array([[c, s], [-s, c]])
        v = points_world @ Rv.T
        return np.stack([self.cx + self.scale * v[..., 0],
                         self.cy - self.scale * v[..., 1]], axis=-1)

    # ---------------- 绘制 ----------------
    def draw(self, snap):
        theta = float(snap['theta'])
        p = snap['base_pos']
        phi = float(snap['base_phi'])
        q = snap['q']
        foot_w = snap['foot_pos']
        contact = snap['contact']
        magnet_ok = snap['magnet_on'] & snap['attach_ok']
        f_mag = snap['f_mag']

        self.screen.fill(COLOR_BG)

        # 地面（渲染视图中随 θ 倾斜）
        ground = np.array([[-3.0, 0.0], [3.0, 0.0]])
        pygame.draw.line(self.screen, COLOR_FLOOR,
                         self.to_screen(ground[0], theta),
                         self.to_screen(ground[1], theta), 3)

        # 重力箭头
        g_world = np.array([-np.sin(theta), -np.cos(theta)]) * 0.3
        tip = self.to_screen(p + g_world, theta)
        base = self.to_screen(p, theta)
        pygame.draw.line(self.screen, COLOR_GRAVITY, base, tip, 3)

        # 躯干（体坐标矩形 → 世界 → 屏幕）
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[c, s], [-s, c]])
        corners = np.array([[-0.165, -0.0655], [0.165, -0.0655],
                            [0.165, 0.0655], [-0.165, 0.0655]])
        corners_w = p + corners @ R.T
        pygame.draw.polygon(
            self.screen, COLOR_TORSO,
            [tuple(round(v) for v in pt)
             for pt in self.to_screen(corners_w, theta)])

        # 腿与足
        hip_offsets = np.array([[-0.165, 0.0], [0.165, 0.0],
                                [-0.165, 0.0], [0.165, 0.0]])
        l1 = l2 = 0.2
        for i in range(4):
            qh, qk = q[2 * i], q[2 * i + 1]
            hip_w = p + hip_offsets[i] @ R.T
            knee_b = np.array([l1 * np.sin(qh), -l1 * np.cos(qh)])
            knee_w = hip_w + knee_b @ R.T
            foot_b = np.array([l1 * np.sin(qh) + l2 * np.sin(qh + qk),
                               -l1 * np.cos(qh) - l2 * np.cos(qh + qk)])
            foot_w_i = foot_w[i] if i < len(foot_w) else p + hip_offsets[i] @ R.T + foot_b @ R.T

            pygame.draw.line(self.screen, COLOR_LEG,
                             self.to_screen(hip_w, theta),
                             self.to_screen(knee_w, theta), 5)
            pygame.draw.line(self.screen, COLOR_LEG,
                             self.to_screen(knee_w, theta),
                             self.to_screen(foot_w_i, theta), 5)
            # 足：绿=磁吸生效，灰=关闭；黄圈=接触中
            color = COLOR_FOOT_ON if magnet_ok[i] else COLOR_FOOT_OFF
            pygame.draw.circle(self.screen, color,
                               tuple(int(v) for v in
                                     self.to_screen(foot_w_i, theta)), 9)
            if contact[i]:
                pygame.draw.circle(self.screen, COLOR_CONTACT,
                                   tuple(int(v) for v in
                                         self.to_screen(foot_w_i, theta)),
                                   12, 2)

        # HUD
        lines = [
            f'theta = {np.degrees(theta):5.1f} deg   t = {snap["time"]:5.1f} s',
            '  '.join(f'{n}:{f_mag[i]:4.0f}N' for i, n in
                      enumerate(['RR', 'FR', 'RL', 'FL'])),
            '按键: 1=地面 2=45度 3=墙面 | M=磁开关 | R=复位 | Esc=退出',
        ]
        for j, text in enumerate(lines):
            surf = self.font.render(text, True, COLOR_TEXT)
            self.screen.blit(surf, (15, 15 + j * 24))

    def close(self):
        pygame.quit()
