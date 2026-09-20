# marvel_repro

两篇磁吸附爬壁四足机器人论文的复现项目（Pygame 自研 2D 仿真环境 + Python 实现）：

1. **论文1（MPC 基线）**：Hong et al., *Agile and versatile climbing on ferromagnetic surfaces
   with a quadrupedal robot (MARVEL)*, Science Robotics, 2022. DOI: 10.1126/scirobotics.add1017
2. **论文2（RL 方法）**：Um et al., *Reinforcement Learning-based Robust Wall Climbing
   Locomotion Controller in Ferromagnetic Environment*, arXiv:2510.20174, 2025

## 复现思路

- 自研 **2D 矢状面** 磁吸附四足机器人环境：物理内核用 numpy 向量化实现（并行批量仿真），
  Pygame 仅负责渲染与键盘交互。
- 复现论文1 的 MPC 控制框架（SRB-MPC 力分配 + Bézier 摆动腿）作为基线。
- 复现论文2 的 PPO 框架（三阶段课程学习 + 随机吸附失效 + 域随机化）。
- 重做论文2 Table II 消融实验与 Fig.5 的 MPC vs RL 吸附失效对比。

## 目录结构

```
marvel_repro/
├── envs/        # 仿真核心：动力学、接触、EPM 模型、地形、向量化 env（Gym 风格）
├── render/      # pygame 渲染器（演示/调试用，训练时不渲染）
├── stage3_mpc/  # 阶段3：论文1 SRB-MPC、步态、测试、Pygame 演示
├── stage4_rl/   # 阶段4：论文2 PPO、课程学习、训练、评估与演示
├── analysis/    # 复现图表与指标（静态分析、消融实验）
└── scripts/     # 阶段2物理测试与基础 Pygame 演示
```

## 环境

使用项目自带 venv（基于 D:\anaconda 的 Python 3.12）：

```bash
# 已由搭建脚本完成；如重建：
D:/anaconda/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# torch 仅阶段4（PPO）需要（本项目实测 torch 2.5.1+cpu 可用）
```

## 阶段进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 环境搭建 | 完成 |
| 1 | 静态分析复现（Fig.4F 可容许反力区域 / Fig.3 气隙力曲线） | 完成 |
| 2 | 2D 物理内核（动力学/接触/EPM/向量化环境） | 完成 |
| 3 | 论文1 MPC 控制器 | 完成（17/17 测试通过） |
| 4 | 论文2 PPO 控制器 | 实现完成（10/10；正式训练待运行） |
| 5 | 对比与消融实验（论文2 Table II） | 待开始 |
| 6 | 复现报告 | 待开始 |

阶段 0–2 详细报告见 [docs/stage_report_0-2.md](docs/stage_report_0-2.md)；
阶段3 详细报告见 [stage3_mpc/docs/stage_report_3.md](stage3_mpc/docs/stage_report_3.md)。
阶段4 详细报告见 [stage4_rl/docs/stage_report_4.md](stage4_rl/docs/stage_report_4.md)。
汇总材料：[复现报告材料](docs/report_material.md)（对老师汇报用）与
[开发日志](docs/dev_log.md)（初定计划→每一步→错误→思考修改→最后结果）。

## 运行

```bash
# 阶段1：静态分析复现图
.venv/Scripts/python.exe analysis/admissible_region.py   # → figures/fig4f_admissible_region.png
.venv/Scripts/python.exe analysis/magnetic_force.py      # → figures/fig3_magnetic_force_gap.png

# 阶段2：物理内核测试（12 项检查）与交互演示
.venv/Scripts/python.exe scripts/test_env.py
.venv/Scripts/python.exe scripts/sim_demo.py             # 键盘: 1/2/3=倾角, M=磁开关, R=复位

# 阶段3：MPC 控制器测试（17 项）与交互演示
.venv/Scripts/python.exe stage3_mpc/scripts/test_mpc.py            # 4 场景: 平地/墙面行走/墙面保持/天花板
.venv/Scripts/python.exe stage3_mpc/scripts/debug_mpc.py           # QP解/速度/力实现/z扰动 诊断
.venv/Scripts/python.exe stage3_mpc/scripts/run_mpc.py             # 键盘: 1/2/3/4=表面, 上下=调速, R=复位
# 阶段4：PPO 接口测试、训练、评估与 Pygame 演示
.venv/Scripts/python.exe stage4_rl/scripts/test_rl.py
.venv/Scripts/python.exe stage4_rl/scripts/train_ppo.py --quick
.venv/Scripts/python.exe stage4_rl/scripts/train_ppo.py
.venv/Scripts/python.exe stage4_rl/scripts/eval_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt
.venv/Scripts/python.exe stage4_rl/scripts/run_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt
```

> 注：所有入口脚本在 `import numpy` 前先 `import _blas`（项目根目录），
> 强制 OpenBLAS 单线程——多线程对 100-300 维稠密 solve 有 ~1000 倍开销
> 病态，详见阶段3 报告第4节。新脚本请沿用此约定。

## 阶段2 物理内核要点

- 躯干单刚体（论文1 SRB 假设）+ 无质量腿（关节 PD + 虚拟惯量），半隐式欧拉积分
- 罚函数法向接触（k_n=1e6, d_n=1500）+ 锚点库仑摩擦（切向弹簧 k_t=2e5 实现 stick-slip）
- EPM 模型（论文2 Eq.1-4）：四步吸附判定链 + 气隙指数衰减 F=F_max·exp(−gap/λ)，
  落地掷骰 Prob_attach（课程 1.0→0.85）
- 物理步长 dt=0.2ms（罚函数刚度 + 磁吸近表面等效刚度 ~1.85e6 N/m 的俯仰模态
  ω≈1.4e3 rad/s，dt=1ms 时半隐式欧拉在刚度切换处注入能量导致俯仰发散）
- 12/12 测试通过：平地站立、垂直墙吸附（ΣF_t=mg=78.5N 静平衡）、断磁滑落、
  随机吸附统计 87%≈85%


## 在另一台电脑恢复项目

```bash
git clone https://github.com/xw585907-stack/marvel_repro.git
cd marvel_repro
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

RTX 50 系显卡应从 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
安装与显卡驱动匹配的 CUDA 版 PyTorch，覆盖 `requirements.txt` 中用于当前电脑验证的
`torch==2.5.1`。安装后先运行共享环境、阶段3和阶段4三组测试，再开始阶段4正式训练。

仓库随附的 `stage4_rl/checkpoints/*.pt` 是 `--quick` 产生的工程链路测试权重，
仅用于确认加载、评估和渲染流程，不代表论文的正式训练结果。正式复现实验需要重新运行
35000 次迭代训练。
