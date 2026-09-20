# 阶段四复现报告：PPO 鲁棒磁吸附攀爬

## 1. 对应论文

Yong Um et al., Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment, arXiv:2510.20174, 2025。

阶段四在阶段二的自研 NumPy 物理环境上实现论文的强化学习框架，Pygame 继续只负责可视化。

## 2. 二维适配

原论文机器人具有 12 个关节动作和 4 个磁铁动作。本项目采用二维矢状面模型，因此动作缩减为：

- 8 个关节目标位置动作；
- 4 个磁铁开关动作；
- 总动作维度 12。

保留论文中的关节状态、两步动作历史、机身姿态、足端相对位置、状态估计量、速度指令和 8 维步态时钟。Actor 输入维度为 61，Critic 输入维度为 61，状态估计目标维度为 10。

## 3. 网络和 PPO

- Actor：MLP [256, 128, 64]
- Critic：MLP [256, 128, 64]
- 状态估计器：MLP [256, 128]
- PPO：clip=0.2，gamma=0.99，GAE lambda=0.95，学习率 3e-4
- 状态估计器与策略并发训练，监督目标为机身线速度、4 足高度和4足接触状态
- 观测使用论文给出的低通滤波系数 alpha=0.35

## 4. 三阶段课程

严格实现论文 Eq. (5)-(7)：

1. iteration < 1200：平地步态学习，物理磁吸关闭，但保留磁铁动作时序奖励；
2. iteration 1200~21200：重力方向从 0 度线性旋转到 90 度，启用磁吸模型；
3. iteration 21200~35000：保持 90 度墙面，吸附成功率从 1.00 线性下降到 0.85。

调试参数 --curriculum-scale 只压缩课程时间轴，默认值 1.0 对应论文原始节点。

## 5. 吸附和随机化

训练复用阶段二的四步吸附判定：接触识别、磁铁动作阈值、随机吸附、气隙衰减。状态估计器输出的接触置信度参与磁铁门控。

每回合随机化：

- 关节 PD 增益相对标称值 0.8~1.2 倍；
- 摩擦系数 0.3~0.5；
- 姿态、关节、速度、历史和足端位置观测噪声；
- 0~8 ms 动作延迟。

由于二维执行器参数与论文 RaiSim 模型量纲不同，PD 随机范围按相对比例映射。

## 6. 奖励

实现 Table I 的二维对应项：线速度、角速度、站立、步态、足高、足滑、足端间隙、姿态、关节力矩、关节位置/速度/加速度、两阶动作平滑、机身运动和磁铁时序。总奖励采用论文 Eq. (14) 的正奖励乘指数惩罚形式。

## 7. 验证结果

stage4_rl/scripts/test_rl.py：10/10 通过，包括：

- 课程节点和吸附概率；
- 51维本体观测、61维 Critic 观测；
- 12维动作、10维估计器输出；
- 域随机化范围；
- 一次完整 PPO 更新；
- 检查点保存/加载；
- 并行环境局部复位。

工程冒烟训练：

- 4 次短训练可完成采样、更新和保存；
- 40 次压缩课程训练依次进入 Phase 1、2、3；
- 最终调度达到 theta=90 度、Prob_attach=0.85；
- 训练过程无 NaN，估计器损失从 0.224 降至 0.142。

40 次训练仅用于验证工程链路。其 5 回合墙面评估为速度 RMSE 5.6442 m/s、提前终止率 100%、平均时长 1.06 s、保持率 4.3%，说明随机初始化策略尚未收敛，不能作为论文效果复现结果。

## 8. 正式训练

从项目根目录运行：

    .venv/Scripts/python.exe stage4_rl/scripts/train_ppo.py

默认使用 64 个并行环境、32 步 rollout 和 35000 次论文课程迭代。本机单次默认迭代实测约 3.2 秒，完整训练预计约 31 小时，因此本次没有在交互会话中自动启动长训练。完成后运行：

    .venv/Scripts/python.exe stage4_rl/scripts/eval_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt --episodes 100
    .venv/Scripts/python.exe stage4_rl/scripts/run_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt

阶段五将使用正式训练权重复现 Table II 的 Full、w/o Curriculum、w/o Probabilistic Adhesion 和 w/o Modeling 对比。
