# 阶段四开源实现核对（2026-09-21）

此次检索未发现 arXiv:2510.20174 作者公开的可直接复用训练仓库；这不等于作者没有公开代码。

## 已核对的项目

- [RSL-RL PPO](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)：
  BSD-3-Clause。参考其对 time_outs 做回报 bootstrap、监控 KL 的做法。
  本项目使用自动复位前的 terminal critic observation 计算 bootstrap，
  而非直接拿当前状态 value 近似。没有引入 RSL-RL 运行时依赖或声称使用其完整算法。
- [CleanRL continuous PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)：
  参考非负采样 KL 估计和 target-KL 停止更新的思路；这是本项目独立实现的局部修改。
  与 RSL-RL 的自适应学习率不同，此处是超阈值后结束当前轮更新。
- [magneto_rl](https://github.com/swanbeck/magneto_rl)：对象是 Magneto，包含 DQN、高层墙面路径规划及 ROS 仿真，
  与当前连续关节 PPO/二维 MARVEL 物理接口不同，未直接合并代码。
- [marmotlab/MARVEL](https://github.com/marmotlab/MARVEL)：同名多机器人探索项目，非磁吸四足机器人，未采用。

## 本项目额外实验

`--bounded-policy` 现在使用 tanh squashed Gaussian：先从高斯采样，再用 tanh 将动作映射到 [-1,1]，
并在 PPO log probability 中加入变量变换修正；熵奖励也对该变换保留梯度。
它是二维适配的实验性约束，不是声称来自原论文或 RSL-RL 的默认设计。
旧未约束检查点保持原策略形式；早期使用“有界均值 + 硬裁剪采样”的旧检查点
不能继续训练，代码会明确拒绝这种恢复操作。
恢复训练时加载保存的 PPO 配置，命令行可显式覆盖 target KL 和学习率。

旧正式权重不会因为代码修改而自动获得修复，需要独立实验验证。
