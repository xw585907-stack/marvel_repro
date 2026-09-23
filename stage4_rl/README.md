# 阶段四：PPO 鲁棒磁吸附攀爬

本目录复现第二篇论文 Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment。

2026-09-23 状态：原端到端模型 `checkpoints/formal_seed0/ppo_final.pt` 在墙面评估失败，保留为对照。
二维工程适配版采用解析步态加 PPO 残差，并取消接触估计对磁铁的硬门控。完整课程后从第 33000 轮以更低学习率续训，当前最佳权重为
`checkpoints/gait_prior_low_lr_from33000_seed0/ppo_final.pt`。90° 墙面、85% 吸附、+0.15 m/s 的固定条件 5 组共 500 回合验证存活率 86.2%；开启训练随机化与观测噪声后为 68.6%。后退和站立在各 100 回合固定条件验证中均为 99%。这不是论文原始三维端到端算法的严格复现，详细设置与限制见 `docs/stage_report_4.md`。
最佳模型、第 33000 轮对照模型及关键训练日志已随本次报告上传；其余冗余检查点默认被 Git 忽略。

本机 D 盘环境由 micromamba 创建，解释器位于 `.venv/python.exe`（不是 Scripts 子目录）。
正式评估命令：

    .venv/python.exe stage4_rl/scripts/validate_stage4.py --checkpoint stage4_rl/checkpoints/gait_prior_low_lr_from33000_seed0/ppo_final.pt --theta 90 --prob-attach 0.85 --iteration 35000 --command 0.15 --episodes 100 --output stage4_rl/results/gait_prior_best/wall85_forward

更严格的随机化评估在同一命令中加入 `--randomized`；逐回合数据见 `results/gait_prior_best/`。

附加回归：

    .venv/python.exe stage4_rl/scripts/test_training_regressions.py

- rl/：PPO、Actor/Critic、状态估计器、课程学习和向量环境
- scripts/：训练、测试、评估和 Pygame 演示
- docs/：阶段四复现报告
- checkpoints/：保留快速测试权重和本报告的关键正式训练权重；其余训练权重默认被 Git 忽略

二维版本将原论文的 12 个关节动作缩减为 8 个关节动作，保留 4 个磁铁动作、8维时钟、三阶段课程、随机吸附失败和域随机化。

从项目根目录运行：

    .venv/Scripts/python.exe stage4_rl/scripts/test_rl.py
    .venv/Scripts/python.exe stage4_rl/scripts/train_ppo.py --quick
    .venv/Scripts/python.exe stage4_rl/scripts/train_ppo.py
    .venv/Scripts/python.exe stage4_rl/scripts/eval_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt
    .venv/Scripts/python.exe stage4_rl/scripts/run_rl.py --checkpoint stage4_rl/checkpoints/ppo_final.pt

--quick 只验证训练链路。正式论文课程使用 35000 次迭代。


当前随仓库上传的 `ppo_iter_00002.pt`、`ppo_iter_00004.pt` 和 `ppo_final.pt`
均来自快速训练，只用于换电脑后验证模型加载、评估和 Pygame 演示；不能作为论文结果。
