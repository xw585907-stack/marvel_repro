# 阶段四：PPO 鲁棒磁吸附攀爬

本目录复现第二篇论文 Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment。

- rl/：PPO、Actor/Critic、状态估计器、课程学习和向量环境
- scripts/：训练、测试、评估和 Pygame 演示
- docs/：阶段四复现报告
- checkpoints/：随仓库保留快速链路测试权重；后续新生成权重默认被 Git 忽略

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
