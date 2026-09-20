# 阶段三：MARVEL SRB-MPC

本目录复现第一篇论文 Agile and versatile climbing on ferromagnetic surfaces with a quadrupedal robot 的控制部分。

- control/：SRB-MPC、步态调度、摆动轨迹和力矩前馈
- scripts/：17项回归测试、Pygame 演示和诊断脚本
- docs/：阶段三复现报告

从项目根目录运行：

    .venv/Scripts/python.exe stage3_mpc/scripts/test_mpc.py
    .venv/Scripts/python.exe stage3_mpc/scripts/run_mpc.py
