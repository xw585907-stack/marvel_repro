"""必须在 import numpy 之前导入：OpenBLAS 单线程设置。

OpenBLAS 多线程对 100-300 维稠密 solve 有 ~1000 倍开销病态
（120x120 LU: 24 线程 210ms vs 单线程 0.22ms），本项目所有矩阵
均 <= 300 维，单线程恒更快。详见 docs/stage_report_3.md 第4节。

用法：每个入口脚本（scripts/*.py、analysis/*.py）在 import numpy
之前 `import _blas`。环境变量必须在 numpy 载入 OpenBLAS 前设置。
"""
import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
