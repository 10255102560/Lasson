基于近端梯度法的 Lasso 稀疏回归求解与数值实验
================================================

这是一份已经实际运行完成的课程研究项目。采用随机生成的稀疏回归问题，
自行实现 ISTA 与 FISTA；scikit-learn 只用于独立的高精度数值核验。

先看这些文件
------------
report/Lasso研究报告.pdf       中文研究报告，适合直接阅读和打印。
report/Lasso研究报告.docx      可编辑 Word 版本，包含可编辑数学公式。
report/lasso_report.tex         完整 LaTeX 源文件；图表数字由程序读取结果生成。
results/validation.json         已保存结果的完整性、精度及数据选参核验。
results/metadata.json           随机种子、软件版本、线程配置、源文件哈希和耗时。

数学约定
--------
目标函数：||Ax-b||²/(2n) + lambda*||x||_1。
梯度：A.T@(Ax-b)/n。软阈值：sign(z)*max(abs(z)-eta*lambda,0)。
默认 eta=1/L，L=||A||_2²/n；矩阵 2 范数为最大奇异值。
训练集中心化并标准化特征；截距不惩罚。
两种算法同时满足归一化 KKT 残差和原始—对偶间隙 <= 1e-6 才停止。
FISTA 为标准未重启版本；利用二次损失的仿射梯度性质复用梯度计算。
solve 的 L 参数应为正确的 Lipschitz 上界；不传时程序自行求得。

完成的实验
----------
1. 算法比较：n=300，p=1000，s=20；rho=0、0.8；每组 5 个种子。
   lambda=0.05*lambda_max。每种方法计时 3 次取中位数，再跨种子统计。
2. 步长比较：ISTA 的 eta 为 0.25/L、0.5/L、1/L。
3. 正则化比较：lambda/lambda_max 为 0.001、0.003、0.01、0.03、0.1、0.3、1。
4. 三个单因素分析：样本数 150/300/600，噪声比例 0.05/0.1/0.3，
   特征相关系数 0/0.5/0.9。共享基础场景，共 7 个场景、35 个数据集。
5. 每个数据集求解 7 个 lambda 候选，共 245 次拟合。只按验证集 MSE 选参；
   再使用 ISTA 复核这 35 个选定模型。测试集不参与选参。
6. 10 个基准问题和 35 个选定模型分别用独立坐标下降求参照解。

关键结论和完整数值请看报告。特别注意：预测效果、参数恢复和支持集识别是
不同指标。较低的预测误差不代表没有假阳性变量；FISTA 的理论加速界也不
保证每个实例、每个停止精度上的固定时间加速。

复现运行
--------
本机测试版本为 Python 3.13.12，Apple M4，OpenBLAS 单线程。
项目的 .venv 是本次工作环境，不放入交付压缩包。
在项目根目录执行（以下 python 指已激活的虚拟环境）：

  python3.13 -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements-report.txt
  python -W error::RuntimeWarning -m unittest discover -s tests -v
  python run_experiments.py
  python verify_results.py
  python make_figures.py
  python build_report.py

若只重新计算和绘图，安装 requirements.txt 即可。
PDF 编译还需要系统安装 XeLaTeX 和 CTeX（如 TeX Live / MacTeX）；
Word 导出使用 pypandoc_binary 自带的 Pandoc。无需下载任何研究数据。
build_report.py 会更新 report/ 下的生成文件，但不重新运行实验。
只读核验已有数值可运行 verify_results.py；无需再次计时。
完整实验本次约 73 秒，具体用时以 metadata.json 为准，其他机器可能不同。

Apple Silicon 后端说明
----------------------
本机原有 NumPy/Accelerate 组合对有限正常矩阵乘法发出异常浮点警告。
为避免屏蔽真实数值错误，本项目改用 OpenBLAS 的 NumPy/SciPy wheel，并在
将 RuntimeWarning 视为异常的条件下完成正式运行。未修改用户全局环境。
如果 Apple Silicon 安装默认 wheel 后遇到同类告警，可在虚拟环境中使用：

  python -m pip download --only-binary=:all: --no-deps --platform macosx_12_0_arm64 --python-version 313 --abi cp313 --implementation cp --dest /tmp/lasso_wheels numpy==2.2.6 scipy==1.15.3
  python -m pip install --force-reinstall --no-deps /tmp/lasso_wheels/numpy-2.2.6-cp313-cp313-macosx_11_0_arm64.whl /tmp/lasso_wheels/scipy-1.15.3-cp313-cp313-macosx_12_0_arm64.whl

以上替换命令仅适用于 macOS ARM64 + Python 3.13；其他平台正常安装依赖即可。
相关上游问题：https://github.com/numpy/numpy/issues/28687
实际使用的后端和线程数已写入 metadata.json。其他 BLAS 后端的时间不可直接等同。

结果文件说明
------------
benchmark.csv           每个种子/方法的迭代次数、计时中位数、证书及统计指标。
timing_runs.csv         全部 60 次正式计时的原始数据。
spectral_setup.csv      一次性计算 L 的耗时；不计入求解循环时间。
lambda_sweep.csv        全部 245 个候选模型的原始数据。
selected_models.csv     35 个验证集选定模型及 ISTA 复核信息。
baselines.csv           基础场景 Lasso、最小范数最小二乘和仅截距的统计比较。
step_sizes.csv          15 组步长实验。
reference_checks.csv    45 个独立坐标下降参照解的最优性证书。
traces/                 代表性实例每轮的目标值、对偶下界、残差与间隙。
base_problem.npz        第一个基础实例的训练/验证/测试数组、真实参数和预处理量。
base_solutions.npz      同一实例的参数路径、选定解、参照解和最小二乘解。
*_summary.csv           图表生成脚本计算的跨种子均值与样本标准差。
report_numbers.json     由 CSV 提取的报告数值宏。
run.log                 正式实验执行日志。

支持集阈值为原始尺度系数绝对值 > 1e-5；KKT 证书使用算法输出的精确零。
所有“均值 ± 标准差”均跨 5 个随机种子计算，标准差分母为 4，不是置信区间。
数据和模型结果由固定种子复现；计时取决于硬件、系统负载与后端，不能逐位复现。
source_sha256 用于识别此次实验对应代码；修改代码后应重新运行实验再更新报告。

主要参考来源
------------
Tibshirani (1996), Regression Shrinkage and Selection via the Lasso.
https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
Beck & Teboulle (2009), A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems.
https://doi.org/10.1137/080716542
Daubechies, Defrise & De Mol, arXiv:math/0307152.
https://arxiv.org/abs/math/0307152
scikit-learn 1.6.1 Lasso 文档：
https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.Lasso.html

PDF 的版面已检查；Word 版本供编辑使用，不同办公软件可能自动重新分页。
