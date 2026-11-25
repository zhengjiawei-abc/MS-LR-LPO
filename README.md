# 物理引导多尺度低秩潜空间 PDE 神经算子

**Physics-Guided Multi-Scale Low-Rank Latent PDE Operator (PG-MS-LR-LPO)**

当前标准模型是在原 MS-LR-LPO 上加入可选周期常速平移分支的物理引导神经算子。
主线流程为：

完整方法、实验协议、结果边界与复现说明见
[技术报告](docs/TECHNICAL_REPORT.md)。

PDF 版本见
[PG-MS-LR-LPO 技术报告（PDF）](output/pdf/PG-MS-LR-LPO-Technical-Report.pdf)。

多维 PDE 扩展、官方数据转换和训练命令见
[多维 PDE 求解扩展](docs/MULTIDIMENSIONAL_PDE.md)。

首个 PDEBench 2D diffusion-reaction 三 seed 结果为 PG-Lite `0.1053 ± 0.0006`、
参数匹配 FNO `0.2840 ± 0.0001`、DeepONet `0.7986 ± 0.0036`
（`t=10` 状态到未来轨迹，64×64×41）。

在 PDEBench 2D incompressible NS 上，PG-Lite 速度 `0.1742 ± 0.0035`、散度
`0.0366 ± 0.0020`；FNO 分别为 `0.3167 ± 0.0273` 和 `0.0994 ± 0.0196`。

一维 advection 在 ±20% 参数先验、8 个时间快照和 0–10% 噪声下，β 反演
100% 落在 5% 误差内，MAPE 为 `0.036%–0.049%`。

β=2、4、7 的最终残差阶段已完成三 seed 稳定性检查。β=2、4 仍优于参数匹配 FNO；
β=7 三 seed 均值为 `0.011357 ± 0.000008`，仍落后 FNO 的 `0.009003`。

Burgers 高监督 DeepONet 三 seed 为 `0.2532 ± 0.0050`，仍落后 PG
`0.0198` 与 FNO `0.0246`。1D 推理延迟 PG `22.0 ms`、FNO `1.20 ms`、
DeepONet `6.07 ms`（batch=32）。

```text
u0(x) -> context encoder
      -> LatentPDE time evolution
      -> multi-scale skips
      -> low-rank grid decoder
      -> physics-guided transport fusion
      -> u(x, t)
```

## 项目图

![PG-MS-LR-LPO architecture](analysis_plots_clean/architecture_diagram.png)

![PG-MS-LR-LPO cross-PDE performance](analysis_plots_clean/performance_comparison.png)

![Advection parameter sweep](analysis_plots_clean/parameter_sweep_comparison.png)

![Residual training budget analysis](analysis_plots_clean/residual_budget_analysis.png)

![High-speed residual branch ablation](analysis_plots_clean/beta7_residual_ablation.png)

![Residual branch three-seed stability](analysis_plots_clean/residual_seed_summary.png)

![Base operator ablation study](analysis_plots_clean/ablation_study.png)

![PDEBench Burgers dataset analysis](analysis_plots_clean/dataset_analysis.png)

![Base operator training curves](analysis_plots_clean/training_curves.png)

![PDEBench 2D diffusion-reaction three-seed comparison](analysis_plots_2d/seed_comparison.png)

![PDEBench 2D diffusion-reaction error over time](analysis_plots_2d/time_error_curve.png)

![PDEBench 2D diffusion-reaction sample fields](analysis_plots_2d/sample_00_channel_0.png)

![PDEBench 2D incompressible NS three-seed comparison](analysis_plots_2d_ns/seed_comparison.png)

![PDEBench 2D incompressible NS error over time](analysis_plots_2d_ns/time_error_curve.png)

![PDEBench 2D incompressible NS sample fields](analysis_plots_2d_ns/sample_00_channel_0.png)

![PDEBench 2D diffusion-reaction module ablation](analysis_plots_2d/ablation_summary.png)

![PDEBench 2D incompressible NS module ablation](analysis_plots_2d_ns/ablation_summary.png)

![PDEBench 2D diffusion-reaction radial spectrum](analysis_plots_2d/radial_spectrum.png)

![PDEBench 2D incompressible NS radial spectrum](analysis_plots_2d_ns/radial_spectrum.png)

![PDEBench 2D diffusion-reaction efficiency](analysis_plots_2d/efficiency_summary.png)

![PDEBench 2D incompressible NS efficiency](analysis_plots_2d_ns/efficiency_summary.png)

![Bounded advection beta inversion](analysis_plots_clean/bounded_beta_inference.png)

![Burgers DeepONet supervision ablation](analysis_plots_clean/deeponet_supervision.png)

![PDEBench 1D efficiency](analysis_plots_clean/efficiency_1d_summary.png)

## 标准模型

神经算子输出记为 `u_n`。当任务已知周期常速平流系数 `beta` 时，模型额外构造：

```text
u_p(x, t) = u0((x - beta * t) mod 1)

u(x, t) = u_p(x, t)
          + alpha * (u_n(x, t) - u_p(x, t))
```

低 beta 的 advection 基准使用固定 `alpha = 0`。高速平移时可以把 `alpha` 设为可学习参数，
让神经分支拟合平移基线的离散残差。固定分支不增加参数，可学习分支只增加 1 个参数。
Burgers 和 diffusion-sorption 没有已知常速平移项，因此关闭该分支，模型退化为原 MS-LR-LPO。

这个分支是物理先验，不是对所有 PDE 都生效的通用加速器。Burgers 的输运速度依赖局部解 `u`，
不能直接复用常数 `beta`。

## 主要结果

所有结果使用 PDEBench 官方数据，采用相同划分：

- train: `8000`
- dev: `1000`
- test: `1000`
- 下表为 seed 7，评价指标为 full-grid relative Test L2

| 数据集 | 原 MS-LR-LPO | PG-MS-LR-LPO | FNO | DeepONet |
|---|---:|---:|---:|---:|
| Burgers `Nu=0.01` | 0.019759 | **0.019758** | 0.024355 | 0.267974 |
| 1D diffusion-sorption | 0.003312 | **0.003310** | 0.701195 | 0.014917 |
| 1D advection `beta=1.0` | 0.017053 | **0.004543** | 0.006542 | 0.440765 |

物理引导模型相对两个参数匹配 baseline：

| 数据集 | 比 FNO 低 | 比 DeepONet 低 |
|---|---:|---:|
| Burgers | 19.61% | 92.57% |
| Diffusion-sorption | 99.52% | 81.06% |
| Advection | 31.44% | 99.09% |

DeepONet 每个训练步只随机采样 `256` 个时空点进行监督；PG-MS-LR-LPO 和 FNO 使用完整
`256 x 101` 输出场。因此 DeepONet 数字反映当前标准配置，不应被解释为完全相同的监督条件。

下面的百分比汇总和性能图使用三 seed 均值；主表保留 seed 7 原始值，便于逐项核对。

### 结果边界与可声明范围

当前可支持的结论是：

- 在 Burgers `Nu=0.01`、diffusion-sorption 和已测六个 advection β 上，
  PG-MS-LR-LPO 在绝大多数设置中优于参数匹配 FNO。
- 在所有已测设置中，PG-MS-LR-LPO 均优于标准 DeepONet。
- β=2、4 的可学习残差可以超过 FNO；β=7 的高频分支三 seed 最终阶段均值为
  `0.011357 ± 0.000008`，仍落后 FNO 的 `0.009003`，是当前明确失败案例。
- β=2、4、7 的三 seed 只改变最终残差微调阶段，三个 seed 共享早期 checkpoint；
  该结果证明最终阶段稳定，不等同于三次从零独立训练。

尚不能声称“全面超过官方 PDEBench”：

- Burgers 只使用 `Nu=0.01`，不是全部 12 个粘度文件。
- Advection 使用了 6 个 β，尚缺官方 `0.1` 和 `0.4`。
- 1D CFD、2D CFD、Darcy 和 SWE 尚未执行；NS 只覆盖二维力条件速度场子集，
  尚无完整压力场和全分辨率结果。
- FNO/DeepONet 是项目内的参数匹配实现，不是 PDEBench 官方榜单结果。
- DeepONet 仍使用 256 点随机监督，需要补充全监督版本。

## Advection 改进

Advection 原模型 Test L2 为 `0.017053`。单纯从 12K checkpoint 继续训练 6K 后降至
`0.015620`，仍落后 FNO。启用周期平移分支后：

```text
Test L2 = 0.004543
```

相比原模型降低约 `73.4%`。参数匹配后的 FNO 配置为 `modes=25, width=68`
（503,165 参数），三 seed Test L2 为 `0.006626 +- 0.000109`；PG-MS-LR-LPO
输出不依赖随机初始化，仍比它低约 `31.4%`。

平移速度只在 dev 的 `0.98..1.02` 范围内搜索，dev 最优值为 `1.0`，没有使用 test 调参。
线性插值在 dev 上优于 Fourier shift 和三次插值，因此作为标准实现。

### Advection 三 seed

| 方法 | Seed 7 | Seed 11 | Seed 23 | Mean +- Std |
|---|---:|---:|---:|---:|
| PG-MS-LR-LPO | 0.004543 | 0.004543 | 0.004543 | **0.004543 +- 0** |
| 参数匹配 FNO | 0.006542 | 0.006749 | 0.006586 | 0.006626 +- 0.000109 |
| 参数匹配 DeepONet | 0.440765 | 0.501917 | 0.548459 | 0.497047 +- 0.054012 |

PG 对 FNO 的 Welch p 为 `0.00091`，对 DeepONet 为 `0.00399`。

### 高速平移与残差融合

`beta=2.0` 时，线性平移的离散误差明显增大。固定平移 Test L2 为 `0.008304`，
反而落后 FNO 的 `0.006300`。将融合系数设为可学习，并从固定平移解继续训练 4K 步后：

本段保留 seed 7 的优化轨迹；下面的六参数汇总改用 FNO/DeepONet 三 seed 均值，
因此 β=2 的 FNO 数值为 `0.006389`。

| beta=2.0 方法 | 参数 | Test L2 |
|---|---:|---:|
| 基础神经模型 | 501,435 | 0.020495 |
| 固定周期平移 | 501,435 | 0.008304 |
| PG + 可学习残差融合（36K） | 501,436 | **0.005043** |
| 参数匹配 FNO | 503,165 | 0.006300 |
| DeepONet | 501,473 | 0.6260 |

学习得到的融合系数为 `0.0423`。残差训练里程碑：

| 残差步数 | 8K | 12K | 18K | 24K | 28K | 32K | 36K |
|---|---:|---:|---:|---:|---:|---:|---:|
| Test L2 | 0.005639 | 0.005503 | 0.005305 | 0.005181 | 0.005127 | 0.005080 | **0.005043** |

32K 后单 1K 收益已低于约 `0.1%`，36K 仍无 dev/test 过拟合迹象，但进入低收益平台。
这说明高速平流不能只依赖解析平移，必须学习插值残差。

### 参数网格与反演可辨识性

当前 advection 参数网格已扩展到：

```text
beta = 0.2, 0.7, 1.0, 2.0, 4.0, 7.0
```

六参数 Test L2（FNO/DeepONet 为三 seed 均值；可学习残差为最终阶段三 seed）：

| beta | 固定平移 PG | 可学习残差 PG | FNO | DeepONet |
|---:|---:|---:|---:|---:|
| 0.2 | **0.001645** | - | 0.005010 | 0.132452 |
| 0.7 | **0.003762** | - | 0.006435 | 0.326126 |
| 1.0 | **0.004543** | - | 0.006626 | 0.497047 |
| 2.0 | 0.008304 | **0.005043 ± 0.000002** | 0.006389 | 0.623089 |
| 4.0 | 0.017987 | **0.007183 ± 0.000001** | 0.007262 | 0.701016 |
| 7.0 | 0.033748 | 0.011357 ± 0.000008 | **0.009003** | 0.706937 |

β≤1 时解析平移已经很强；β≥2 后离散插值误差快速增大，必须依靠可学习残差。
β=2 和 β=4 的独立残差模型已经重新超过 FNO。β=7 的最强变体使用 128 点解码网格、
30 个 Fourier bands、17 个 rollout anchors 和 hidden-20 局部残差滤波器，从
`0.014044` 改善到 `0.011357 ± 0.000008`，提升约 19%，但仍落后 FNO。最终参数为 `503,161`，
比 FNO 少 4 个。256 点网格在约 500 步时需要 10 分钟，暂不采用。

### 残差分支最终阶段三 seed

三个 seed 使用相同入口 checkpoint，仅最终残差阶段分别使用 seed 7、11、23：

| beta | Seed 7 | Seed 11 | Seed 23 | Mean +- Std | FNO |
|---:|---:|---:|---:|---:|---:|
| 2.0 | 0.00504318 | 0.00504162 | 0.00504465 | **0.00504315 +- 0.00000152** | 0.006389 |
| 4.0 | 0.00718284 | 0.00718236 | 0.00718401 | **0.00718307 +- 0.00000085** | 0.007262 |
| 7.0 | 0.01135244 | 0.01135171 | 0.01136538 | 0.01135651 +- 0.00000769 | 0.009003 |

β=2 和 β=4 的优势在 seed 间稳定存在；β=7 的失败也没有被 seed 选择掩盖。

对多参数轨迹做无先验 β 扫描时发现：周期离散平移存在速度混叠。在时间步长
`dt=0.02`、空间网格 `NX=256` 时，相隔 `1/(dt*NX)=0.1953` 的候选速度会产生近似
整数格点平移。因此“无范围约束的全局参数反演”是不适定的。

这不代表反演不可做，而是论文必须加入：

- β 的物理先验范围；
- 非均匀时间观测；
- 多初值或多视图数据；
- 或直接学习参数后验分布。

当前反演实验用于诊断可辨识性，不作为最终成功率结果。

## Burgers 三 seed

关闭平移分支后，PG-MS-LR-LPO 与 Burgers 原模型等价。三 seed Test L2：

| 方法 | Seed 7 | Seed 11 | Seed 23 | Mean +- Std |
|---|---:|---:|---:|---:|
| PG-MS-LR-LPO，平移分支关闭 | 0.019759 | 0.019775 | 0.019767 | **0.019767 +- 0.000008** |
| 参数匹配 FNO | 0.024355 | 0.024388 | 0.025020 | 0.024588 +- 0.000375 |
| 参数匹配 DeepONet | 0.267974 | 0.261488 | 0.268660 | 0.266041 +- 0.003958 |

与 FNO 的三 seed Welch t 检验为 `p = 0.0020`。

## Diffusion-sorption 三 seed

| 方法 | Seed 7 | Seed 11 | Seed 23 | Mean +- Std |
|---|---:|---:|---:|---:|
| PG-MS-LR-LPO | 0.003312 | 0.003326 | 0.003472 | **0.003370 +- 0.000089** |
| 最佳参数匹配 FNO | 0.701195 | 0.701195 | 0.701195 | 0.701195 +- 0.000000 |
| 参数匹配 DeepONet | 0.014917 | 0.020987 | 0.017468 | 0.017791 +- 0.003048 |

PG 对 FNO 的 Welch p 为 `5.37e-9`，对 DeepONet 为 `0.0145`。

## 参数口径

| 模型 | 参数量 | 说明 |
|---|---:|---|
| PG-MS-LR-LPO | 501,435 | rank17 低秩解码器，标准参数口径 |
| PG + 可学习残差融合 | 501,436 | 高速平移变体，仅增加 1 个融合参数 |
| PG + 128 网格 + x_bands 32 | 502,972 | β=7 高频变体，接近 FNO 参数预算 |
| PG + 高频残差滤波器 | 503,161 | β=7 最终变体，比 FNO 少 4 个参数 |
| Burgers 原 checkpoint | 498,555 | rank16，三 seed 精度实测来源 |
| 参数匹配 FNO | 502,997-503,165 | Burgers 使用 width 104, modes 10；跨 PDE 使用 width 68, modes 25 |
| 参数匹配 DeepONet | 501,473 | basis 128, branch 512, trunk 148 |

固定平移分支不使用可训练参数；可学习残差融合只增加 1 个标量。Burgers 表中新模型使用
rank16 checkpoint，向 rank17 的零扩展不会改变前向函数，因此与 `501,435` 输出严格一致。

## 物理残差

Burgers 可选物理残差微调：

```text
r = u_t + u u_x - nu u_xx
PDE relative = ||r|| / characteristic_scale
```

| 模型 | Test L2 | PDE Relative | 参数量 |
|---|---:|---:|---:|
| 数据-only 模型 | 0.01976 | 0.4523 | 501,435 |
| 加 PDE residual 微调 | 0.02038 | 0.3864 | 501,435 |
| 真值场参考 | - | 0.1469 | - |

残差下降约 `14.6%`，但数据误差会略有增加。当前标准结果仍以数据-only 模型为准。

## 数据准备

### Burgers

```bash
python datasets/download_parallel.py \
  --url https://hf-mirror.com/datasets/pdebench/Burgers/resolve/main/1D_Burgers_Sols_Nu0.01.hdf5 \
  --output datasets/pdebench_raw/1D_Burgers_Sols_Nu0.01.hdf5 \
  --md5 e6d9a4f62baf9a29121a816b919e2770 \
  --connections 32

python datasets/pdebench_burgers_to_parquet.py \
  --input datasets/pdebench_raw/1D_Burgers_Sols_Nu0.01.hdf5 \
  --output-dir datasets/pdebench_burgers_official
```

### Advection

```bash
python datasets/download_parallel.py \
  --url https://hf-mirror.com/datasets/pdebench/Advection/resolve/main/1D_Advection_Sols_beta1.0.hdf5 \
  --output datasets/pdebench_raw/1D_Advection_Sols_beta1.0.hdf5 \
  --md5 1fe41923a4123db55bf4e89bea32e142 \
  --connections 32

python datasets/pdebench_group_to_parquet.py \
  --input datasets/pdebench_raw/1D_Advection_Sols_beta1.0.hdf5 \
  --output-dir datasets/pdebench_advection_beta1_official \
  --name advection_beta1 \
  --parameters beta=1.0 \
  --t-stride 2 --x-stride 4
```

### Diffusion-sorption

```bash
python datasets/download_parallel.py \
  --url https://darus.uni-stuttgart.de/api/access/datafile/133020 \
  --output datasets/pdebench_raw/1D_diff-sorp_NA_NA.h5 \
  --md5 9d466d1213065619d087319e16d9a938 \
  --connections 32

python datasets/pdebench_group_to_parquet.py \
  --input datasets/pdebench_raw/1D_diff-sorp_NA_NA.h5 \
  --output-dir datasets/pdebench_diff_sorp_official \
  --name diff_sorp \
  --parameters D=0.0005 \
  --t-stride 1 --x-stride 4
```

Diffusion-sorption 原始时间范围为 `0..500`。项目把 `101` 个均匀时间层映射到模型现有的
`[0, 2]` 坐标域，仅用于数据-only 学习，不计算 Burgers 物理残差。

## 训练与复现

安装依赖：

```bash
pip install -r code/requirements.txt
pip install torch h5py
```

训练基础神经算子：

```bash
python code/train_coord_attention.py \
  --data-dir datasets/pdebench_advection_beta1_official \
  --dataset-prefix advection_beta1 \
  --train-samples 8000 \
  --steps 12000 \
  --batch-size 64 \
  --loss-mode full \
  --output-mode grid \
  --grid-hidden 48 \
  --grid-blocks 3 \
  --grid-length 64 \
  --grid-lowrank-rank 17 \
  --dynamics latent_pde \
  --latent-channels 96 \
  --latent-length 64 \
  --pde-layers 1 \
  --pde-rank 24 \
  --pde-modes 32 \
  --pde-substeps 2 \
  --rollout-steps 9 \
  --seed 7 \
  --save-model \
  --output-dir experiments/my_advection_base
```

启用 advection 周期平移分支并保存标准模型：

```bash
python code/train_coord_attention.py \
  --data-dir datasets/pdebench_advection_beta1_official \
  --dataset-prefix advection_beta1 \
  --grid-lowrank-rank 17 \
  --grid-transport-velocity 1.0 \
  --grid-transport-blend 0.0 \
  --init-checkpoint experiments/my_advection_base/model.pt \
  --evaluate-only \
  --save-model \
  --output-dir experiments/my_advection_pg
```

一次运行跨 PDE 四模型矩阵（base、PG、FNO、DeepONet）：

```bash
python code/run_cross_pde_matrix.py \
  --seed 7 \
  --current-steps 12000 \
  --baseline-steps 24000
```

参数匹配 FNO：

```bash
python code/baselines_1d_burgers.py \
  --model fno \
  --data-dir datasets/pdebench_advection_beta1_official \
  --dataset-prefix advection_beta1 \
  --train-samples 8000 --dev-samples 1000 --test-samples 1000 \
  --steps 24000 --batch-size 64 \
  --width 104 --modes 10 --layers 4 \
  --seed 7 \
  --output-dir experiments/my_fno
```

参数匹配 DeepONet：

```bash
python code/baselines_1d_burgers.py \
  --model deeponet \
  --data-dir datasets/pdebench_advection_beta1_official \
  --dataset-prefix advection_beta1 \
  --train-samples 8000 --dev-samples 1000 --test-samples 1000 \
  --steps 24000 --batch-size 64 \
  --basis 128 --branch-hidden 512 --trunk-hidden 148 \
  --query-points 256 \
  --seed 7 \
  --output-dir experiments/my_deeponet
```

## 代码目录

| 路径 | 内容 |
|---|---|
| `code/coord_attention_operator.py` | LatentPDE、上下文编码、低秩解码、skip 和周期平移分支 |
| `code/train_coord_attention.py` | 训练、checkpoint 加载、evaluate-only 和物理分支配置 |
| `code/baselines_1d_burgers.py` | 参数匹配 FNO 与 DeepONet |
| `code/benchmark_1d_models.py` | 1D 参数、延迟、吞吐与显存基准 |
| `code/run_cross_pde_matrix.py` | Burgers、advection、diffusion-sorption 四模型矩阵 |
| `code/run_residual_seed_queue.py` | β=2、4、7 最终残差阶段 seed 队列 |
| `code/evaluate_advection_beta_inference.py` | advection β 多参数反演与离散可辨识性诊断 |
| `code/burgers_common.py` | 数据读取、split 解析、误差与坐标工具 |
| `datasets/pdebench_group_to_parquet.py` | 分组和整块 tensor 两种 PDEBench 1D HDF5 转换 |
| `code/coord_attention_operator_2d.py` | 二维潜空间谱 PDE、多尺度编码和分解式时空解码 |
| `code/train_operator_2d.py` | PG/FNO/DeepONet 二维统一训练、相对时刻损失和早停 |
| `code/baselines_2d.py` | 参数匹配二维 FNO 与 Fourier-trunk DeepONet 基线 |
| `code/analyze_2d_models.py` | 二维场图、逐时刻误差和指标汇总 |
| `code/benchmark_2d_models.py` | 二维参数、延迟、吞吐和显存基准 |
| `code/spectral_2d_diagnostics.py` | 二维径向功率谱和高频能量诊断 |
| `datasets/prepare_pdebench_2d.py` | PDEBench 二维分组 HDF5 转换与降采样 |
| `datasets/prepare_ns_incom_2d.py` | 二维不可压缩 NS 时间窗、外力条件输入和轨迹切分 |
| `code/operator_2d_common.py` | 二维误差、时窗和不可压缩散度工具 |
| `analysis_plots_clean/make_current_figures.py` | 生成 README 当前图片 |

## 已知限制

- Burgers、diffusion-sorption、advection 的模型对比均补齐三 seed；
  advection 物理分支输出确定，因此标准差为 0。
- β=2、4、7 的可学习残差仅完成最终阶段三 seed；早期 checkpoint 仍共享。
- 周期平移分支只适用于已知常速、周期边界的线性平流。
- 高 beta 下定点平移受离散插值误差限制，需要使用可学习残差分支。
- Diffusion-sorption 使用归一化时间坐标，不能直接复用 Burgers PDE residual。
- 当前模型训练成本明显高于 FNO 和 DeepONet。
- 二维 diffusion-reaction 当前从 `t=10` 的可辨识状态推进；从随机噪声 `t=0`
  直接预测全过程会出现严重信息损失，不能称为已解决该病态设定。
- 二维实验使用 `64 x 64 x 41` 的降采样轨迹，尚未覆盖全分辨率、二维 NS、
  湍流或复杂边界。
- 二维正式结果尚未加入观测场 Gray-Scott 物理残差；官方样本配置缺少完整
  反应项参数，因此当前“PDE”指潜空间谱演化，而不是完整二维 PINN 残差约束。
- NS 已加入速度散度约束，并显式输入 `(f_x, f_y)` 外力，但官方文件没有
  压力场，因此尚未加入完整 NS 动量方程残差；结果也只覆盖 `64 x 64` 和
  四条官方轨迹。
- 二维最终轻量配置去掉非线性潜空间谱混合块，保留 Fourier 线性推进和
  multi-scale skip；该配置在 reaction/NS 上都优于带谱块版本。
- PG 精度优于 FNO，但当前推理延迟约为 FNO 的 14–15 倍，峰值显存约 231 MB
  对 29 MB，尚不具备推理效率优势。
- 将 FNO 改为 `width=68, modes=25` 后，diffusion-sorption 仍为 `0.701195`，说明低分不只是
  `modes=10` 的限制，后续需要检查任务条件、输入输出归一化和训练目标。
- 数据、checkpoint 和实验输出保存在本地 `datasets/` 与 `experiments/`，不提交到 Git。

## 联系

zhengjiawei-abc  
1103493759@qq.com

最后更新：2026-11-25
