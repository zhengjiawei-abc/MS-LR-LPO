# 多维 PDE 求解扩展

## 当前范围

二维扩展首先面向 PDEBench 的 2D diffusion-reaction 任务。数据和评价保持项目既有原则：

- 使用 PDEBench 官方 HDF5 轨迹；
- 先降采样再训练，原始大文件不提交到 Git；
- 使用空间、时间、通道顺序统一为 `[N, C, X, Y, T]`；
- 以逐样本相对 L2 作为主指标；
- 与参数量接近的二维 FNO 使用同一数据切分和训练预算。

该扩展验证的是二维时空算子学习，不等同于已经解决通用二维 NS、湍流或复杂边界问题。

## 模型

二维算子保持一维版本的核心结构：

```text
u0(x, y)
  -> 2D context encoder
  -> 2D manifold encoder
  -> latent pseudo-spectral PDE rollout
  -> factorized spatial/temporal decoder
  -> multi-scale skips
  -> u(x, y, t)
```

潜空间线性推进使用二维 Fourier 指数积分，包含可学习的通道平流和扩散项。非线性反应项由循环卷积与低秩谱混合块组成。解码器在低分辨率网格上交替执行二维空间卷积和一维时间卷积，从而避免全分辨率三维卷积。

二维 FNO 基线直接学习：

```text
u0(x, y) -> [u(x, y, t_1), ..., u(x, y, t_T)]
```

## 数据准备

准备脚本只读取所选训练、验证和测试区间，并输出紧凑 NPZ：

```powershell
& .\code\.venv\Scripts\python.exe .\datasets\prepare_pdebench_2d.py `
  --input .\datasets\pdebench_raw\2D_diff-react_NA_NA.h5 `
  --output-dir .\datasets\pdebench_diff_react_2d `
  --x-stride 2 --y-stride 2 --t-stride 2
```

默认切分使用官方文件中的轨迹编号：

```text
train = 0:800
dev   = 800:900
test  = 900:1000
```

官方二维 diffusion-reaction 序列的 `t=0` 是随机初场，经过一步后与后期图案的
空间相关系数快速衰减。测试集上，`t=0` 与 `t=50` 的平均相关系数只有约
`0.013`，而从 `t=10` 到 `t=50` 约为 `0.845`。因此当前二维实验从可辨识的
`t=10` 状态开始推进：

```powershell
& .\code\.venv\Scripts\python.exe .\datasets\slice_2d_trajectory.py `
  --input-dir .\datasets\pdebench_diff_react_2d `
  --output-dir .\datasets\pdebench_diff_react_2d_t10 `
  --time-start 10
```

这是“状态到未来轨迹”的算子设定。它不声称已经解决从随机噪声初场直接恢复全部
混沌演化的问题。

## 训练

PG 2D：

```powershell
& .\code\.venv\Scripts\python.exe .\code\train_operator_2d.py `
  --model pg2d `
  --data-dir .\datasets\pdebench_diff_react_2d_t10 `
  --output-dir .\experiments\outputs_2d\pg2d_diff_react_t10_seed7 `
  --steps 2500 --batch-size 2 --seed 7
```

FNO 2D：

```powershell
& .\code\.venv\Scripts\python.exe .\code\train_operator_2d.py `
  --model fno2d `
  --data-dir .\datasets\pdebench_diff_react_2d_t10 `
  --output-dir .\experiments\outputs_2d\fno2d_diff_react_t10_seed7 `
  --steps 2500 --batch-size 2 --seed 7
```

统一训练目标使用逐样本、逐通道、逐时刻的空间方差归一化 MSE，并额外约束
预测轨迹第一帧等于输入状态。这样可避免高幅随机初场压过后期低幅反应图案。

## 诊断图

```powershell
& .\code\.venv\Scripts\python.exe .\code\analyze_2d_models.py `
  --data-dir .\datasets\pdebench_diff_react_2d_t10 `
  --output-dir .\analysis_plots_2d `
  --pg-checkpoint .\experiments\outputs_2d\pg2d_diff_react_t10_seed7\model.pt `
  --fno-checkpoint .\experiments\outputs_2d\fno2d_diff_react_t10_seed7\model.pt
```

## 结果

实验使用 PDEBench 官方 2D diffusion-reaction 文件，轨迹切分为
`800/100/100`，空间降采样到 `64 x 64`，时间降采样到 `51` 帧后再从
`t=10` 起算。三个随机种子均为 `7, 11, 23`，每种模型训练 `2500` 步，
batch size 为 `2`。

| 模型 | 参数量 | Test RelL2 | 训练时间 |
|---|---:|---:|---:|
| PG-MS-LR-LPO-2D | 318,683 | **0.1091 ± 0.0078** | 490 s |
| FNO-2D | 302,602 | 0.2840 ± 0.0001 | 46 s |

PG 的平均相对误差比 FNO 低约 `61.6%`，参数量多约 `5.3%`。三次 PG
训练都没有触发早停，验证集仍在缓慢改善。代价是当前实现训练时间约为 FNO
的 `10.5 倍`，尚未针对二维推理效率做优化。三 seed 的 Welch t 检验
`p ≈ 6.6e-4`；样本数只有三组，这个数值用于描述当前差异，不能替代更大规模
的统计验证。

诊断图位于 `analysis_plots_2d/`，包括：

- 三个种子的 test 误差对比；
- 两个通道的 target/prediction/error 场图；
- 逐时刻相对 L2 曲线。

## 边界

当前结果只覆盖二维、双通道、周期区域的 diffusion-reaction，且使用了空间和
时间降采样。它证明了二维潜空间 PDE 解码链路和 `t=10` 状态算子设定有效，
但还不能据此声称解决二维不可压缩 NS、湍流、复杂边界或多物理耦合问题。

当前正式训练使用数据场损失与初始状态约束。HDF5 配置给出了 `Du`、`Dv` 和
`k`，但没有在样本配置中给出完整的反应项参数，因此本轮没有把 Gray-Scott
观测场残差直接加入训练；模型中的“PDE”部分指潜空间谱演化，而不是已经完成
的二维 PINN 物理残差约束。

## 二维不可压缩 NS

PDEBench 官方 `ns_incom_inhom_2d_512-0.h5` 的结构为：

```text
velocity: [4 trajectories, 1000 frames, 512, 512, 2]
force:    [4 trajectories, 512, 512, 2]
domain:   [0, 1] x [0, 1]
NU:       0.01
```

速度通道为 `(u, v)`，外力通道为 `(f_x, f_y)`。模型采用力条件的状态到未来
窗口算子：

```text
(u0, v0, f_x, f_y) -> (u(x, y, t), v(x, y, t))
```

数据准备命令：

```powershell
& .\code\.venv\Scripts\python.exe .\datasets\prepare_ns_incom_2d.py `
  --input .\datasets\pdebench_raw\ns_incom_inhom_2d_512-0.h5 `
  --output-dir .\datasets\pdebench_ns_incom_2d `
  --x-stride 8 --y-stride 8 --window 41 `
  --train-stride 10 --eval-stride 20
```

空间从 `512 x 512` 降采样到 `64 x 64`。采用两条轨迹训练、一条验证、一条
测试，避免把同一条轨迹的重叠窗口泄漏到测试集：

```text
train = trajectories 0, 1 (192 windows)
dev   = trajectory  2    (48 windows)
test  = trajectory  3    (48 windows)
```

训练目标额外加入有限差分不可压缩散度约束：

```powershell
& .\code\.venv\Scripts\python.exe .\code\train_operator_2d.py `
  --model pg2d `
  --data-dir .\datasets\pdebench_ns_incom_2d `
  --output-dir .\experiments\outputs_2d\ns_pg_seed7 `
  --steps 2500 --batch-size 2 --seed 7 --divergence-weight 0.02
```

三个随机种子的结果为：

| 模型 | 参数量 | Velocity RelL2 | Divergence RelError | 训练时间 |
|---|---:|---:|---:|---:|
| PG-MS-LR-LPO-2D | 322,683 | **0.1816 ± 0.0058** | **0.0344 ± 0.0015** | 406 s |
| FNO-2D | 302,650 | 0.3167 ± 0.0273 | 0.0994 ± 0.0196 | 59 s |

逐窗口散度指标在降采样真值上的有限差分底噪约为 `0.0065`。PG 的 `0.0344`
约为底噪 `5.3` 倍，FNO 的 `0.0994` 约为底噪 `15.3` 倍。PG 明显更接近
离散真值，但仍需结合速度谱检查平滑偏差。PG 的速度误差均值比 FNO 低约
`42.7%`，三个 seed 的 Welch t 检验
`p ≈ 0.011`，但样本量仍只有三组。PG 三次分别在 1300–1400 步触发早停，
FNO 三次均未触发早停。

该实验明确验证了力条件输入、二维速度多通道输出和散度约束。由于官方文件
只提供速度与外力的 `(u, v)` 分量，没有压力场，当前还没有加入完整
Navier-Stokes 动量方程残差，也没有覆盖 512² 全分辨率或复杂几何边界。
