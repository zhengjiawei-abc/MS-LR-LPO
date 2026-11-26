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

潜空间线性推进使用二维 Fourier 指数积分，包含可学习的通道平流和扩散项。非线性反应项由循环卷积组成；消融显示额外低秩谱混合块在两个二维任务上均冗余，因此最终配置设为 `pde_layers=0`。解码器在低分辨率网格上交替执行二维空间卷积和一维时间卷积，从而避免全分辨率三维卷积。

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
  --steps 2500 --batch-size 2 --seed 7 --pde-layers 0
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
| PG-MS-LR-LPO-2D Lite | 245,155 | **0.1053 ± 0.0006** | 240 s |
| FNO-2D | 302,602 | 0.2840 ± 0.0001 | 46 s |
| DeepONet-2D | 293,698 | 0.7986 ± 0.0036 | 27 s |

二维轻量标准去掉了非线性潜空间谱混合块，但保留 Fourier 线性潜空间推进。
相较带谱块的完整版，参数从 `318,683` 降到 `245,155`，Test RelL2 从
`0.1091` 改善到 `0.1053`。去掉全部 skip 后 Seed 7 退化到 `0.3065`，
说明多尺度 skip 是必要模块。

PG-Lite 的平均相对误差比 FNO 低约 `62.9%`，并且比 FNO 少约 `19%`
参数；三 seed Welch t 检验 `p ≈ 2.6e-6`。DeepONet 在该任务上明显欠拟合。
PG 训练时间约为 FNO 的 `5.2 倍`。

诊断图位于 `analysis_plots_2d/`，包括：

- 三个种子的 test 误差对比；
- 两个通道的 target/prediction/error 场图；
- 逐时刻相对 L2 曲线。

## 边界

当前结果只覆盖二维、双通道、周期区域的 diffusion-reaction，且使用了空间和
时间降采样。它证明了二维潜空间 PDE 解码链路和 `t=10` 状态算子设定有效，
但还不能据此声称解决二维不可压缩 NS、湍流、复杂边界或多物理耦合问题。

当前正式训练使用数据场损失与初始状态约束。官方配置给出的反应项实际为
`u-u^3-k-v` 与 `u-v`，参数为 `Du=1e-3`、`Dv=5e-3`、`k=5e-3`；先前把它写成
经典 Gray-Scott 形式是错误的。修正后的 test residual 为真值 `0.009507`、
PG `0.034724`、FNO `0.136032`。正式模型尚未使用该修正 residual 重训。

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
  --steps 2500 --batch-size 2 --seed 7 `
  --divergence-weight 0.02 --pde-layers 0
```

三个随机种子的结果为：

| 模型 | 参数量 | Velocity RelL2 | Divergence RelError | 训练时间 |
|---|---:|---:|---:|---:|
| PG-MS-LR-LPO-2D Lite | 249,155 | **0.1742 ± 0.0035** | **0.0366 ± 0.0020** | 233 s |
| FNO-2D | 302,650 | 0.3168 ± 0.0273 | 0.0994 ± 0.0196 | 59 s |
| DeepONet-2D | 285,442 | 8.1238 ± 4.7577 | 0.9379 ± 0.0722 | 24 s |

逐窗口散度指标在降采样真值上的有限差分底噪约为 `0.0065`。PG-Lite 的
`0.0366` 约为底噪 `5.6` 倍，FNO 的 `0.0994` 约为 `15.3` 倍。PG-Lite
的速度误差均值比 FNO 低约 `45.0%`，且参数少约 `17.7%`；三 seed 的
Welch t 检验 `p ≈ 0.011`。PG 三次分别在 1400–1700 步触发早停。

去掉非线性潜空间谱块后，NS 速度误差从 `0.1816` 降到 `0.1742`，参数从
`322,683` 降到 `249,155`；散度从 `0.0344` 略升到 `0.0366`。去掉 skip
后 Seed 7 误差升到 `0.3077`，散度升到 `0.0515`，因此最终配置保留 skip。

该实验明确验证了力条件输入、二维速度多通道输出和散度约束。审计阶段新增了
压力梯度消去后的动量散度缺陷，并使用源文件 particles/force；真值、PG、FNO
分别为 `1.305054 / 1.136847 / 1.095111`。真值本身不接近零，说明该粗网格
诊断不能作为绝对物理正确性证明。仍未覆盖 512² 全分辨率或复杂几何边界。

### NS 参数反演的数据边界

当前使用的 `ns_incom_inhom_2d_512-0.h5` 在文件属性中固定：

```text
NU = 0.01
force_scale = 0.15
force_smoothness = 3.0
scale = 0.4
```

四条轨迹只改变初始速度与随机外力实现，不提供不同粘度或不同外力强度下的
参数标签。因此，在该文件上做“从速度场恢复 `NU` 或 `force_scale`”没有可核验
的真值变化；用训练集自造一组标量再拟合不能形成有含金量的官方数据反演实验。

要补齐二维 NS 标量参数反演，需要额外下载或生成一组按 `NU` / `force_scale`
参数化的轨迹，并在训练轨迹与测试轨迹之间按参数取值切分。当前 2D 结果只支持
外力条件前向预测和散度约束，不支持二维 NS 标量参数反演结论。

## 效率

统一基准使用 RTX 4060 Laptop GPU、batch size 8、`64 x 64 x 41` 输出，
报告 100 次重复的中位延迟：

| 任务/模型 | 延迟 | 吞吐 | 峰值显存 | checkpoint |
|---|---:|---:|---:|---:|
| Reaction PG-Lite | 26.1 ms | 306 samples/s | 231 MB | 0.97 MB |
| Reaction FNO | 1.72 ms | 4,641 samples/s | 29 MB | 1.16 MB |
| Reaction DeepONet | 2.33 ms | 3,434 samples/s | 206 MB | 33.8 MB |
| NS PG-Lite | 26.6 ms | 301 samples/s | 231 MB | 0.98 MB |
| NS FNO | 1.90 ms | 4,213 samples/s | 29 MB | 1.16 MB |
| NS DeepONet | 2.33 ms | 3,435 samples/s | 207 MB | 33.8 MB |

PG 的精度优势伴随明显推理代价：延迟约为 FNO 的 `14–15` 倍。DeepONet
推理快，但 checkpoint 包含全时空 trunk 坐标，文件约 `34 MB`，且两个任务
精度都明显落后。对应图见 `analysis_plots_2d*/efficiency_summary.png`。

## 频谱诊断

对输出做二维 Hann 窗 FFT，并统计半径超过 Nyquist `25%` 的高频能量占比：

| 任务 | Target | PG-Lite | FNO | DeepONet |
|---|---:|---:|---:|---:|
| Reaction | 4.70e-4 | 5.95e-4 | 1.77e-3 | 2.15e-5 |
| NS | 9.30e-6 | 7.82e-6 | 5.01e-5 | 8.07e-5 |

Reaction 上 PG-Lite 的高频占比最接近 Target，FNO 明显放大高频，DeepONet
过度平滑。NS 上 PG-Lite 略偏平滑，但没有出现 FNO/DeepONet 的高频放大。
这支持“PG 的精度不是单纯来自空间均值坍缩”的判断，同时提示 NS 仍有少量
谱偏差。
