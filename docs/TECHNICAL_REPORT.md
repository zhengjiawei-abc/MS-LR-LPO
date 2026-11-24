# PG-MS-LR-LPO 技术报告

**Physics-Guided Multi-Scale Low-Rank Latent PDE Operator**

版本：2026-11-24  
代码仓库：<https://github.com/zhengjiawei-abc/PG-MS-LR-LPO>

## 摘要

PG-MS-LR-LPO 是一个面向参数化一维偏微分方程的物理引导神经算子。模型使用 CNN/上下文编码器和流形编码器建立低维潜空间，通过 LatentPDE 在潜空间中进行时间演化，再由多尺度 skip 和低秩二维网格解码器恢复全场解。当问题存在已知周期常速平移项时，模型加入解析平移基线，并通过可学习残差分支修正离散插值误差。

当前实验覆盖 PDEBench 的三个一维标量任务：Burgers、diffusion-sorption 和 advection。Burgers 与 diffusion-sorption 已完成三 seed 验证；advection 已覆盖六个参数：

```text
beta = 0.2, 0.7, 1.0, 2.0, 4.0, 7.0
```

实验结果表明，在参数预算接近时，PG-MS-LR-LPO 在绝大多数已测设置中优于 FNO，并在所有已测设置中优于标准 DeepONet。β=7 仍是明确失败案例：PG 的高频残差分支将误差从 `0.014044` 降到 `0.011352`，但仍落后 FNO 的 `0.009003`。

本报告同时说明当前模型的能力边界、参数反演可辨识性、复现方法和投稿前仍需补充的实验。

## 1. 问题定义

模型学习如下时空解算子：

```text
G: u0(x) -> u(x,t)
```

输入为初始场 `u0(x)`，输出为有限时间范围内的完整时空场。当前网格为：

```text
x in [0, 1), periodic
NX = 256
NT = 101
```

训练目标为全场数据 MSE：

```text
L_data = mean((u_pred(x,t) - u_true(x,t))^2)
```

评价指标为逐样本相对 L2 的平均值：

```text
RelL2 = mean_n( ||u_pred_n - u_true_n||_2
                / max(||u_true_n||_2, eps) )
```

## 2. 模型结构

### 2.1 总体流程

```text
u0(x)
  -> context encoder
  -> latent manifold encoder
  -> LatentPDE rollout
  -> multi-scale skip features
  -> low-rank grid decoder
  -> physics-guided output fusion
  -> u(x,t)
```

### 2.2 Context encoder

Context encoder 使用周期卷积和 Transformer token 表示输入场。主干使用 circular padding，避免周期边界处的零填充不连续。默认产生 32 个上下文 token，用于多尺度 skip 门控和输出解码。

### 2.3 流形编码器与 LatentPDE

流形编码器将输入场压缩为：

```text
96 channels x 64 latent points
```

LatentPDE 在潜空间中执行时间演化，包含：

- Fourier 域线性平流项；
- 频域扩散项；
- 局部非线性项；
- 可学习反应项；
- 低秩谱混合块。

默认 rollout 为 9 个时间锚点。β=7 高频实验中将 rollout 提升到 17 个锚点，不增加可训练参数。

### 2.4 多尺度 skip

Context encoder 提供 256、128、64 三种空间尺度特征。解码器使用时间门控融合不同尺度的 skip。Burgers 消融显示 64 点粗尺度 skip 最重要。

### 2.5 低秩网格解码器

标准解码器在 `64 x 101` 的粗网格上执行二维卷积，再双线性恢复到 `256 x 101`。三个残差块使用低秩 SVD 分解：

```text
rank = 17
```

β=7 实验中解码网格提高到 128，并增加 Fourier 坐标 band 数。该改动不改变卷积块参数。

### 2.6 物理引导平移

对于周期常速线性平流，解析基线为：

```text
u_p(x,t) = u0((x - beta*t) mod 1)
```

实际实现使用周期索引和线性插值。最终输出：

```text
u = u_p + alpha * (u_n - u_p)
```

其中：

- `u_n` 是神经解码器输出；
- `alpha` 可以是固定标量，也可以是可学习标量；
- 固定平移分支不增加参数；
- 可学习融合只增加 1 个参数。

β≤1 使用 `alpha=0`。β=2、4、7 使用可学习残差融合。

### 2.7 高频残差修正

β=7 的最终变体额外使用一个局部残差滤波器。令 `r = u_n - u_p`，则：

```text
r <- r + Conv3x3(r)
u = u_p + alpha * r
```

滤波器结构为：

```text
1 -> hidden -> 1
```

最终卷积初始化为零，因此加入滤波器时初始函数保持不变。最终配置：

```text
grid_length = 128
x_bands = 30
rollout_steps = 17
hidden = 20
```

## 3. 参数预算

| 模型/变体 | 参数量 |
|---|---:|
| Burgers rank16 checkpoint | 498,555 |
| 标准 PG rank17 | 501,435 |
| PG + 可学习融合 | 501,436 |
| FNO Burgers | 502,997 |
| PG + 128 网格 + x_bands 32 | 502,972 |
| 跨 PDE FNO | 503,165 |
| PG + 高频残差滤波器 | 503,161 |
| DeepONet | 501,473 |

标准 PG 与跨 PDE FNO 的参数差约 `0.35%`。β=7 高频 PG 比 FNO 少 4 个参数。

## 4. 数据与实验协议

### 4.1 数据

所有实验使用 PDEBench 官方 HDF5 文件，转换为项目 parquet。

官方划分：

```text
test  = 0:1000
train = 1000:9000
dev   = 9000:10000
```

Advection 原始数据为 `201 x 1024`，转换时使用：

```text
t-stride = 2
x-stride = 4
```

Diffusion-sorption 原始时间范围为 `0..500`，项目把 101 个均匀时间层映射到模型 `[0,2]` 坐标域。该任务只做数据层面的学习，不使用 Burgers 物理残差。

### 4.2 Baselines

FNO：

```text
Burgers: width=104, modes=10, layers=4
跨 PDE:  width=68, modes=25, layers=4
```

DeepONet：

```text
basis=128
branch_hidden=512
trunk_hidden=148
query_points=256
```

DeepONet 每个训练步使用 256 个随机时空点监督，而 PG 和 FNO 使用全场监督。因此 DeepONet 结果不能视为完全相同的监督条件。

### 4.3 随机种子

- Burgers：seed 7、11、23。
- Diffusion-sorption：seed 7、11、23。
- Advection FNO/DeepONet：seed 7、11、23。
- advection 固定平移分支：确定性，无 seed 方差。
- β=2、4、7 的可学习残差：当前主结果使用 seed 7。

## 5. 实验结果

### 5.1 Burgers

| 方法 | Test L2 |
|---|---:|
| PG，三 seed 均值 | **0.019767 ± 0.000008** |
| FNO，三 seed 均值 | 0.024588 ± 0.000375 |
| DeepONet，三 seed 均值 | 0.266041 ± 0.003958 |

PG 相对 FNO 降低约 `19.6%`。Welch t 检验 `p=0.0020`。

### 5.2 Diffusion-sorption

| 方法 | Test L2 |
|---|---:|
| PG，三 seed 均值 | **0.003370 ± 0.000089** |
| FNO，三 seed 均值 | 0.701195 |
| DeepONet，三 seed 均值 | 0.017791 ± 0.003048 |

PG 相对 FNO 降低约 `99.5%`，相对 DeepONet 降低约 `81.1%`。

FNO 在 modes=16、24 和 width 重分配后仍停在 `0.701195`，说明异常不只是谱模态数量不足。该 baseline 仍需进一步检查任务条件和归一化。

### 5.3 Advection 参数扫描

FNO 和 DeepONet 为三 seed 均值；固定平移为确定性结果；可学习残差当前为 seed 7。

| beta | 固定平移 PG | 可学习残差 PG | FNO | DeepONet |
|---:|---:|---:|---:|---:|
| 0.2 | **0.001645** | - | 0.005010 | 0.132452 |
| 0.7 | **0.003762** | - | 0.006435 | 0.326126 |
| 1.0 | **0.004543** | - | 0.006626 | 0.497047 |
| 2.0 | 0.008304 | **0.005043** | 0.006389 | 0.623089 |
| 4.0 | 0.017987 | **0.007183** | 0.007262 | 0.701016 |
| 7.0 | 0.033748 | 0.011352 | **0.009003** | 0.706937 |

结论：

- β≤1 时，解析平移先验占主导。
- β=2、4 时，独立可学习残差可以超过 FNO。
- β=7 时，高频残差分支显著改善 PG，但仍未超过 FNO。

### 5.4 残差训练预算

β=2 的残差训练从 1K 扫描到 36K：

| 残差步数 | Test L2 |
|---:|---:|
| 1K | 0.006351 |
| 8K | 0.005639 |
| 12K | 0.005503 |
| 18K | 0.005305 |
| 24K | 0.005181 |
| 32K | 0.005080 |
| **36K** | **0.005043** |

32K 后收益降到低水平，36K 仍无 dev/test 过拟合迹象。

β=4 独立残差在 18K 达到 `0.007183`，首次超过 FNO 的 `0.007262`。

### 5.5 β=7 结构消融

| 配置 | Test L2 |
|---|---:|
| 固定平移 | 0.033748 |
| 64 网格残差 | 0.013577 |
| 128 网格残差 | 0.013429 |
| + x_bands 32 | 0.012490 |
| + rollout 17 | 0.012297 |
| + 10 参数修正器 | 0.011945 |
| + 153 参数修正器 | 0.011689 |
| + hidden 16 | 0.011494 |
| + hidden 20 | **0.011352** |
| FNO | **0.009003** |

256 解码网格在约 500 步时耗时约 10 分钟，成本过高，未作为最终配置。

## 6. 参数反演与可辨识性

无先验地从轨迹估计 β 时，周期离散平移存在速度混叠。当前时间步长和空间网格下：

```text
Delta_beta approximately 1 / (dt * NX)
```

在 `dt=0.02`、`NX=256` 时，混叠间隔约 `0.1953`。多组 β 会产生近似相同的整数格点平移。

因此：

- 全局无约束反演是不适定的；
- 实际反演必须给定物理先验范围；
- 应使用非均匀时间观测、多初值或多视角数据；
- 仅报告干净轨迹上的全局 argmin 会误导。

当前反演实验用于可辨识性诊断，不作为最终成功率结论。

## 7. 当前能力边界

### 已支持

- 一维周期标量 PDE 数据学习。
- 参数化 advection 前向预测。
- 已知参数下的物理引导残差修正。
- 多参数、三 seed baseline 对比。
- checkpoint 保护、指标记录和经验复现。

### 未支持或未验证

- 多变量可压缩 NS。
- 二维不可压缩 NS 和湍流。
- 无先验通用参数反演。
- 时变边界、源项和复杂几何。
- 生产级推理服务和不确定性保证。

## 8. 投稿前仍需补充

1. 全监督或高监督 DeepONet。
2. β=0.1、0.4，补齐官方八个参数值。
3. β=2、4、7 残差分支的三 seed 验证。
4. 训练时间、推理延迟、显存和吞吐量表。
5. 最终模块消融表。
6. 带参数先验范围的逆问题实验。
7. 若声称通用 PDE 算子，需要增加 1D CFD 或二维 NS。

## 9. 可声明与不可声明

可声明：

> 在 PDEBench 一维标量任务和项目内参数匹配基线上，PG-MS-LR-LPO 在绝大多数已测设置中优于 FNO，并在所有已测设置中优于标准 DeepONet。

不可声明：

> 已全面超过官方 PDEBench 的 FNO/DeepONet，或已经解决通用多维 NS、逆问题和参数反演。

## 10. 复现入口

主要脚本：

| 路径 | 作用 |
|---|---|
| `code/coord_attention_operator.py` | PG-MS-LR-LPO 与残差融合 |
| `code/train_coord_attention.py` | 训练、里程碑保存、checkpoint 恢复 |
| `code/baselines_1d_burgers.py` | FNO/DeepONet |
| `code/run_cross_pde_matrix.py` | 跨 PDE 训练矩阵 |
| `code/evaluate_advection_beta_inference.py` | β 可辨识性诊断 |
| `datasets/pdebench_group_to_parquet.py` | 官方 HDF5 转换 |
| `analysis_plots_clean/make_current_figures.py` | 主图 |
| `analysis_plots_clean/make_parameter_analysis.py` | 参数与残差分析图 |

实验输出和 checkpoint 位于本地 `experiments/`，不提交到 Git。关键 checkpoint 快照和 SHA256 清单位于：

```text
experiments/preserved_checkpoints_20261123/
```

## 11. 结论

当前 PG-MS-LR-LPO 已经形成一套可复现的一维参数化 PDE 神经算子框架。其优势集中在已知物理先验与神经残差结合的部分，而不是对所有 PDE 的统一求解能力。

最稳妥的论文定位是：

> 面向一维周期 PDE 的物理引导低秩潜空间算子，在约 501K 参数下具备参数匹配 FNO 的竞争力，并能通过物理平移与可学习残差扩展到参数化 advection。

在补齐公平 baseline、完整参数网格、效率分析和受限先验反演后，该工作适合作为应用计算或工程 AI 方向的中高水平期刊论文。
