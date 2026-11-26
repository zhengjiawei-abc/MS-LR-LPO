# PG-MS-LR-LPO 技术报告

**Physics-Guided Multi-Scale Low-Rank Latent PDE Operator**

版本：2026-11-27

代码仓库：<https://github.com/zhengjiawei-abc/PG-MS-LR-LPO>

## 摘要

PG-MS-LR-LPO 是一个面向参数化一维偏微分方程的物理引导神经算子。模型使用 CNN/上下文编码器和流形编码器建立低维潜空间，通过 LatentPDE 在潜空间中进行时间演化，再由多尺度 skip 和低秩二维网格解码器恢复全场解。当问题存在已知周期常速平移项时，模型加入解析平移基线，并通过可学习残差分支修正离散插值误差。

当前实验覆盖 PDEBench 的三个一维标量任务：Burgers、diffusion-sorption 和 advection。Burgers 与 diffusion-sorption 已完成三 seed 验证；advection 已覆盖官方全部八个参数：

```text
beta = 0.1, 0.2, 0.4, 0.7, 1.0, 2.0, 4.0, 7.0
```

实验结果表明，在参数预算接近时，PG-MS-LR-LPO 在绝大多数已测设置中优于 FNO，并在所有已测设置中优于参数匹配 DeepONet，包括已完成的高监督版本。Burgers 新增导数感知 PDE 修正器：它保留 11×11 场值分支，并并联一个只含 59 个参数的残差分支，显式输入归一化 `r = u_t + u u_x - nu u_xx`、`u_x` 与 `u_xx`。三 seed 结果为 Test L2 `0.022853 ± 0.000123`、总体 PDE residual `0.244626 ± 0.001483`，两项分别比参数匹配 PINO 的 `0.024261 / 0.253304` 低约 `5.8% / 3.4%`。但独立审计发现 Burgers interior residual 和四阶差分 residual 仍落后 PINO，因此只能声明总体指标领先。Advection β=1 新增解析平移后的输出端物理修正器，三 seed 结果为 Test L2 `0.005789 ± 0.000110`、PDE residual `0.102403 ± 0.000327`，分别比 PINO 的 `0.006619 / 0.104517` 低约 `12.5% / 2.0%`，且在 overall、boundary、interior 和四阶 residual 上均领先。历史修正器、外部泛化和统计审计详见 `docs/EXPERIMENT_AUDIT_20261127.md`。

2026-11-27 更新新增 Advection β=1 输出端物理修正器三 seed 结果，并把两个一维任务的 L2/PDE residual 双指标标准统一到 PINO 对比中。

本报告同时说明当前模型的能力边界、参数反演可辨识性、复现方法、已完成实验和条件性扩展。

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
  -> derivative-aware PDE refiner (optional)
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

### 2.8 导数感知 PDE 修正

Burgers 双指标配置在冻结 PG 主干上保留原 11×11 场值修正器，同时并联一个轻量导数分支。分支输入为：

```text
[r / characteristic_scale, u_x / rms(u_x), u_xx / rms(u_xx)]
r = u_t + u u_x - nu u_xx
```

网络结构为：

```text
3 -> 2, 1x1
SiLU
2 -> 1, 5x5
```

最后一个卷积零初始化，因此加载旧 11×11 checkpoint 时前向函数不变。分支只增加 59 个参数，使总参数从 `502,930` 变为 `502,989`，仍比匹配 PINO 少 8 个。训练采用两阶段路线：先以 `lambda_phys=0.03` 做 4K 步残差预热，再从该点以 `lr=2.5e-4` 训练导数分支 4K 步；dev L2 相对入口最多增加 2%，保存预算内最低 dev residual 检查点。

### 2.9 平移后的输出端物理修正

固定周期平移 `u_p(x,t)=u0(x-beta t)` 已有很低的函数 L2，但观测插值造成的时间边界残差较大。若修正器放在 `blend=0` 的平移融合之前，其输出会被解析基线完全覆盖，因此 Advection 使用独立的输出端分支：

```text
u_n = transport(u0)
features = [r / scale, u_x / rms(u_x), u_xx / rms(u_xx)]
u = u_n + ConvPDE(features)
```

分支为 `3 -> 8, 1x1 -> SiLU -> 8 -> 1, 5x5`，最后卷积零初始化，共 233 个可训练参数。训练时冻结固定平移主干，以归一化 Advection residual 为物理损失，dev L2 上限设为入口值的 `1.30` 倍，并仍在预算内选择最低 residual。三 seed 总参数为 `501,668`。

### 2.10 其他 PDE 物理残差修正

官方 diffusion-sorption HDF5 的样本属性包含 `D=5e-4`、`por=0.29`、`rho_s=2880`、`k_f=3.5e-4`、`n_f=0.874`。物理残差按 Freundlich 阻滞模型计算：

```text
R(c)c_t = D c_xx
R(c) = 1 + (1-por)/por * rho_s*k_f*n_f*c^(n_f-1)
```

只在不受左侧源项和右侧通量边界影响的 interior 上评分。

2D diffusion-reaction 已改为 PDEBench 官方反应项：

```text
u_t = Du lap(u) + u - u^3 - k - v
v_t = Dv lap(v) + u - v
Du = 1e-3, Dv = 5e-3, k = 5e-3
```

NS 没有保存压力场，因此使用压力梯度消去后的动量散度缺陷：

```text
div(u_t + u grad(u) - nu lap(u) - particles*force) = 0
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
| PG + 导数感知修正器 | 502,989 |
| PG + Advection 输出修正器 | 501,668 |
| 参数匹配 PINO，Burgers | 502,997 |

标准 PG 与跨 PDE FNO 的参数差约 `0.35%`。β=7 高频 PG 比 FNO 少 4 个参数。
Burgers 导数感知配置比匹配 PINO 少 8 个参数；Advection 输出修正器比对应 PINO
少 1,497 个参数。

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

DeepONet 支持 256 与 4096 两种查询点监督。Burgers、advection β=1 和 diffusion-sorption 的主结果使用 4096 查询点；其他 advection β 仍使用 256 查询点。高监督消融用于排除“DeepONet 落后只是因为每步采点太少”的解释。

PINO：

```text
backbone = matched FNO
data_loss = full-field MSE
physics_loss = normalized PDE residual MSE
training_steps = 24000
```

PINO 的物理权重先在 8K 步、seed 7 上进行选择。Burgers 使用 `1e-4`，advection β=1 使用 `5e-4`；最终结果使用三 seed 和 24K 步。物理残差采用时间边界复制、空间周期边界的中心差分，避免错误地对非周期时间窗做周期卷绕。

### 4.3 随机种子

- Burgers：seed 7、11、23。
- Diffusion-sorption：seed 7、11、23。
- Advection FNO/DeepONet：seed 7、11、23。
- advection 固定平移分支：确定性，无 seed 方差。
- β=2、4、7 的可学习残差：seed 7、11、23；三者共享早期 checkpoint，仅最终残差微调阶段独立。

## 5. 实验结果

### 5.1 Burgers

| 方法 | Test L2 |
|---|---:|
| PG，三 seed 均值 | **0.019767 ± 0.000008** |
| FNO，三 seed 均值 | 0.024588 ± 0.000375 |
| DeepONet-high，4096 点，三 seed 均值 | 0.253151 ± 0.005041 |

PG 相对 FNO 降低约 `19.6%`。Welch t 检验 `p=0.0020`。

### 5.2 Diffusion-sorption

| 方法 | Test L2 |
|---|---:|
| PG，三 seed 均值 | **0.003370 ± 0.000089** |
| FNO，三 seed 均值 | 0.701195 |
| DeepONet-high，4096 点，三 seed 均值 | 0.015725 ± 0.003566 |

PG 相对 FNO 降低约 `99.5%`，相对高监督 DeepONet 降低约 `78.6%`。

FNO 在 modes=16、24 和 width 重分配后仍停在 `0.701195`，说明异常不只是谱模态数量不足。该 baseline 仍需进一步检查任务条件和归一化。

### 5.3 Advection 参数扫描

FNO 和 DeepONet 为三 seed 均值；固定平移为确定性结果；可学习残差为最终阶段三 seed 均值。三组残差共享早期 checkpoint，因此这里检验的是最终微调阶段的稳定性，不是三次从零独立训练。

| beta | 固定平移 PG | 可学习残差 PG（最终阶段三 seed） | FNO | DeepONet |
|---:|---:|---:|---:|---:|
| 0.1 | **0.001154** | - | 0.003802 | 0.115324 |
| 0.2 | **0.001645** | - | 0.005010 | 0.132452 |
| 0.4 | **0.002547** | - | 0.006037 | 0.243269 |
| 0.7 | **0.003762** | - | 0.006435 | 0.326126 |
| 1.0 | **0.004543** | - | 0.006626 | 0.496381 |
| 2.0 | 0.008304 | **0.005043 ± 0.000002** | 0.006389 | 0.623089 |
| 4.0 | 0.017987 | **0.007183 ± 0.000001** | 0.007262 | 0.701016 |
| 7.0 | 0.033748 | 0.011357 ± 0.000008 | **0.009003** | 0.706937 |

DeepONet 在 β=1 行使用 4096 点高监督，其余 advection 参数使用 256 点标准配置。

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

### 5.5 残差分支三 seed 稳定性

为避免把单次 seed 波动误判为结构收益，β=2、4、7 均从各自选定的最终阶段入口继续微调 seed 7、11、23。三个 seed 共享入口 checkpoint，只有最终残差阶段重新初始化随机状态：

| beta | Seed 7 | Seed 11 | Seed 23 | Mean ± std | FNO | 结果 |
|---:|---:|---:|---:|---:|---:|---|
| 2.0 | 0.00504318 | 0.00504162 | 0.00504465 | **0.00504315 ± 0.00000152** | 0.006389 | PG 低 21.1% |
| 4.0 | 0.00718284 | 0.00718236 | 0.00718401 | **0.00718307 ± 0.00000085** | 0.007262 | PG 低 1.1% |
| 7.0 | 0.01135244 | 0.01135171 | 0.01136538 | 0.01135651 ± 0.00000769 | 0.009003 | PG 高 26.1% |

β=2 与 β=4 的优势在三 seed 下保持稳定；β=7 的失败也没有被 seed 选择掩盖。对应参数量分别为 `501,436`、`501,436` 和 `503,161`。标准差只说明最终微调阶段稳定，不能替代完整独立训练三 seed。

### 5.6 β=7 结构消融

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

表中结构消融使用 seed 7；三 seed 最终阶段均值为 `0.011357 ± 0.000008`。256 解码网格在约 500 步时耗时约 10 分钟，成本过高，未作为最终配置。

### 5.7 二维扩展结果

二维轻量标准保留 Fourier 线性潜空间推进和 multi-scale skip，但去掉非线性潜空间谱混合块。三 seed 结果如下：

| 任务 | 模型 | 参数量 | Test RelL2 | 物理指标 |
|---|---|---:|---:|---:|
| 2D diffusion-reaction | PG-Lite | 245,155 | **0.1053 ± 0.0006** | - |
| 2D diffusion-reaction | FNO | 302,602 | 0.2840 ± 0.0001 | - |
| 2D diffusion-reaction | DeepONet | 293,698 | 0.7986 ± 0.0036 | - |
| 2D incompressible NS | PG-Lite | 249,155 | **0.1742 ± 0.0035** | 散度 0.0366 ± 0.0020 |
| 2D incompressible NS | FNO | 302,650 | 0.3168 ± 0.0273 | 散度 0.0994 ± 0.0196 |
| 2D incompressible NS | DeepONet | 285,442 | 8.1238 ± 4.7577 | 散度 0.9379 ± 0.0722 |

去掉非线性谱块后，reaction 参数减少 `73,528` 且精度改善；NS 参数减少 `73,528`，速度误差从 `0.1816` 降到 `0.1742`。去掉 skip 会使两个任务显著退化，因此 skip 保留。

统一 GPU 基准中，PG-Lite 推理延迟约为 FNO 的 `14–15` 倍，峰值显存约 `231 MB` 对 `29 MB`。因此二维版本目前是精度优先，不是效率优先。详细协议、频谱和图表见 `docs/MULTIDIMENSIONAL_PDE.md`。

### 5.8 一维效率与高监督 DeepONet

Burgers、advection β=1 和 diffusion-sorption 上将 DeepONet 训练查询点从每步 256 增至 4096，保持其余参数和训练步数不变。三 seed 结果：

| 任务 | 256 查询点 | 4096 查询点 | 相对改善 |
|---|---:|---:|---:|
| Burgers | 0.2660 ± 0.0040 | 0.2532 ± 0.0050 | 4.8% |
| Advection beta=1 | 0.4970 ± 0.0540 | 0.4964 ± 0.0136 | 0.1% |
| Diffusion-sorption | 0.01779 ± 0.00305 | **0.01573 ± 0.00357** | 11.6% |

高监督只在 Burgers 与 diffusion-sorption 上带来有限改善，advection β=1 基本不变。三个任务上 DeepONet 仍显著落后 PG；Burgers 也仍落后 FNO。因此当前优势并非 DeepONet 监督不足造成的简单假象。

统一 1D 推理基准使用 batch size 32、输出 `256 x 101`：

| 模型 | 参数量 | 延迟 | 吞吐 | 峰值显存 |
|---|---:|---:|---:|---:|
| PG | 501,435 | 22.04 ms | 1,452 samples/s | 855 MB |
| FNO | 502,997 | 1.20 ms | 26,561 samples/s | 70 MB |
| DeepONet-high | 501,473 | 6.07 ms | 5,273 samples/s | 494 MB |

一维也是精度优先而不是效率优先，PG 推理延迟约为 FNO 的 18 倍。

PG 使用真实 checkpoint；FNO 和 DeepONet 的权重未在旧基线流程中保存，因此计时使用相同训练配置实例化的计算图。该基准适合比较前向复杂度，不等价于对已保存权重的逐文件复核。

### 5.9 PINO 对比

PINO 使用参数匹配 FNO 主干，并在全场数据损失上加入归一化 PDE residual loss。两个任务均使用 24K 步、batch size 64、相同划分和 seed 7/11/23。最终权重由 8K 步 seed-7 扫描选择，而不是使用测试集调参。

| 模型，Burgers | Test L2 | PDE residual | 参数量 |
|---|---:|---:|---:|
| PG 基础版 | 0.019767 ± 0.000009 | 0.4843 | 498,555 |
| PG + 3×3 | **0.018761 ± 0.000038** | 0.3718 | 498,860 |
| PG + 5×5 | 0.019342 ± 0.000159 | 0.3511 | 499,780 |
| PG + 9×9，两阶段 | 0.019457 ± 0.000074 | 0.289839 ± 0.001795 | 502,468 |
| PG + 11×11，两阶段 | 0.019554 ± 0.000084 | 0.284789 ± 0.002106 | 502,930 |
| PG + 导数感知，双指标标准 | **0.022853 ± 0.000123** | **0.244626 ± 0.001483** | 502,989 |
| PINO | 0.024261 ± 0.000393 | 0.253304 ± 0.003167 | 502,997 |
| 观测离散底噪 | - | 0.213228 | - |

基础 PG 对 PINO 的 Welch 检验为 `p=0.0025`；3×3、5×5、9×9、11×11 修正器对 PINO 的 Test L2 检验分别为 `p=0.00156`、`p=0.00058`、`p=0.00164`、`p=0.00157`。导数感知修正器对 PINO 的 Test L2 和 PDE residual 检验分别为 `p=0.0177` 和 `p=0.0259`，两项均在三 seed 上稳定低于 PINO。

| 模型，Advection β=1 | Test L2 | PDE residual | 参数量 |
|---|---:|---:|---:|
| PG 固定周期平移 | **0.004543 ± 0** | 0.126372 | 501,435 |
| PG + 输出端物理修正器 | 0.005789 ± 0.000110 | **0.102403 ± 0.000327** | 501,668 |
| PINO | 0.006619 ± 0.000068 | 0.104517 ± 0.000139 | 503,165 |
| 观测离散底噪 | - | 0.126372 | - |

Advection 输出修正器对 PINO 的 Test L2 与 residual Welch 检验分别为 `p=0.00098` 和 `p=0.00304`。它主动牺牲部分固定解析平移的极低 L2，换取低于 PINO 的物理残差，同时仍保留约 12.5% 的数据误差优势。

Burgers 物理修正器使用冻结主干和零初始化二维卷积残差分支。3×3 版本训练 305 个参数，5×5 版本训练 1,225 个参数，9×9 和 11×11 版本分别训练 3,913、4,375 个参数并使用两阶段训练。历史版本都在 dev 数据误差不超过入口值 105% 的约束下选择最低残差检查点。11×11 版本将残差从 `0.4843` 降到 `0.284789`，但尚未超过 PINO。导数感知版本把 dev 预算收紧到 2%，新增 59 个参数学习归一化 PDE 残差及一、二阶空间导数，三 seed 平均残差降到 `0.244626`，比 PINO 低约 `3.4%`，同时 Test L2 比 PINO 低约 `5.8%`。

Burgers 上导数感知修正器仍比观测残差底噪 `0.213228` 高，但已经低于 PINO。Advection 上 PINO 的残差低于观测底噪，说明更平滑的解可以压低残差，但函数误差仍高于 PG。解析平移 PG 的残差与观测底噪完全一致，因此不能仅凭“残差更低”判断 PINO 更准确。

当前可支持的结论是：Burgers 学习型修正器的总体 Test L2/residual 低于 PINO，但 interior 和四阶 residual 仍落后；Advection β=1 在 overall、boundary、interior 和高阶 residual 上均低于 PINO。该结论不外推到 diffusion-sorption 或二维任务。

### 5.10 主训练残差约束

`train_coord_attention.py` 支持在 full-grid 主训练损失中加入归一化 PDE residual，并可设置 `physics_start_step` 和 `physics_warmup`。测试了从零全程约束，以及从已收敛数据-only checkpoint 晚启用约束两种课程：

| 模型，Burgers | Test L2 | PDE residual | 参数量 |
|---|---:|---:|---:|
| PG 数据-only，三 seed | **0.019767 ± 0.000009** | 0.4843 | 498,555 |
| 从零全程加 residual，seed 7 | 0.027682 | 0.4879 | 501,435 |
| 晚启用 residual，三 seed | 0.019183 ± 0.000037 | 0.4470 ± 0.0001 | 498,555 |
| PG 11×11 后处理修正器 | 0.019554 ± 0.000084 | **0.284789** | 502,930 |
| PG 导数感知修正器，三 seed | 0.022853 ± 0.000123 | **0.244626 ± 0.001483** | 502,989 |
| PINO，三 seed | 0.024261 ± 0.000393 | 0.253304 ± 0.003167 | 502,997 |

全程主干残差约束明显拖慢数据收敛，测试 PDE residual 也没有下降。原因是 PG 的潜空间和低分辨率解码器需要先形成稳定的数据表征，再接收差分残差梯度；过早反向传播会同时扰动编码器、潜空间推进和解码器。

从已收敛的数据-only checkpoint 晚启用主训练残差更有效：三 seed 数据误差降到 `0.019183 ± 0.000037`，残差降到 `0.4470 ± 0.0001`，相对数据-only 分别改善约 `3.0%` 和 `7.7%`。后处理 11×11 修正器仍把残差降到 `0.284789`；导数分支进一步降到 `0.244626`，并保持 Test L2 低于 PINO。因此当前仍采用“数据表征预训练 + 输出端物理修正”的两阶段路线。

### 5.11 残差可信度、统计与外部泛化

Burgers 的总体 residual 优势主要来自时间边界：

| residual | PG | PINO | 观测数据 |
|---|---:|---:|---:|
| overall | **0.244643** | 0.253297 | 0.213228 |
| boundary | **0.634041** | 0.702292 | 0.492399 |
| interior | 0.227397 | **0.221918** | 0.167455 |
| fourth-order | 1.372452 | **1.367258** | 1.375831 |

因此 Burgers 只能声明官方总体 residual 更低，不能声明内部或高阶物理一致性全面胜出。Advection β=1 则四项均低于 PINO。

Seed-7 common 1000 样本的 paired bootstrap 与 Holm 校正在四个总体指标上均显著。统计、公式和外部数据细节见 `docs/EXPERIMENT_AUDIT_20261127.md`。

外部验证暴露了两个明确边界：

- Burgers `Nu=0.01` 模型零样本迁移到官方 `Nu=0.1` 时，PG/FNO L2 均约 `0.39`，当前模型不具备跨粘度泛化。
- Advection β=1 输出修正器可零样本迁移到 β≤0.7；β≥2 应使用专用残差分支，β=7 仍落后 FNO。

Diffusion-sorption 参数和 Freundlich residual 已恢复，但当前 PG residual `0.999855` 高于观测数据 `0.531420`，未通过物理约束。二维 diffusion-reaction 的正确 residual 为真值 `0.009507`、PG `0.034724`、FNO `0.136032`。NS 压力消去后的动量散度缺陷为真值 `1.305054`、PG `1.136847`、FNO `1.095111`，由于观测数据本身不接近零，只作为诊断。

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

### 6.1 带先验范围的反演

为验证“物理先验可以恢复可辨识性”，对每个真值 β 设置 ±20% 的候选范围，仅使用 8 个时间快照，并加入 0%、1%、5%、10% 相对观测噪声。每个参数使用 200 个测试样本：

| 噪声 | MAPE | MAE | 5% 内比例 |
|---:|---:|---:|---:|
| 0% | 0.036% | 0.00069 | 100% |
| 1% | 0.036% | 0.00071 | 100% |
| 5% | 0.040% | 0.00072 | 100% |
| 10% | 0.049% | 0.00077 | 100% |

该结果只说明：在明确物理先验和有限候选范围内，advection β 可以被稳定恢复。它不推翻无先验全局反演的不适定性，也不等价于多维 NS 参数反演。

### 6.2 二维 NS 参数反演数据审计

当前官方 NS 文件在属性中固定 `NU=0.01`、`force_scale=0.15`、`force_smoothness=3.0` 和 `scale=0.4`。四条轨迹的变化来自初始速度与随机外力实现，没有不同粘度或外力强度对应的参数标签。

因此，当前数据不能构造有真值支撑的二维 NS 标量参数反演。若人工生成一个标量再以此拟合，只能算自建合成任务，不能写成官方 PDEBench 反演结果。补齐该实验需要额外参数化轨迹，并按参数取值划分训练与测试。

## 7. 当前能力边界

### 已支持

- 一维周期标量 PDE 数据学习。
- 参数化 advection 前向预测。
- 已知参数下的物理引导残差修正。
- 多参数、三 seed baseline 对比。
- 二维 diffusion-reaction 与力条件二维不可压缩 NS 前向预测。
- checkpoint 保护、指标记录和经验复现。

### 未支持或未验证

- 多变量可压缩 NS。
- 全分辨率二维 NS、压力场、湍流和复杂几何。
- 无先验通用参数反演。
- 时变边界、源项和复杂几何。
- Burgers 跨粘度零样本泛化；官方 `Nu=0.1` 外部验证明显失败。
- Advection β≥2 的 β=1 修正器零样本迁移；这些区间需专用分支。
- Diffusion-sorption 达到观测数据 residual 水平的物理一致性。
- NS 的绝对动量平衡；当前压力消去指标仍受粗网格和交错量近似限制。
- 生产级推理服务和不确定性保证。

## 8. 完成状态与条件性扩展

已完成：

1. 高监督 DeepONet：二维参数匹配版本和三个一维任务的 4096 点版本。
2. 官方 advection 八个参数值；对应原始 HDF5 均通过官方 MD5 校验。
3. β=2、4、7 残差分支最终阶段三 seed 稳定性检查。
4. 训练时间、推理延迟、显存和吞吐量表：一维与二维。
5. 模块消融：二维任务和 β=7 残差结构均已完成。
6. 一维 advection 带 ±20% 先验范围的逆问题；二维 NS 文件参数边界审计。
7. Burgers 与 advection β=1 的参数匹配 PINO 对比和物理权重扫描。
8. Burgers 轻量物理残差修正器三 seed；3×3、5×5、9×9、11×11 历史版本和导数感知版本已完成。导数分支新增 59 个参数，在总参数比 PINO 少 8 个的条件下同时降低 Test L2 和 PDE residual。
9. PG 主训练循环已支持 PDE residual、物理权重预热和延迟启用；从零训练为负结果，晚启用三 seed 为小幅正结果。
10. Advection β=1 解析平移后的输出端物理修正器三 seed 已完成；新增 233 个参数，同时低于 PINO 的 Test L2 和 PDE residual。
11. Burgers/Advection 的 boundary、interior、四阶差分和谱残差审计已完成；Burgers interior 反例已披露。
12. Seed-7 1000 样本 paired bootstrap、Wilcoxon 与 Holm 多重比较校正已完成。
13. 官方 Burgers `Nu=0.1` 外部 test 与 Advection 八 β 零样本迁移已完成，泛化失败边界已记录。
14. Diffusion-sorption Freundlich residual、二维 diffusion-reaction 官方残差和 NS 压力消去动量散度残差已完成，并记录各自限制。

条件性扩展：

- 若要把“最终阶段三 seed”升级为“严格从零独立训练三 seed”，需再完成九次完整训练。这是稳健性增强，不是当前结果声明的必要前提。
- 若要报告二维 NS 标量参数反演，需额外获取或生成按 `NU` / `force_scale` 参数化的轨迹。
- 若要声称通用 PDE 算子，仍需增加 1D CFD、压力场和全分辨率二维 NS；当前论文声明不覆盖这一范围。

## 9. 可声明与不可声明

可声明：

> 在 PDEBench 一维标量任务和项目内参数匹配基线上，PG-MS-LR-LPO 在绝大多数已测设置中优于 FNO，并在所有已测设置中优于参数匹配 DeepONet，包括已完成的高监督版本。Burgers 学习型修正器取得低于 PINO 的 Test L2 和总体 residual；Advection β=1 在 overall、boundary、interior 与四阶 residual 上均低于 PINO。样本级配对统计经 Holm 校正后支持四个总体差异。

不可声明：

> 已全面超过官方 PDEBench 的 FNO/DeepONet/PINO，或已经在所有任务上同时取得更低数据误差和更低 PDE 残差，或已经解决完整二维压力场 NS、湍流、逆问题和参数反演。

PINO 对比的正确表述是：Burgers 导数感知修正器总体 Test L2/PDE residual 为 `0.022853 / 0.244626`，低于 PINO 的 `0.024261 / 0.253304`，但 interior residual 和四阶 residual 仍落后；Advection β=1 输出修正器为 `0.005789 / 0.102403`，四项 residual 审计均优于 PINO。该声明不外推到其他 PDE。

外部验证的正确表述是：Burgers 当前不具备跨粘度零样本泛化；Advection β=1 修正器只可零样本用于 β≤0.7；diffusion-sorption 和完整 NS 尚未达到可声明的物理残差优势。

同时不能把“共享早期 checkpoint、仅最终阶段改变 seed”写成“从零独立训练三 seed”。当前正确表述是最终残差微调阶段三 seed。

## 10. 复现入口

主要脚本：

| 路径 | 作用 |
|---|---|
| `code/coord_attention_operator.py` | PG-MS-LR-LPO、残差融合、场值与导数物理修正器 |
| `code/train_coord_attention.py` | 主训练、PDE residual 课程、里程碑保存和 checkpoint 恢复 |
| `code/baselines_1d_burgers.py` | FNO/PINO/DeepONet |
| `code/run_cross_pde_matrix.py` | 跨 PDE 训练矩阵 |
| `code/run_residual_seed_queue.py` | β=2、4、7 最终残差阶段 seed 队列 |
| `code/run_advection_beta_queue.py` | β=0.1、0.4 固定平移与匹配基线队列 |
| `code/run_pino_seed_queue.py` | Burgers 与 advection 的参数匹配 PINO 三 seed |
| `code/run_pg_physics_refiner_queue.py` | Burgers 轻量物理残差修正器三 seed |
| `code/run_pg_derivative_refiner_queue.py` | Burgers 残差预热与导数感知修正器两阶段三 seed |
| `code/run_pg_advection_output_refiner_queue.py` | Advection β=1 输出端物理修正器三 seed |
| `code/finetune_pde_residual.py` | 可配置残差微调、导数分支、数据预算与最优检查点选择 |
| `code/pde_residual_1d.py` | 统一的 1D 归一化 PDE 残差与导数特征 |
| `code/run_pg_main_physics_queue.py` | 数据-only 预训练后晚启用主训练残差三 seed |
| `analysis_plots_clean/compile_advection_eight.py` | 官方八个 β 三方结果汇总 |
| `analysis_plots_clean/make_pino_comparison.py` | PG/PINO 数据精度与残差对比 |
| `code/analyze_residual_credibility.py` | boundary/interior/高阶/谱残差审计 |
| `code/analyze_paired_statistics.py` | 配对 bootstrap、Wilcoxon 与 Holm 校正 |
| `code/analyze_2d_physics.py` | 修正 reaction 与压力消去 NS residual |
| `code/evaluate_burgers_external_viscosity.py` | 外部粘度零样本验证 |
| `code/evaluate_advection_refiner_generalization.py` | 八 β 零样本迁移 |
| `code/evaluate_advection_beta_inference.py` | β 可辨识性诊断 |
| `datasets/download_parallel.py` | 官方大文件分片续传与 MD5 校验 |
| `datasets/http_hdf5.py` | curl range 远程 HDF5 reader |
| `datasets/fetch_burgers_external.py` | 外部 Burgers test 抽样 |
| `datasets/pdebench_advection_sources.json` | 官方八个 advection 文件来源 |
| `datasets/pdebench_group_to_parquet.py` | 官方 HDF5 转换 |
| `analysis_plots_clean/make_current_figures.py` | 主图 |
| `analysis_plots_clean/make_parameter_analysis.py` | 参数与残差分析图 |

实验输出和 checkpoint 位于本地 `experiments/`，不提交到 Git。关键 checkpoint 快照和 SHA256 清单位于：

```text
experiments/preserved_checkpoints_20261123/
```

## 11. 结论

当前 PG-MS-LR-LPO 已经形成一套可复现的一维参数化 PDE 神经算子框架。其优势集中在已知物理先验与神经残差结合的部分，而不是对所有 PDE 的统一求解能力。官方 advection 八个 β 已全部补齐；残差分支最终阶段的三 seed 结果确认 β=2、4 的优势稳定存在，而 β=7 仍落后 FNO。Advection β=1 输出修正器以 `0.005789 / 0.102403` 在总体、边界、内部和高阶 residual 上均低于 PINO。Burgers 的总体 Test L2/residual 为 `0.022853 / 0.244626`，低于 PINO 的 `0.024261 / 0.253304`，但 interior 和四阶 residual 仍落后；Burgers 模型也不能零样本迁移到其他粘度。把残差约束前移到主训练阶段的首轮实验未能优于输出端修正，说明当前数据-only 预训练加轻量物理后处理仍是更合适的两阶段方案。

最稳妥的论文定位是：

> 面向一维周期 PDE 的物理引导低秩潜空间算子，在约 501K 参数下具备参数匹配 FNO 的竞争力；Advection β=1 在三个 seed 上于总体、内部和高阶残差上优于 PINO。Burgers 的总体指标低於 PINO，但内部与跨粘度泛化仍有明确边界。

二维扩展进一步表明，低秩潜空间加多尺度 skip 的轻量配置可在 245K–249K 参数下显著优于参数匹配 FNO/DeepONet；但推理效率仍明显落后于 FNO，完整压力场 NS 尚未完成。

在已补齐公平 baseline、完整官方 advection 参数网格、效率分析和受限先验反演后，该工作适合作为应用计算或工程 AI 方向的中高水平期刊论文；二维 NS 标量参数反演仍需要额外参数化数据。
