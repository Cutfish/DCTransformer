# DCTransformer 模型架构详解

> **论文**: Dual Cross Transformer for Hyperspectral Image Super-Resolution
> **核心思想**: 利用双向 Cross-Attention 在 HSI 和 MSI 之间进行特征交互，结合 Swin Transformer Block 增强全局依赖建模

---

## 1. 符号定义与问题形式化

### 输入输出

| 符号 | 形状 | 含义 |
|------|------|------|
| **Y** (HR-MSI) | $\mathcal{Y} \in \mathbb{R}^{W \times H \times s}$ | 高空间分辨率、低光谱分辨率的多光谱图像 |
| **Z** (LR-HSI) | $\mathcal{Z} \in \mathbb{R}^{w \times h \times S}$ | 低空间分辨率、高光谱分辨率的高光谱图像 |
| **X** (HR-HSI/GT) | $\mathcal{X} \in \mathbb{R}^{W \times H \times S}$ | 目标：同时具有高空间和高光谱分辨率的图像 |

- $l = W/w = H/h$: 超分辨率因子 (scale factor)
- $s$: HR-MSI 的光谱波段数（通常为 3，RGB）
- $S$: LR-HSI 的光谱波段数（通常为 31）

### 观测模型
$$ Y = X R, \quad Z = D X $$

其中 $R \in \mathbb{R}^{S \times s}$ 为光谱响应函数(SRF)，$D \in \mathbb{R}^{wh \times WH}$ 为降质算子（卷积+下采样）。

---

## 2. 整体网络架构（对应 Fig.1(a)）

```
输入: Z(LR-HSI) ──→ Bicubic上采样(X⁰) → Conv(浅层特征F_X⁰) ─┐
                                                            │
输入: Y(HR-MSI) ──→ Conv(浅层特征F_Y) ─────────────────────┤
                                                            ↓
                                              ┌─── RDCTG-1 ───┐
                                              │   (密集连接)    │
                                              ├─── RDCTG-k ───┤  K=3个RDCTG
                                              │   (密集连接)    │
                                              └─── RDCTG-K ───┘
                                                            │
                                              Concat(F_X⁰, F_X^K)
                                                            ↓
                                                    Conv 3×3 + F_X⁰ (残差)
                                                            ↓
                                                      Conv 3×3 (重建)
                                                            ↓
                                                     输出: X̂ (HR-HSI)
```

**关键设计 — 密集连接 (Dense Connection)**:
$$ F_X^i = \text{Conv}\left(\text{Concat}(F_X^0, ..., F_X^{i-1}, \text{RDCTG}_i(F_X^{i-1}))\right), \quad i \in \{1,...,K\} $$

每个 RDCTG 的输出都与之前所有层的特征拼接，促进特征复用和梯度传播。

---

## 3. Residual Dual Cross Transformer Group (RDCTG)

对应 Fig.1(b)，论文公式 (9)-(10):

$$ F_{X}^{i,j} = \text{DCATB}(F_{X}^{i,j-1}), \quad j \in \{1,...,N\} $$

$$ F_{X}^{i,\text{out}} = \text{Conv}(F_{X}^{i,N}) + F_{X}^{i,0} $$

**结构**: N 个 DCATB 串联 + 尾部 Conv 残差连接
- N = 6 (DCATB数量，代码中 `depths=[6,6,6]`)
- 每个 DCATB 内部包含 M-HCA、H-MCA 和 STB 三个子模块

---

## 4. Dual Cross Attention Transformer Block (DCATB)

对应 Fig.2，论文公式 (11)-(20)，**这是整个模型最核心的模块**：

### 4.1 整体数据流
```
输入: F_X^{i,j-1}, F_Y
         │
    ┌────┴────┐
    ↓         ↓
[M-HCATB]  [H-MCATB]        ← 双向交叉注意力
(MS→HS)    (HS→MS)
    │         │
    └────┬────┘
         ↓ Element-wise Addition
      [STB]                    ← Swin Transformer自注意力
         │
    输出: F_X^{i,j}
```

数学表达：
$$ F_{X}^{i,j} = \text{STB}\left(\text{H-MCATB}(F_{X}^{i,j-1}, F_Y) + \text{M-HCATB}(F_{X}^{i,j-1}, F_Y)\right) $$

### 4.2 M-HCATB (Multi-spectral → Hyperspectral Cross Attention)

**目的**: 将 HR-MSI 的高频纹理信息注入到 HSI 特征中

**Query/Key/Value 分配策略** (论文公式12):
- **Q** ← 来自 **FY** (HR-MSI特征): $Q = P_m^Y W_Q$  ← 提供查询"需要什么纹理"
- **K, V** ← 来自 **FX** (HSI特征): $K = P_m^X W_K$, $V = P_m^X W_V$  ← 提供被检索的内容

**Cross-Attention 计算** (论文公式13):
$$ \text{CA} = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}} + B\right) V $$

其中 B 为可学习的相对位置编码(Relative Position Bias)

**后处理** (论文公式14-15):
$$ F_{X}^{i,j-1} = \text{M-HMCA}(\text{LN}(F_Y), \text{LN}(F_{X}^{i,j-1})) + F_{X}^{i,j-1} $$
$$ F_{X}^{i,j-1} = \text{MLP}(\text{LN}(F_{X}^{i,j-1})) + F_{X}^{i,j-1} $$

### 4.3 H-MCATB (Hyperspectral → Multi-spectral Cross Attention)

**目的**: 将 HSI 的丰富光谱信息传递给 MSI 特征

**Query/Key/Value 分配策略** (论文公式16):
- **Q** ← 来自 **FX** (HSI特征): $Q = P_m^{\hat{X}} W_Q$
- **K, V** ← 来自 **FY** (MSI特征): $K = P_m^{\hat{Y}} W_K$, $V = P_m^{\hat{Y}} W_V$

注意：这里 Q/K/V 的分配与 M-HCATB **相反**

### 4.4 STB (Swin Transformer Block)

**目的**: 通过 Window-based Self-Attention 捕获融合后特征的**长程空间依赖关系**

标准 Swin Transformer 结构:
1. LayerNorm → Window Partition → W-MSA/SW-MSA → 反窗口合并
2. 残差连接
3. LayerNorm → MLP → 残差连接

**Shifted Window 机制**: 相邻两个 DCATB 使用 shift_size=0 和 shift_size=window_size//2 交替，实现跨窗口的信息交流。

---

## 5. 代码模块与论文对应关系

| 论文符号 | 代码中的类名 | 文件 | 说明 |
|---------|------------|------|------|
| 整体模型 | `DCT` | dual.py:987 | 对应 Fig.1(a) 完整架构 |
| 浅层特征提取 | `headX`, `headY` | dual.py:993-998 | 公式(4)(5), 3×3 Conv |
| RDCTG | `RSTB` | dual.py:714 | 公式(9)(10), 包含 BasicLayer + 残差 |
| DCATB | `DualCrossTransformerBlock` | dual.py:596 | 公式(11), 核心双交叉注意力块 |
| M-HCATB | `CrossSwinTransformerBlock` (cross1) | dual.py:265 | 公式(12)(13)(14)(15), MS→HS |
| H-MCATB | `CrossSwinTransformerBlock` (cross2) | dual.py:265 | 公式(16)(17), HS→MS (Q/K/V互换) |
| STB | `SwinTransformerBlock` | dual.py:412 | 公式(18)(19)(20), 自注意力 |
| M-HCA 注意力 | `CrossWindowAttention` | dual.py:62 | 窗口内交叉注意力, Q来自y, KV来自x |
| Self-Attention | `WindowAttention` | dual.py:165 | 窗口内自注意力 |
| Patch嵌入/逆嵌入 | `PatchEmbed` / `PatchUnEmbed` | dual.py:790/833 | 图像↔序列转换 |
| 密集连接+重建 | `DCT.forward()` | dual.py:1008 | 公式(6)(7), 3级级联+残差 |

---

## 6. 关键超参数

| 参数 | 默认值 | 论文说明 |
|------|--------|----------|
| `n_feats` (C) | 180 | 特征通道数 |
| `depths` | [6,6,6] | 每个 RSTB 中 DCATB 数量 N=6 (但实际只有1个RSTB, 取depths[0]=6) |
| `num_heads` | [6,6,6] | 多头注意力的头数 |
| `window_size` | 8 | 窗口大小 (Swin默认为7, 此处用8) |
| `mlp_ratio` | 2 | MLP隐藏层倍数 (Swin默认为4, 此处为2以减少参数) |
| `upscale_factor` | 8 | 超分辨率倍率 |
| `K` (RDCTG数量) | 3 | 代码中通过3次 body 调用实现 |
| `drop_path_rate` | 0.1 | Stochastic Depth 丢弃率 |
| 训练 epochs | 200 | Adam, lr=1e-4, batch=8 |

---

## 7. 训练配置细节 (Section 3.5)

| 配置项 | 值 |
|--------|-----|
| 优化器 | Adam |
| 初始学习率 | 1e-4 |
| 学习率衰减 | MultiStepLR @ [100,150,175,190,195], gamma=0.5 |
| Batch Size | 8 |
| 总迭代次数 | 250,000 (每epoch 2500 iterations) |
| Loss 函数 | L1 Loss |
| 数据增强 | 随机旋转(0~270°)、随机水平/垂直翻转 |
| GPU | 2 × NVIDIA GTX3090 |
| 分布式训练 | DistributedDataParallel (DDP) |
| TensorBoard | 有记录 loss 和 psnr |

---

## 8. 数据流总结

```
Z (LR-HSI) [B,S,w,h]          Y (HR-MSI) [B,s,W,H]
    │                               │
    ▼ Bicubic(×l)                  │
    X⁰ [B,S,W,H]                   │
    │                               │
    ▼ Conv 3×3                     ▼ Conv 3×3(两层)
    F_X⁰ [B,C,W,H]                 F_Y  [B,C,W,H]
    │                               │
    ╰───────┬───────────────────────╯
            │
            ▼
     ┌─ RDCTG-1 ─┐  (dense concat F_X⁰)
     │  ├ DCATB_1 │
     │  ├ DCATB_2 │
     │  ├ ...     │  N=6个DCATB
     │  ├ DCATB_6 │
     │  └ Conv + residual
     │       │
     │  Concat(F_X⁰, out) → Conv → F_X¹
     │       │
     ├─ RDCTG-2 ─┐  (dense concat F_X⁰,F_X¹)
     │  ...同上...
     │       │
     │  Concat(F_X⁰,F_X¹,out) → Conv → F_X²
     │       │
     ├─ RDCTG-3 ─┐  (dense concat F_X⁰,F_X¹,F_X²)
     │  ...同上...
     │       │
     │  Concat(F_X⁰,F_X¹,F_X²,out) → Conv → F_X³
     │
     ▼
   F_X⁰ + F_X³  (全局残差)
        │
        ▼ Conv 3×3
      X̂ [B,S,W,H]  (预测的 HR-HSI)
```
