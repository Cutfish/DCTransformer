import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """[论文 Fig.2] 多层感知机模块 (MLP/FFN)
    用于 Transformer Block 中的前馈网络部分。

    结构: Linear → GELU → Dropout → Linear → Dropout

    Args:
        in_features (int): 输入特征维度
        hidden_features (int): 隐藏层维度，默认为 in_features
        out_features (int): 输出维度，默认为 in_features
        act_layer: 激活函数，默认 GELU
        drop (float): Dropout 比率

    输入:
        x: [B, L, C] 序列特征
    输出:
        x: [B, L, C] 变换后的特征
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 升维投影
        self.act = act_layer()                               # GELU激活
        self.fc2 = nn.Linear(hidden_features, out_features)   # 降维恢复
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 输入: [B, L, C_in], 输出: [B, L, C_out]
        x = self.fc1(x)       # [B, L, C_hidden]
        x = self.act(x)       # GELU 非线性激活
        x = self.drop(x)      # 正则化
        x = self.fc2(x)       # [B, L, C_out]
        x = self.drop(x)      # 正则化
        return x


def window_partition(x, window_size):
    """[Swin Transformer] 将特征图分割为不重叠的局部窗口

    Args:
        x: [B, H, W, C] 输入特征图
        window_size (int): 窗口大小

    Returns:
        windows: [num_windows*B, window_size, window_size, C] 分割后的窗口
            num_windows = (H/window_size) * (W/window_size)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # 重排为 [B, nW_h, nW_w, ws, ws, C] → 展平窗口维度 → [B*nW, ws, ws, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """[Swin Transformer] 将窗口合并回特征图 (window_partition的逆操作)

    Args:
        windows: [num_windows*B, window_size, window_size, C] 窗口特征
        window_size (int): 窗口大小
        H (int): 原始图像高度
        W (int): 原始图像宽度

    Returns:
        x: [B, H, W, C] 恢复的特征图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # [B*nW, ws, ws, C] → [B, nW_h, nW_w, ws, ws, C] → [B, H, W, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class CrossWindowAttention(nn.Module):
    r"""[论文 Fig.2 M-HCA/H-MCA 部分] 窗口化多头交叉注意力 (Cross-Attention)

    ⭐ 这是 DCTransformer 的核心创新之一：Cross Attention 而非 Self Attention
    支持 Shifted 和 Non-shifted 两种窗口模式

    关键区别于标准 W-MSA:
    - Q 来自一个输入 (y)，K/V 来自另一个输入 (x)
    - 实现了论文公式(13): CA = softmax(QK^T/sqrt(d) + B) * V
    - QKV分离: qkv1(x) → K,V ; qkv2(y) → Q

    Args:
        dim (int): 输入通道数 C
        window_size (tuple[int]): 窗口大小 (Wh, Ww)
        num_heads (int): 注意力头数
        qkv_bias (bool): 是否给Q/K/V加偏置
        qk_scale: 手动设置缩放因子（默认 head_dim^-0.5）
        attn_drop / proj_drop: Dropout 比率

    forward 输入/输出:
        x: [num_windows*B, N, C] - HSI特征，用于生成 K, V
        y: [num_windows*B, N, C] - MSI特征，用于生成 Q
        mask: 可选的注意力掩码 [nW, Wh*Ww, Wh*Ww]
        return: [num_windows*B, N, C] - 交叉注意力输出特征
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # 缩放因子 = 1/sqrt(d_k)

        # 可学习的相对位置编码表 (2Wh-1)*(2Ww-1) × nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 预计算相对位置索引（固定不变，注册为buffer）
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # ⭐ 核心：分离的QKV投影
        self.qkv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)   # 对x投影 → K+V
        self.qkv2 = nn.Linear(dim, dim * 1, bias=qkv_bias)   # 对y投影 → Q (注意只有dim*1!)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)                        # 输出投影
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)  # 初始化位置编码
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, mask=None):
        """[论文公式(13)] 交叉注意力前向传播

        实现流程: x→K,V ; y→Q ; Attention(Q,K,V) → 输出

        Args:
            x: [nW*B, N, C] - 被检索的特征 (HSI特征)，用于 K, V
            y: [nW*B, N, C] - 查询特征 (MSI特征)，用于 Q
            mask: 可选注意力掩码 [nW, N, N]

        Returns:
            [nW*B, N, C] - 交叉注意力增强后的特征
        """
        B_, N, C = x.shape
        # x → K, V (HSI提供内容): [nW*B,N,C] → [2,nW*B,nH,N,d]
        kv = self.qkv1(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # y → Q (MSI提供查询): [nW*B,N,C] → [1,nW*B,nH,N,d]
        qq = self.qkv2(y).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qq[0]

        # 缩放 + 注意力分数: Q @ K^T / sqrt(d) → [nW*B, nH, N, N]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 加入相对位置编码 B (论文公式13中的+B)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, N, N]
        attn = attn + relative_position_bias.unsqueeze(0)

        # Softmax 归一化 + Dropout
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 加权求和: Attention @ V → 投影输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # [nW*B, nH, N, d] → [nW*B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class WindowAttention(nn.Module):
    r"""[论文 Fig.2 STB 部分] 窗口化多头自注意力 (Self-Attention, 非Cross)

    与 CrossWindowAttention 的区别: Q/K/V 全部来自同一个输入 x
    这是 Swin Transformer Block (STB) 中使用的标准自注意力

    forward 输入/输出:
        x: [num_windows*B, N, C] - 输入特征，同时用于 Q, K, V
        mask: 可选掩码
        return: [num_windows*B, N, C] - 自注意力输出
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 可学习相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 预计算相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 标准的QKV合并投影 (与Cross版本不同!)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops



class CrossSwinTransformerBlock(nn.Module):
    r"""[论文 Fig.2 上半部分 M-HCA] 交叉 Swin Transformer Block (用于M-HCATB或H-MCATB)

    这是 DCATB 中两个交叉注意力分支之一:
    - cross1 (M-HCATB): Q来自FY(MSI), K,V来自FX(HSI) - 将MS纹理注入HSI
    - cross2 (H-MCATB): Q来自FX(HSI), K,V来自FY(MSI) - 将HS光谱信息传递给MS

    结构: LN → [Window Partition → CrossWindowAttention → Window Reverse] → 残差 → LN → MLP → 残差
    支持 Shifted Window 机制以实现跨窗口交互

    forward 输入/输出:
        x: [B, L, C] - HSI特征序列 (L=H*W)
        y: [B, L, C] - MSI特征序列
        x_size: (H, W) 空间分辨率，用于window partition
        return: [B, L, C] - 交叉注意力增强后的HSI特征
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            dim (int): 特征维度/通道数, 当前值=180
                       (即 Q/K/V 每个头的维度 = dim / num_heads = 180/6 = 30)
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率, 当前值=(16, 16)
                                               (来自 img_size(64)/patch_size(4) = 16)
            num_heads (int): 多头注意力的头数, 当前值=6
                             (每头 dim_head=180//6=30, 6个头并行计算不同子空间的注意力)
            window_size (int): 窗口大小, 当前值=8
                               (将 16×16 特征图划分为多个 8×8 的局部窗口, 在窗口内做注意力,
                                复杂度从 O((HW)²) 降为 O(W²×H×W))
            shift_size (int): 窗口偏移量, 当前值=0 或 4(交替)
                              (奇数层shift=0: 规则窗口; 偶数层shift=4: 窗口偏移半个window_size,
                               实现跨窗口信息交互, SWIN Transformer 的核心设计)
            mlp_ratio (float): MLP 隐藏层维度倍率, 当前值=2.0
                               (MLP隐藏层维度 = dim * mlp_ratio = 180*2 = 360,
                                FFN结构: Linear(180→360)→GELU→Linear(360→180), 含残差连接)
            qkv_bias (bool): Q/K/V 线性投影是否加偏置, 当前值=True
            qk_scale (float | None): 注意力缩放因子, 当前值=None (自动用 head_dim**-0.5)
                                     (Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, 控制点积尺度)
            drop (float): 全局 Dropout 率, 当前值=0.0
            attn_drop (float): 注意力权重 Dropout 率, 当前值=0.0
                               (在 softmax 后对 attention weights 做 dropout, 防止过拟合)
            drop_path (float): Stochastic Depth 随机深度丢弃率, 当前值=0.1(线性递增分配到各层)
                              (训练时随机丢弃整个 block 的输出, 正则化 + 节省推理时延)
            act_layer: MLP 激活函数, 当前=GELU (Gaussian Error Linear Unit)
            norm_layer: 归一化层, 当前=LayerNorm (对最后一个维度C做归一化)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # LayerNorm for x (HSI) and y (MSI) 分别做归一化
        self.norm1 = norm_layer(dim)   # 对 x 归一化
        self.norm2 = norm_layer(dim)   # 对 y 归一化

        # ⭐ 核心交叉注意力模块 (Q来自y, KV来自x 或 反之)
        self.attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 预计算 shifted window 的 attention mask
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, y, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        y = self.norm2(y)
        y = y.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            shifted_y = y

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        y_windows = window_partition(shifted_y, self.window_size)  # nW*B, window_size, window_size, C
        y_windows = y_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, y_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops




class SwinTransformerBlock(nn.Module):
    r"""[论文 Fig.2 STB 绿色部分] 标准 Swin Transformer Block (Self-Attention)

    用于 DCATB 的最后阶段：对 M-HCA 和 H-MCA 融合后的特征做自注意力增强
    捕获融合后特征的内部长程空间依赖关系

    结构: LN → [Window Partition → W-MSA/SW-MSA → Window Reverse] → 残差 → LN → MLP → 残差

    forward 输入/输出:
        x: [B, L, C] - 输入序列 (来自 M-HCA + H-MCA 的加和)
        x_size: (H, W) - 空间分辨率
        return: [B, L, C] - 自注意力增强后的特征
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        """
        Args:
            dim (int): 特征维度/通道数, 当前值=180
                       (即 Q/K/V 每个头的维度 = dim / num_heads = 180/6 = 30)
            input_resolution (tuple[int, int]): 输入特征图的空间分辨率, 当前值=(16, 16)
                                               (来自 img_size(64)/patch_size(4) = 16)
            num_heads (int): 多头注意力的头数, 当前值=6
                             (每头 dim_head=180//6=30, 6个头并行计算不同子空间的注意力)
            window_size (int): 窗口大小, 当前值=8
                               (将 16×16 特征图划分为多个 8×8 的局部窗口, 在窗口内做注意力,
                                复杂度从 O((HW)²) 降为 O(W²×H×W))
            shift_size (int): 窗口偏移量, 当前值=0 或 4(交替)
                              (奇数层shift=0: 规则窗口; 偶数层shift=4: 窗口偏移半个window_size,
                               实现跨窗口信息交互, SWIN Transformer 的核心设计)
            mlp_ratio (float): MLP 隐藏层维度倍率, 当前值=2.0
                               (MLP隐藏层维度 = dim * mlp_ratio = 180*2 = 360,
                                FFN结构: Linear(180→360)→GELU→Linear(360→180), 含残差连接)
            qkv_bias (bool): Q/K/V 线性投影是否加偏置, 当前值=True
            qk_scale (float | None): 注意力缩放因子, 当前值=None (自动用 head_dim**-0.5)
                                     (Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V, 控制点积尺度)
            drop (float): 全局 Dropout 率, 当前值=0.0
            attn_drop (float): 注意力权重 Dropout 率, 当前值=0.0
                               (在 softmax 后对 attention weights 做 dropout, 防止过拟合)
            drop_path (float): Stochastic Depth 随机深度丢弃率, 当前值=0.1(线性递增分配到各层)
                              (训练时随机丢弃整个 block 的输出, 正则化 + 节省推理时延)
            act_layer: MLP 激活函数, 当前=GELU (Gaussian Error Linear Unit)
            norm_layer: 归一化层, 当前=LayerNorm (对最后一个维度C做归一化)
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        # 标准自注意力 (Q/K/V 均来自同一个输入)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class DualCrossTransformerBlock(nn.Module):
    """[论文 Fig.2 完整 DCATB] 双交叉注意力 Transformer Block

    这是 DCATB 的完整实现，对应论文公式(11)-(20)
    包含三个子模块:
      1. cross1 (M-HCATB): MS→HS 交叉注意力 (Q来自FY, KV来自FX)
      2. cross2 (H-MCATB): HS→MS 交叉注意力 (Q来自FX, KV来自FY) - Q/K/V与cross1相反!
      3. swin (STB): 自注意力增强

    数据流: FX → M-HCA(FX,FY) + H-MCA(FX,FY) → 加和 → STB → 输出

    forward 输入/输出:
        x: [B, C, H, W] - HSI 特征图
        y: [B, C, H, W] - MSI 特征图
        x_size: (H, W) - 空间分辨率
        return: [B, L, C] - 融合后的特征序列
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,shift_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()

        # ⭐ 分支1: M-HCATB - 将 MSI 高频纹理注入 HSI (Q=y, KV=x)
        # 当前数据集参数值:
        #   dim=180, input_resolution=(16,16), num_heads=6,
        #   window_size=8, shift_size=0或4(交替), mlp_ratio=2.0
        self.cross1 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=shift_size,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)

        # ⭐ 分支2: H-MCATB - 将 HSI 光谱信息传递给 MSI (Q=x, KV=y)，注意Q/K/V分配相反!
        # 参数与 cross1 完全相同 (同一 DCATB 内共享)
        #   dim=180, input_resolution=(16,16), num_heads=6,
        #   window_size=8, shift_size=0或4(交替), mlp_ratio=2.0
        self.cross2 = CrossSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, window_size=window_size,
                                                shift_size=shift_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer)

        # STB: 对两个分支的融合结果做自注意力增强 (论文 Fig.2 绿色部分)
        # 参数与上面相同 (同一 DCATB 内共享)
        #   dim=180, input_resolution=(16,16), num_heads=6,
        #   window_size=8, shift_size=0或4(交替), mlp_ratio=2.0
        self.swin = SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                num_heads=num_heads, window_size=window_size,
                                                shift_size=shift_size,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                drop=drop, attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer)

    def forward(self, x, y, x_size):
        """[论文公式(11)] DCATB 前向传播"""
        dual1 = self.cross1(x, y, x_size)   # M-HCA: FY→Q, FX→KV → 纹理注入HSI
        dual2 = self.cross2(y, x, x_size)   # H-MCA: FX→Q, FY→KV → 光谱信息传递
        dual = dual1 + dual2                 # 论文公式(18): 两分支特征相加
        out = self.swin(dual, x_size)       # 论文公式(19)(20): STB自注意力增强
        return out










class BasicLayer(nn.Module):
    """[对应论文 Fig.1(b)] RDCTG 的内部结构: 多个 DCATB 串联

    每个 DCATB 交替使用 shift_size=0 和 shift_size=window_size//2
    实现论文公式(9): F_X^{i,j} = DCATB(F_X^{i,j-1})

    forward 输入/输出:
        x: [B, L, C] - HSI 特征序列 (L=H*W)
        y: [B, L, C] - MSI 特征序列 (在RDCTG中保持不变)
        x_size: (H, W) 空间分辨率
        return: 融合后的特征序列
    """
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            DualCrossTransformerBlock(dim=dim, input_resolution=input_resolution,depth=depth,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]) # depth = 6

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, y, x_size):
        """[公式(9)] BasicLayer 前向: N=6 个 DCATB 串联

        输入:  x[B, L = 64 * 64, C=180], y[B, L = 64 * 64, C=180] (MSI引导信号, 在整个循环中不变), x_size=(H,W)
        输出: [B, L = 64 * 64, C=180]

        流程:
        for i in range(6):                    # depth=6, 每个DCATB是一个 DualCrossTransformerBlock
            x = blk(x, y, x_size)            # DCATB: HSI↔MSI 交叉注意力 + 窗口自注意力
                                               # x:[B,L,180] → [B,L,180]
                                               # 奇偶层交替使用 shift_size=0 和 shift=window//2 (SWIN风格)
        return x
        """
        for blk in self.blocks:
            if self.use_checkpoint: # False
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, y, x_size)  # 每个 DCATB: [B, L, = 64 * 64, 180] → [B, L, 180]
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class RSTB(nn.Module):
    """[论文 Fig.1(b) + 公式(10)] Residual Dual Cross Transformer Group (RDCTG)

    对应论文公式(9)(10):
        F_X^{i,j} = DCATB(F_X^{i,j-1})     (j=1..N个DCATB串联)
        F_X^{i,out} = Conv(F_X^{i,N}) + F_X^{i,0}  (残差连接)

    内部结构:
        - BasicLayer: N=6 个 DCATB 串联
        - Conv 3×3: 卷积层用于残差前的特征变换
        - PatchEmbed/UnEmbed + Conv: 实现序列↔图像转换后的残差

    forward 输入/输出:
        x: [B, C, H, W] - 输入 HSI 特征图
        y: [B, C, H, W] - MSI 特征图
        x_size: (H, W) 空间分辨率
        return: [B, C, H, W] - RDCTG 输出特征 (与输入同shape，通过全局残差相加)
    """

    """
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # [B, L, C=180] [B, L, C=180] (64, 64)
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, y, x_size):
        """[公式(10)] RDCTG 前向传播: F_X^{i,out} = Conv(PatchEmbed(Conv(PatchUnEmbed(residual_group(x,y))))) + x

        输入:  x[B, L, C=180], y[B, L, C=180], x_size=(H = 64,W = 64)
        输出: [B, L, C=180] (与输入同shape)

        数据流 (从内到外):
        ① residual_group(x, y, x_size)   BasicLayer: N=6个DCATB串联
           x[B,L,180] → [B,L,180]
        ② patch_unembed(x_out, x_size)   序列→图像: [B,L,180] → [B,180,H,W]
        ③ conv(x_img)                    Conv3x3:    [B,180,H,W] → [B,180,H,W]
        ④ patch_embed(x_conv)            图像→序列: [B,180,H,W] → [B,L,180]
        ⑤ + x                            全局残差:   [B,L,180] + [B,L,180] = [B,L,180]

        注意: ②→③→④ 的 序列→Conv→序列 转换是为了让残差连接在图像空间进行特征变换,
              避免在序列空间直接相加导致的维度/语义不匹配问题
        """
        return self.patch_embed(
          self.conv(
            self.patch_unembed(
              self.residual_group(
                x, y, x_size
              ), 
              x_size
            )
          )
        ) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    # img_size 64 patch_size 4 in_chans 180 embed_dim 180 norm_layer nn.Normal
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        # x:[B,180,H = 64,W = 64] → [B, L = 64 * 64, 180]
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class dualTransformer(nn.Module):
    """[DCT 的核心融合模块] 双交叉注意力 Transformer

    整体流程: 图像→Patch序列→N层DCATB(RSTB)→LayerNorm→序列→图像 → Conv → 残差相加

    forward(x[B,180,8w,8h], y[B,180,W,H]) → [B,180,8w,8h]
    """
    def __init__(self, n_feats, img_size=64, patch_size=4, depths=[6,6,6], num_heads=[6,6,6],
                 window_size=8, mlp_ratio=2, qkv_bias=True, qk_scale=None, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, resi_connection='1conv',
                 **kwargs):
        super(dualTransformer, self).__init__()
        self.patch_norm = patch_norm

        # ── PatchEmbed: 图像 → 序列 (用于 Transformer 输入) ──
        # 将 [B,C,H,W] flatten(2).transpose(1,2) → [B, H*W/patch²×patch², C]
        # 实际上就是 reshape 为序列形式, 每个 "token" 对应一个 patch_size×patch_size 的图像块
        # [B,180,H = 64,W = 64] → [B, L = 64 * 64, 180]
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution  # e.g. (16, 16) 当 img_size=64, patch=4
        self.patches_resolution = patches_resolution

        # ── PatchUnEmbed: 序列 → 图像 (用于 Transformer 输出还原) ──
        # 将 [B, L, C] transpose(1,2).view(B,C,H,W) → [B, C, H, W], 是 PatchEmbed 的逆操作
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=n_feats, embed_dim=n_feats,
            norm_layer=norm_layer if self.patch_norm else None)

        # ── conv_after_body: 深度特征提取后的卷积层 (残差连接前) ──
        # Conv(180→180, k=3, s=1, p=1), W/H不变, 用于在残差前做一次特征变换
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ── RSTB (Residual Dual Cross Transformer Group): 核心融合单元 ──
        # 内部包含: BasicLayer(N个DCATB串联) → Conv(残差变换) → 全局残差跳跃
        self.mlp_ratio = mlp_ratio
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(1):
            layer = RSTB(dim=n_feats,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)

        # ── LayerNorm: 最终输出前的归一化 ──
        # 对序列维度做 LayerNorm, 输入输出 shape 不变: [B, L, C] → [B, L, C]
        self.norm = norm_layer(n_feats)

    def forward_features(self, x, y):
        """核心特征提取流程 (不含最终残差):

        输入: x[B,180,H,W], y[B,180,H,W]
        输出: [B,180,H,W]

        内部流程:
        ① x_size = (H, W)  记录空间分辨率供 PatchUnEmbed 使用
        ② x = patch_embed(x)   图像→序列: [B,180,H,W] → [B, L, 180], L=H*W
        ③ y = patch_embed(y)   同样:      [B,180,H,W] → [B, L, 180]
        ④ for layer in layers:
             x = layer(x, y, x_size)   每层 RSTB: [B,L,180] → [B,L,180]
        ⑤ x = norm(x)          LayerNorm: [B,L,180] → [B,L,180]
        ⑥ x = patch_unembed(x) 序列→图像: [B,L,180] → [B,180,H,W]
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)     # x:[B,180,H = 64,W = 64] → [B, L = 64 * 64, 180], L=H*W/patch²×patch²=H*W
        y = self.patch_embed(y)     # y:[B,180,H = 64,W = 64] → [B, L = 64 * 64, 180] (y 仅作为引导信号传入 layer)

        for i, layer in enumerate(self.layers):  # 遍历所有 RSTB 层 (当前只有1层)
            x = layer(x, y, x_size)  # RSTB: [B,L,180] → [B,L,180] (内部含全局残差)

        x = self.norm(x)              # LayerNorm: [B, L, 180] → [B, L, 180]
        x = self.patch_unembed(x, x_size)  # 序列→图像: [B, L, 180] → [B, 180, H, W]

        return x

    def forward(self, x, y):
        """完整前向传播 (含最终残差连接):

        输入: x[B,180,8w,8h], y[B,180,W,H]
        输出: [B,180,8w,8h]

        流程: conv_after_body(forward_features(x,y)) + x
        即: Conv([B,180,8w,8h]) + 原始输入x → [B,180,8w,8h]
        """
        x = self.conv_after_body(self.forward_features(x, y)) + x
        # forward_features: x[B,180,8w,8h] → [B,180,8w,8h]
        # conv_after_body:   [B,180,8w,8h] → [B,180,8w,8h] (Conv3x3, same padding)
        # + x:               残差相加, 保证梯度直通

        return x




class Downsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Downsample, self).__init__()
        self.ratio = ratio
        dconvs = []
        for i in range(int(np.log2(ratio))):
            dconvs.append(nn.Conv2d(n_channels, n_channels, 3, stride=2, padding=1, dilation=1, groups=n_channels, bias=True))

        self.downsample = nn.Sequential(*dconvs)

    def forward(self,x):
        h = self.downsample(x)
        return h

class Upsample(nn.Module):
    def __init__(self, n_channels, ratio):
        super(Upsample, self).__init__()
        uconvs = []
        for i in range(int(np.log2(ratio))):
            uconvs.append(nn.ConvTranspose2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.upsample = nn.Sequential(*uconvs)


    def forward(self,x):
        h = self.upsample(x)
        return h

class DCT(nn.Module):
    """ [论文 Fig.1(a) 完整模型] DCTransformer - 双交叉注意力 Transformer

    整体架构对应论文 Fig.1(a)，实现:
        公式(3): X⁰ = BicubicUpsample(Z)
        公式(4)(5): F_X⁰ = Conv3x3(X⁰), F_Y = Conv3x3(Y)
        K个 RDCTG 级联 + 密集连接 (公式(6))
        公式(7): X̂ = Reconstruct(F_X⁰ + F_X^K)

    代码中通过3次 self.body() 调用实现K=3级RDCTG级联和密集连接

    Args:
        n_colors (int): HSI 的光谱波段数 S，如 Chikusei=31
        upscale_factor (int): 超分辨率倍率 l，如 8
        n_feats (int): 特征通道数 C，默认 180
    """
    def __init__(self, n_colors = 31, upscale_factor = 8, n_feats=180):
        super(DCT, self).__init__()
        kernel_size = 3
        self.up_factor = upscale_factor

        # [公式(4)] HSI 浅层特征提取: LR-HSI → 上采样 → Conv → F_X⁰ [B, C = 180, W, H]
        self.headX = nn.Conv2d(n_colors, n_feats, kernel_size, stride=1, padding=3 // 2)

        # [公式(5)] MSI 浅层特征提取: HR-MSI → 两层Conv(Relu中间) → F_Y [B, C = 180, W, H]
        self.headY = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size, stride=1, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(64, n_feats, kernel_size, stride=1, padding=3 // 2)
        )

        # 核心融合模块: dualTransformer = RSTB(BasicLayer(DualCrossTransformerBlock(...)))
        self.body = dualTransformer(n_feats)

        # 密集连接的融合卷积层 (对应论文公式(6)中的Conv)
        # fe_conv1: concat(F_X⁰, F_X¹) → [2C] → C
        # fe_conv2: concat(F_X⁰, F_X¹, F_X²) → [3C] → C
        # fe_conv3: concat(F_X⁰, F_X¹, F_X², F_X³) → [4C] → C
        self.fe_conv1 = torch.nn.Conv2d(in_channels=2*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv2 = torch.nn.Conv2d(in_channels=3*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)
        self.fe_conv3 = torch.nn.Conv2d(in_channels=4*n_feats, out_channels=n_feats, kernel_size=3, padding=3 // 2)

        # [公式(7)] 重建卷积: 融合特征 → HR-HSI X̂ [B, S, W, H]
        self.final = nn.Conv2d(n_feats, n_colors, kernel_size, stride=1, padding=3 // 2)

    def forward(self, x, y):
        """[论文完整前向传播] 公式(3)~(7)"""
        # [公式③] 双三次上采样: LR-HSI [B,31,w,h] → [B,31,8w,8h], 空间对齐到 MSI 尺寸
        x = torch.nn.functional.interpolate(x, scale_factor=self.up_factor, mode='bicubic', align_corners=False)
        # [公式④] HSI浅层特征提取: Conv(31→180), W/H不变 → x:[B,180,8w,8h]
        x = self.headX(x)
        # 保存全局残差 F_X⁰ [B,180,8w,8h], 供最终跳跃连接 (公式⑦ res + F_X^K)
        res = x
        # [公式⑤] MSI浅层特征提取: HR-MSI [B,3,W,H] → Conv(3→64)→ReLU→Conv(64→180) → y:[B,180,W,H]
        # 注: y 在后续所有 body() 调用中作为引导信号传入, 形状不再改变
        y = self.headY(y)

        # ── 第1级 RDCTG (Residual Dense Cross Transformer Group) ──
        # dualTransformer 内部: patch_embed → N层DualCrossTransformerBlock(RSTB) → norm → patch_unembed → conv + 残差
        # 输入x[B,180,8w,8h]与引导y[B,180,W,H]做交叉注意力融合 → x:[B,180,8w,8h]
        x = self.body(x,y)
        # 密集连接: cat(res[B,180],x[B,180]) 沿通道维拼接 → [B,360,8w,8h]
        x1 = torch.cat((res,x), 1)
        # 通道压缩: Conv(360→180) 融合密集特征 → x1:[B,180,8w,8h], 作为第2级输入

        x1 = self.fe_conv1(x1)

        # ── 第2级 RDCTG ──
        # 输入x1[B,180,8w,8h]已包含第1级信息, 再次与y[B,180,W,H]交叉注意力融合 → x2:[B,180,8w,8h]
        x2 = self.body(x1,y)
        # 密集连接: cat(res[B,180],x1[B,180],x2[B,180]) 三级特征拼接 → [B,540,8w,8h]
        x2 = torch.cat((res,x1, x2), 1)
        # 通道压缩: Conv(540→180) → x2:[B,180,8w,8h], 作为第3级输入
        x2 = self.fe_conv2(x2)

        # ── 第3级 RDCTG (最终级联) ──
        # 输入x2[B,180,8w,8h]包含前两级密集信息, 最终融合 → x3:[B,180,8w,8h]
        x3 = self.body(x2,y)
        # 密集连接: cat(res[B,180],x1[B,180],x2[B,180],x3[B,180]) 全部四级特征 → [B,720,8w,8h]
        x3= torch.cat((res,x1, x2, x3), 1)
        # 通道压缩: Conv(720→180) → x3:[B,180,8w,8h], 即最终的深度特征 F_X^K
        x3 = self.fe_conv3(x3)

        # [公式⑦ 前半部分] 全局残差相加: res[B,180]+x3[B,180] → [B,180,8w,8h]
        x_out = res + x3
        # [公式⑦ 后半部分] 重建卷积: [B,180,8w,8h] → Conv(180→31) 映射回光谱波段数 → ★ X̂:[B,31,8w,8h]
        x_out = self.final(x_out)

        return x_out