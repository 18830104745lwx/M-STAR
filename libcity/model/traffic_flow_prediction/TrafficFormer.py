"""
TrafficFormer: 基于 Transformer 的时空交通流量预测模型

主要特性：
    1. 四分支时空自注意力机制（Temporal + Geographic + Semantic + Adaptive）
    2. GQA (Grouped Query Attention) 优化，提升推理效率
    3. 多尺度时间精炼器，捕获多周期时间模式
    4. SwiGLU 门控前馈网络，增强表示能力
    5. 自适应动态图学习，数据驱动的节点关系建模
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel


def drop_path(x, drop_prob=0., training=False):
    """
    DropPath (随机深度) 正则化方法
    
    在训练时随机丢弃整个样本的路径（而非单个元素），用于深层网络的正则化。
    这种方法比 Dropout 更适合残差网络，因为它保持了每个样本内部的相关性。
    
    工作原理：
        - 以概率 drop_prob 将整个样本置零
        - 保留的样本需要缩放 1/(1-drop_prob) 以保持期望不变
        - 推理时不做任何操作
    
    Args:
        x (Tensor): 输入张量，shape 任意
        drop_prob (float): 丢弃概率，范围 [0, 1]
        training (bool): 是否处于训练模式
        
    Returns:
        Tensor: DropPath 后的张量，shape 与输入相同
        
    Example:
        >>> x = torch.randn(32, 10, 256)  # (B, T, C)
        >>> out = drop_path(x, drop_prob=0.1, training=True)
        >>> out.shape  # (32, 10, 256)
    """
    # 测试模式或 drop_prob=0 时，直接返回输入
    if drop_prob == 0. or not training:
        return x
    
    keep_prob = 1 - drop_prob  # 保留概率
    
    # 构造 mask shape: (B, 1, 1, ..., 1)
    # 对 batch 维度随机，其他维度广播，确保每个样本整体被保留或丢弃
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    
    # 生成随机 mask: [keep_prob, 1+keep_prob) 的均匀分布
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    
    # floor 操作：>= 1 的位置为 1（保留），< 1 的位置为 0（丢弃）
    random_tensor.floor_()
    
    # 缩放并应用 mask：保留的样本乘以 1/keep_prob 保持期望
    output = x.div(keep_prob) * random_tensor
    
    return output


class TokenEmbedding(nn.Module):
    """
    Token 嵌入层
    
    将输入的原始特征（如交通流量）通过线性变换映射到高维嵌入空间。
    可选择性地应用归一化以稳定训练。
    
    Args:
        input_dim (int): 输入特征维度（如流量特征数）
        embed_dim (int): 嵌入维度（模型隐藏层维度）
        norm_layer (nn.Module, optional): 归一化层（如 LayerNorm），默认不使用
        
    Shape:
        - Input: (B, T, N, input_dim)
        - Output: (B, T, N, embed_dim)
    """
    def __init__(self, input_dim, embed_dim, norm_layer=None):
        super().__init__()
        # 线性投影：将输入特征映射到嵌入空间
        self.token_embed = nn.Linear(input_dim, embed_dim, bias=True)
        # 可选的归一化层
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入特征，shape (B, T, N, input_dim)
            
        Returns:
            Tensor: 嵌入后的特征，shape (B, T, N, embed_dim)
        """
        x = self.token_embed(x)  # 线性投影
        x = self.norm(x)  # 归一化（可选）
        return x


class PositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    
    使用不同频率的正弦和余弦函数为序列中的每个位置生成固定的位置编码。
    这种编码方式能够让模型学习到相对位置关系。
    
    公式：
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
    其中 pos 是位置索引，i 是维度索引，d_model 是嵌入维度。
    
    Args:
        embed_dim (int): 嵌入维度
        max_len (int): 最大序列长度，默认 100
        
    Shape:
        - Input: (B, T, N, embed_dim)
        - Output: (B, T, N, embed_dim) - 与输入形状相同的位置编码
    """
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        # 初始化位置编码矩阵: (max_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim).float()
        pe.requires_grad = False  # 位置编码是固定的，不需要梯度

        # 位置索引: [0, 1, 2, ..., max_len-1], shape (max_len, 1)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # 计算分母项: 10000^(2i/d_model) 的倒数
        # 使用 exp 和 log 技巧避免数值溢出
        div_term = (torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)).exp()

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列

        # 添加 batch 维度: (max_len, embed_dim) -> (1, max_len, embed_dim)
        pe = pe.unsqueeze(0)
        # 注册为 buffer（模型状态的一部分，但不是参数）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播：返回与输入形状匹配的位置编码
        
        Args:
            x (Tensor): 输入张量，shape (B, T, N, embed_dim)
            
        Returns:
            Tensor: 位置编码，shape (B, T, N, embed_dim)
        """
        # 提取前 T 个时间步的位置编码
        # (1, max_len, embed_dim) -> (1, T, embed_dim) -> (1, T, 1, embed_dim) -> (B, T, N, embed_dim)
        return self.pe[:, :x.size(1)].unsqueeze(2).expand_as(x).detach()


class LaplacianPE(nn.Module):
    """
    拉普拉斯位置编码 (Laplacian Positional Encoding)
    
    使用图拉普拉斯矩阵的特征向量作为节点的位置编码。
    拉普拉斯特征向量能够捕获图的结构信息，为不同节点提供结构感知的位置编码。
    
    原理：
        - 图拉普拉斯矩阵 L = D - A（D是度矩阵，A是邻接矩阵）
        - 对 L 进行特征分解，取前 k 个最小特征值对应的特征向量
        - 这些特征向量编码了节点在图中的相对位置关系
        
    Args:
        lape_dim (int): 拉普拉斯特征向量维度（输入维度）
        embed_dim (int): 嵌入维度（输出维度）
        
    Shape:
        - Input: (N, lape_dim) - N 个节点的拉普拉斯特征
        - Output: (1, 1, N, embed_dim) - 广播到所有 batch 和时间步
    """
    def __init__(self, lape_dim, embed_dim):
        super().__init__()
        # 线性投影：将拉普拉斯特征映射到嵌入空间
        self.embedding_lap_pos_enc = nn.Linear(lape_dim, embed_dim)

    def forward(self, lap_mx):
        """
        前向传播
        
        Args:
            lap_mx (Tensor): 拉普拉斯特征矩阵，shape (N, lape_dim)
            
        Returns:
            Tensor: 空间位置编码，shape (1, 1, N, embed_dim)
                   可广播到 (B, T, N, embed_dim)
        """
        # 投影并添加维度用于广播
        # (N, lape_dim) -> (N, embed_dim) -> (1, 1, N, embed_dim)
        lap_pos_enc = self.embedding_lap_pos_enc(lap_mx).unsqueeze(0).unsqueeze(0)
        return lap_pos_enc


class DataEmbedding(nn.Module):
    """
    数据嵌入层 - 整合多种嵌入
    
    将原始输入数据转换为模型的初始嵌入表示，整合以下信息：
        1. 值嵌入 (Value Embedding): 交通流量等数值特征
        2. 位置编码 (Positional Encoding): 时间序列位置信息
        3. 时间嵌入 (Time Embedding): 一天中的时刻 (分钟级)
        4. 星期嵌入 (Weekday Embedding): 星期几（周一到周日）
        5. 空间嵌入 (Spatial Embedding): 节点的图结构位置
        
    所有嵌入相加后得到最终的输入表示。
    
    Args:
        feature_dim (int): 原始特征维度（流量特征数）
        embed_dim (int): 嵌入维度
        lape_dim (int): 拉普拉斯特征维度
        adj_mx: 邻接矩阵（保留参数，实际未使用）
        drop (float): Dropout 概率
        add_time_in_day (bool): 是否添加时刻嵌入
        add_day_in_week (bool): 是否添加星期嵌入
        device: 计算设备
        
    Shape:
        - Input: (B, T, N, F) - F 可能包含流量特征 + 时间特征
        - Output: (B, T, N, embed_dim)
    """
    def __init__(
        self, feature_dim, embed_dim, lape_dim, adj_mx, drop=0.,
        add_time_in_day=False, add_day_in_week=False, device=torch.device('cpu'),
    ):
        super().__init__()

        self.add_time_in_day = add_time_in_day  # 是否使用时刻特征
        self.add_day_in_week = add_day_in_week  # 是否使用星期特征

        self.device = device
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        
        # 1. 值嵌入：将交通流量等数值特征映射到嵌入空间
        self.value_embedding = TokenEmbedding(feature_dim, embed_dim)

        # 2. 位置编码：时间序列的位置信息
        self.position_encoding = PositionalEncoding(embed_dim)
        
        # 3. 时刻嵌入：一天中的分钟数 (0-1439)
        if self.add_time_in_day:
            self.minute_size = 1440  # 24 * 60 = 1440 分钟
            self.daytime_embedding = nn.Embedding(self.minute_size, embed_dim)
            
        # 4. 星期嵌入：星期几 (0-6)
        if self.add_day_in_week:
            weekday_size = 7  # 周一到周日
            self.weekday_embedding = nn.Embedding(weekday_size, embed_dim)
            
        # 5. 空间嵌入：节点的图结构位置
        self.spatial_embedding = LaplacianPE(lape_dim, embed_dim)
        
        # Dropout 正则化
        self.dropout = nn.Dropout(drop)

    def forward(self, x, lap_mx):
        """
        前向传播：整合多种嵌入
        
        Args:
            x (Tensor): 输入特征，shape (B, T, N, F)
                       F 维度可能包含：[流量特征, 时刻归一化值, 星期one-hot编码]
            lap_mx (Tensor): 拉普拉斯特征矩阵，shape (N, lape_dim)
            
        Returns:
            Tensor: 嵌入后的特征，shape (B, T, N, embed_dim)
        """
        origin_x = x  # 保存原始输入用于提取时间特征
        
        # 1. 值嵌入：提取前 feature_dim 维的流量特征
        # (B, T, N, feature_dim) -> (B, T, N, embed_dim)
        x = self.value_embedding(origin_x[:, :, :, :self.feature_dim])
        
        # 2. 位置编码：加入时间序列位置信息
        x += self.position_encoding(x)
        
        # 3. 时刻嵌入：如果输入包含时刻特征（归一化到 [0, 1]）
        if self.add_time_in_day and origin_x.shape[-1] > self.feature_dim:
            # 提取时刻特征（第 feature_dim 维），反归一化到分钟数 [0, 1439]
            time_in_day = (origin_x[:, :, :, self.feature_dim] * self.minute_size).round().long()
            x += self.daytime_embedding(time_in_day)
        
        # 4. 星期嵌入：如果输入包含星期 one-hot 编码（7维）
        if self.add_day_in_week and origin_x.shape[-1] >= self.feature_dim + 8:
            # 提取星期 one-hot（第 feature_dim+1 到 feature_dim+7 维）
            # argmax 得到星期索引 [0, 6]
            day_of_week = origin_x[:, :, :, self.feature_dim + 1: self.feature_dim + 8].argmax(dim=3)
            x += self.weekday_embedding(day_of_week)
        
        # 5. 空间嵌入：加入节点的图结构位置信息
        if lap_mx is not None:
            x += self.spatial_embedding(lap_mx)
        
        # Dropout 正则化
        x = self.dropout(x)
        
        return x


class DropPath(nn.Module):
    """
    DropPath (随机深度) 模块封装
    
    将 drop_path 函数封装为 nn.Module，方便在网络中使用。
    主要用于深层残差网络的正则化，随机丢弃整条路径以防止过拟合。
    
    Args:
        drop_prob (float): 丢弃概率，None 表示不丢弃
        
    Shape:
        - Input: 任意形状
        - Output: 与输入相同形状
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入张量
            
        Returns:
            Tensor: DropPath 后的张量
        """
        return drop_path(x, self.drop_prob, self.training)


# ============================================================================
# 多尺度时间精炼器 (Multi-Scale Temporal Refiner)
# ============================================================================
class MultiScaleTemporalRefiner(nn.Module):
    """
    多尺度时间精炼器
    
    通过四个并行分支捕获不同粒度的时间依赖：
    - 多膨胀率卷积分支：捕获多周期时间模式
    - 时间自注意力分支：捕获长距离时间依赖
    - 跨节点时间交互分支：建模节点间的时间耦合
    - 时间残差学习分支：保持细节信息
    
    Args:
        channels: 特征通道数
        dilations: 膨胀率列表，默认(1, 2, 4, 8)
        kernel_size: 卷积核大小
        dropout: Dropout概率
    """
    def __init__(self, channels: int, dilations=(1, 2, 4, 8), kernel_size=5, dropout=0.1):
        super().__init__()
        
        # 多膨胀率卷积分支
        self.branches = nn.ModuleList()
        padding = lambda k, d: (k + (k - 1) * (d - 1) - 1) // 2  # 保持长度
        for d in dilations:
            self.branches.append(nn.Sequential(
                # Depthwise
                nn.Conv1d(channels, channels, kernel_size,
                          padding=padding(kernel_size, d), dilation=d,
                          groups=channels, bias=False),
                nn.GELU(),
                # Pointwise
                nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            ))
        
        # 轻量级时间自注意力分支
        self.time_attn = nn.MultiheadAttention(
            embed_dim=channels//4,  # 降维减少计算量
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.time_norm = nn.LayerNorm(channels//4)
        self.time_proj_in = nn.Linear(channels, channels//4)
        self.time_proj_out = nn.Linear(channels//4, channels)
        
        # 跨节点时间交互分支
        self.cross_node_conv = nn.Conv1d(channels, channels, 
                                       kernel_size=3, padding=1, 
                                       groups=channels//8)
        
        # 时间残差学习分支
        self.residual_conv = nn.Sequential(
            nn.Conv1d(channels, channels//2, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels//2, channels, kernel_size=1),
        )
        
        # 特征融合
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False) if len(dilations) > 1 else nn.Identity()
        self.gate = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1),  # 3个分支：膨胀卷积+时间注意力+跨节点交互
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        四分支并行处理：
            1. 多膨胀率卷积：捕获不同尺度的时间模式（周期性）
            2. 时间自注意力：捕获长距离时间依赖
            3. 跨节点时间交互：建模节点间的时间耦合关系
            4. 时间残差学习：保持细节信息
        
        Args:
            x: 输入特征, shape (B, T, N, C)
               B - batch size
               T - 时间步数
               N - 节点数
               C - 特征通道数
            
        Returns:
            输出特征, shape (B, T, N, C)，通过残差连接保持输入形状
        """
        B, T, N, C = x.shape
        
        # ========== 分支1: 多膨胀率卷积 ==========
        # 捕获多周期时间模式（如日周期、周周期等）
        
        # 重塑为卷积格式：(B, T, N, C) -> (B, N, C, T) -> (B*N, C, T)
        z = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        
        # 对每个膨胀率分支并行计算
        dilated_outs = [branch(z) for branch in self.branches]  # 每个分支输出 (B*N, C, T)
        
        # 堆叠并求和融合多尺度特征
        dilated_feat = torch.stack(dilated_outs, dim=0).sum(dim=0) if len(dilated_outs) > 1 else dilated_outs[0]
        dilated_feat = self.proj(dilated_feat)  # 投影：(B*N, C, T)
        
        # 恢复时空维度：(B*N, C, T) -> (B, N, C, T) -> (B, T, N, C)
        dilated_feat = dilated_feat.reshape(B, N, C, T).permute(0, 3, 1, 2)
        
        # ========== 分支2: 时间自注意力 ==========
        # 捕获长距离时间依赖，使用降维减少计算量
        
        # 降维投影：(B, T, N, C) -> (B, T, N, C//4)
        x_time = self.time_proj_in(x)
        # 重塑为注意力格式：(B, T, N, C//4) -> (B*N, T, C//4)
        x_time_flat = x_time.view(B*N, T, C//4)
        
        # 多头自注意力计算
        attn_out, _ = self.time_attn(x_time_flat, x_time_flat, x_time_flat)  # (B*N, T, C//4)
        # 残差连接 + 层归一化
        attn_out = self.time_norm(attn_out + x_time_flat)
        
        # 升维投影：(B*N, T, C//4) -> (B*N, T, C)
        attn_feat = self.time_proj_out(attn_out)
        # 恢复时空维度：(B*N, T, C) -> (B, T, N, C)
        attn_feat = attn_feat.view(B, T, N, C)
        
        # ========== 分支3: 跨节点时间交互 ==========
        # 在空间维度上建模时间交互，捕获节点间的动态关系
        
        # 重塑为空间卷积格式：(B, T, N, C) -> (B, T, C, N) -> (B*T, C, N)
        x_cross = x.permute(0, 1, 3, 2).reshape(B*T, C, N)
        # 分组卷积：捕获节点间交互
        cross_out = self.cross_node_conv(x_cross)  # (B*T, C, N)
        # 恢复时空维度：(B*T, C, N) -> (B, T, C, N) -> (B, T, N, C)
        cross_feat = cross_out.view(B, T, C, N).permute(0, 1, 3, 2)
        
        # ========== 分支4: 时间残差学习 ==========
        # 通过瓶颈结构学习时间残差，保持细节信息
        
        # 重塑：(B, T, N, C) -> (B, N, C, T) -> (B*N, C, T)
        z_res = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        # 残差卷积：C -> C/2 -> C
        residual_out = self.residual_conv(z_res)  # (B*N, C, T)
        # 恢复时空维度：(B*N, C, T) -> (B, N, C, T) -> (B, T, N, C)
        residual_feat = residual_out.reshape(B, N, C, T).permute(0, 3, 1, 2)
        
        # ========== 门控融合 ==========
        # 使用可学习的门控机制自适应融合多个分支
        
        # 拼接三个主分支：(B, T, N, 3*C)
        combined = torch.cat([dilated_feat, attn_feat, cross_feat], dim=-1)
        # 重塑为卷积格式：(B, T, N, 3*C) -> (B, N, 3*C, T) -> (B*N, 3*C, T)
        combined = combined.permute(0, 2, 3, 1).reshape(B*N, 3*C, T)
        
        # 生成门控权重：(B*N, 3*C, T) -> (B*N, C, T)
        gate_weights = self.gate(combined)  # Sigmoid 激活，值域 [0, 1]
        
        # 应用门控：加权膨胀卷积分支
        dilated_reshaped = dilated_feat.permute(0, 2, 3, 1).reshape(B*N, C, T)
        gated_out = gate_weights * dilated_reshaped  # 逐元素相乘
        
        # ========== 最终融合 ==========
        # 门控输出 + 残差分支
        final_out = gated_out + residual_out  # (B*N, C, T)
        final_out = self.dropout(final_out)  # Dropout 正则化
        
        # 恢复时空维度：(B*N, C, T) -> (B, N, C, T) -> (B, T, N, C)
        final_out = final_out.reshape(B, N, C, T).permute(0, 3, 1, 2)
        
        # 残差连接到原输入
        return x + final_out


# ============================================================================
# 时空自注意力模块 (Spatiotemporal Self-Attention)
# ============================================================================
class STSelfAttention(nn.Module):
    """
    Spatiotemporal Self-Attention with GQA (Grouped Query Attention) - 四分支
    
    核心思想：
        使用四种不同的注意力头捕获不同类型的时空依赖：
        1. Temporal Head (T): 使用MQA捕获时间序列依赖
           - 为什么用MQA？时间维度计算量最大（T×T矩阵），需要最激进的优化
           - 时间注意力对每个节点独立计算（B×N个独立的T×T注意力矩阵）
           - MQA（Multi-Query Attention）：所有头共享同一组K/V，最大化减少计算量
        2. Geographic Head (Geo): 使用GQA捕获空间邻近性（基于地理距离）
           - 为什么用GQA？空间维度较小（N×N矩阵），可以保留更多表达能力
           - GQA在效率和表达能力之间取得平衡
        3. Semantic Head (Sem): 使用GQA捕获功能相似性（基于DTW距离）
        4. Adaptive Graph Head (Adp): 使用GQA捕获数据驱动的动态图结构
        
    自适应图注意力（Adaptive Graph Attention）设计：
        - 动态特征相似度：根据输入特征实时计算节点间相似度
        - 稀疏化Top-K机制：只保留最相关的K个邻居，提升效率
        - 可学习的图生成器：使用MLP学习如何从特征生成图结构
        - 与先验图互补：Geographic基于位置，Semantic基于历史模式，Adaptive基于当前状态
        
    GQA优势（参考论文：GQA: Training Generalized Multi-Query Transformer Models）：
        - Geographic/Semantic使用GQA：查询头分组，组内共享K/V
        - Temporal使用MQA：所有头共享K/V（时间维度计算量大，需要更激进优化）
        - 在推理速度和模型质量之间取得最佳平衡
        - 相比MHA减少KV缓存，相比MQA保持更好的表达能力
        
    Args:
        dim (int): 输入特征维度
        s_attn_size (int): 空间注意力窗口大小
        t_attn_size (int): 时间注意力窗口大小
        geo_num_heads (int): 地理注意力查询头数，默认4
        sem_num_heads (int): 语义注意力查询头数，默认2
        adp_num_heads (int): 自适应图注意力查询头数，默认2
        t_num_heads (int): 时间注意力头数，默认2
        geo_num_groups (int): 地理注意力分组数，默认2
        sem_num_groups (int): 语义注意力分组数，默认1
        adp_num_groups (int): 自适应图注意力分组数，默认1
        adp_topk (int): 自适应图Top-K邻居数
        qkv_bias (bool): QKV投影是否使用偏置
        attn_drop (float): 注意力dropout概率
        proj_drop (float): 投影层dropout概率
        device: 计算设备
        output_dim (int): 输出维度
    """
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, adp_num_heads=2, t_num_heads=2, 
        geo_num_groups=2, sem_num_groups=1, adp_num_groups=1, adp_topk=10, qkv_bias=False,
        attn_drop=0., proj_drop=0., device=torch.device('cpu'), output_dim=1,
    ):
        super().__init__()
        # 计算总头数（排除为0的头）
        total_heads = geo_num_heads + sem_num_heads + adp_num_heads + t_num_heads
        assert total_heads > 0, "At least one attention head type must have num_heads > 0"
        assert dim % total_heads == 0, f"dim ({dim}) must be divisible by total_heads ({total_heads})"
        # 只有当head数量>0时，才检查分组整除关系
        if geo_num_heads > 0:
            assert geo_num_heads % geo_num_groups == 0, f"geo_num_heads ({geo_num_heads}) must be divisible by geo_num_groups ({geo_num_groups})"
        if sem_num_heads > 0:
            assert sem_num_heads % sem_num_groups == 0, f"sem_num_heads ({sem_num_heads}) must be divisible by sem_num_groups ({sem_num_groups})"
        if adp_num_heads > 0:
            assert adp_num_heads % adp_num_groups == 0, f"adp_num_heads ({adp_num_heads}) must be divisible by adp_num_groups ({adp_num_groups})"
        
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.adp_num_heads = adp_num_heads
        self.t_num_heads = t_num_heads
        self.geo_num_groups = geo_num_groups  # GQA: Geographic分组数
        self.sem_num_groups = sem_num_groups  # GQA: Semantic分组数
        self.adp_num_groups = adp_num_groups  # GQA: Adaptive分组数
        self.adp_topk = adp_topk
        
        # 每个头的维度 = 总维度 / 总头数
        self.head_dim = dim // total_heads
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子: 1/√d
        self.device = device
        self.s_attn_size = s_attn_size
        self.t_attn_size = t_attn_size
        
        # 每种注意力的通道数 = 头数 × 每头维度
        self.geo_channels = self.head_dim * geo_num_heads
        self.sem_channels = self.head_dim * sem_num_heads
        self.adp_channels = self.head_dim * adp_num_heads
        self.t_channels = self.head_dim * t_num_heads
        self.output_dim = output_dim

        # ========== Geographic Attention (GQA) ==========
        # Query: geo_num_heads个头，每个头维度为head_dim
        # Key/Value: geo_num_groups组，每组维度为head_dim
        # 例如：4个Q头，2组KV -> 每2个Q头共享1组KV
        if geo_num_heads > 0:
            self.geo_q_conv = nn.Conv2d(dim, self.geo_channels, kernel_size=1, bias=qkv_bias)
            self.geo_k_conv = nn.Conv2d(dim, self.head_dim * geo_num_groups, kernel_size=1, bias=qkv_bias)
            self.geo_v_conv = nn.Conv2d(dim, self.head_dim * geo_num_groups, kernel_size=1, bias=qkv_bias)
            self.geo_attn_drop = nn.Dropout(attn_drop)
        else:
            self.geo_q_conv = None
            self.geo_k_conv = None
            self.geo_v_conv = None
            self.geo_attn_drop = None

        # ========== Semantic Attention (GQA) ==========
        # Query: sem_num_heads个头，每个头维度为head_dim
        # Key/Value: sem_num_groups组，每组维度为head_dim
        # 例如：2个Q头，1组KV -> 2个Q头共享1组KV（等价于MQA）
        if sem_num_heads > 0:
            self.sem_q_conv = nn.Conv2d(dim, self.sem_channels, kernel_size=1, bias=qkv_bias)
            self.sem_k_conv = nn.Conv2d(dim, self.head_dim * sem_num_groups, kernel_size=1, bias=qkv_bias)
            self.sem_v_conv = nn.Conv2d(dim, self.head_dim * sem_num_groups, kernel_size=1, bias=qkv_bias)
            self.sem_attn_drop = nn.Dropout(attn_drop)
        else:
            self.sem_q_conv = None
            self.sem_k_conv = None
            self.sem_v_conv = None
            self.sem_attn_drop = None

        # ========== Adaptive Graph Attention (GQA) ==========
        if adp_num_heads > 0:
            self.adp_q_conv = nn.Conv2d(dim, self.adp_channels, kernel_size=1, bias=qkv_bias)
            self.adp_k_conv = nn.Conv2d(dim, self.head_dim * self.adp_num_groups, kernel_size=1, bias=qkv_bias)
            self.adp_v_conv = nn.Conv2d(dim, self.head_dim * self.adp_num_groups, kernel_size=1, bias=qkv_bias)
            self.adp_attn_drop = nn.Dropout(attn_drop)
            # 图生成器（时间聚合 + MLP）
            self.graph_generator = nn.Sequential(
                nn.Linear(dim, dim // 2),
                nn.LayerNorm(dim // 2),
                nn.GELU(),
                nn.Dropout(attn_drop * 0.5),
                nn.Linear(dim // 2, dim // 4),
            )
        else:
            self.adp_q_conv = None
            self.adp_k_conv = None
            self.adp_v_conv = None
            self.adp_attn_drop = None
            self.graph_generator = None

        # Temporal Attention Head使用Multi-Query Attention降低计算复杂度
        # 多头Query: (B, D, T, N) -> (B, t_channels, T, N)
        # 共享Key/Value: (B, D, T, N) -> (B, head_dim, T, N)
        self.t_q_conv = nn.Conv2d(dim, self.t_channels, kernel_size=1, bias=qkv_bias)
        self.t_k_conv = nn.Conv2d(dim, self.head_dim, kernel_size=1, bias=qkv_bias)
        self.t_v_conv = nn.Conv2d(dim, self.head_dim, kernel_size=1, bias=qkv_bias)
        self.t_attn_drop = nn.Dropout(attn_drop)

        # 最终投影层：拼接四个分支的输出后投影回原维度
        # 维度: (B, T, N, dim) -> (B, T, N, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, geo_mask=None, sem_mask=None, return_attention=False):
        """
        四分支时空自注意力前向传播（使用GQA优化）
        
        计算流程：
            1. Temporal Head: 捕获时间序列依赖（MQA - 所有头共享K/V）
            2. Geographic Head: 捕获空间邻近性（GQA - 分组共享K/V）
            3. Semantic Head: 捕获功能相似性（GQA - 分组共享K/V）
            4. Adaptive Graph Head: 捕获数据驱动的动态图结构（GQA - 分组共享K/V）
            5. 拼接四个分支的输出并投影
            
        GQA机制说明：
            - Geographic: geo_num_heads个Q头分成geo_num_groups组，组内共享K/V
            - Semantic: sem_num_heads个Q头分成sem_num_groups组，组内共享K/V
            - 相比MHA减少了KV参数量和缓存，提升推理效率
            - 相比MQA保持了更好的表达能力和模型质量
            
        Args:
            x: 输入时空特征, shape (B, T, N, D)
            geo_mask: 地理掩码, shape (N, N), 屏蔽远距离节点
            sem_mask: 语义掩码, shape (N, N), 屏蔽不相似节点
            return_attention: 是否返回注意力权重
            
        Returns:
            如果return_attention=False: 输出特征, shape (B, T, N, D)
            如果return_attention=True: (输出特征, 注意力字典)
        """
        B, T, N, D = x.shape
        
        # 预处理: 转置为Conv2d所需格式，只permute一次提高效率
        # 维度: (B, T, N, D) -> (B, D, T, N)
        x_perm = x.permute(0, 3, 1, 2)
        
        # ==================== Temporal Head (Multi-Query Attention) ====================
        # Multi-Query Attention: 多头Query，共享Key和Value，降低计算复杂度

        # 生成多头Query
        # 维度: (B,D,T,N) -> (B,t_channels,T,N) -> (B,T,N,t_channels)
        t_q = self.t_q_conv(x_perm).permute(0, 2, 3, 1)
        # 重塑为多头格式: (B,T,N,t_channels) -> (B,T,N,H,dh) -> (B,N,H,T,dh)
        t_q = t_q.reshape(B, T, N, self.t_num_heads, self.head_dim).permute(0, 2, 3, 1, 4)
        
        # 生成共享的Key和Value
        # 维度: (B,D,T,N) -> (B,head_dim,T,N) -> (B,T,N,head_dim)
        k_shared = self.t_k_conv(x_perm).permute(0, 2, 3, 1)
        v_shared = self.t_v_conv(x_perm).permute(0, 2, 3, 1)
        # 转置时空维度: (B,T,N,dh) -> (B,N,T,dh)
        k_shared = k_shared.permute(0, 2, 1, 3)
        v_shared = v_shared.permute(0, 2, 1, 3)
        # 扩展到多头: (B,N,T,dh) -> (B,N,H,T,dh)
        t_k = k_shared.unsqueeze(2).expand(-1, -1, self.t_num_heads, -1, -1)
        t_v = v_shared.unsqueeze(2).expand(-1, -1, self.t_num_heads, -1, -1)
        
        # 使用SDPA高效计算注意力
        # 重塑为SDPA格式: (B,N,H,T,dh) -> (B*N,H,T,dh)
        q_sdpa = t_q.reshape(B * N, self.t_num_heads, T, self.head_dim)
        k_sdpa = t_k.reshape(B * N, self.t_num_heads, T, self.head_dim)
        v_sdpa = t_v.reshape(B * N, self.t_num_heads, T, self.head_dim)
        
        # 计算temporal attention (保存用于可视化)
        t_attn = (q_sdpa @ k_sdpa.transpose(-2, -1)) * self.scale
        t_attn = t_attn.softmax(dim=-1)
        t_attn = self.t_attn_drop(t_attn)
        # 加权求和: Attn @ V, shape (B*N,H,T,dh)
        t_out = t_attn @ v_sdpa
        t_out = t_out.reshape(B, N, self.t_num_heads, T, self.head_dim)
        
        # 恢复时空维度: (B,N,H,T,dh) -> (B,N,T,H,dh) -> (B,N,T,t_channels) -> (B,T,N,t_channels)
        t_x = t_out.transpose(2, 3).reshape(B, N, T, self.t_channels).transpose(1, 2)

        # ==================== Geographic Head (GQA) ====================
        # 捕获基于地理距离的空间依赖，使用分组查询注意力（GQA）
        if self.geo_num_heads > 0:
            # 生成Query（所有头）和Key/Value（分组）
            # Q: (B,D,T,N) -> (B,geo_channels,T,N) -> (B,T,N,geo_channels)
            # K/V: (B,D,T,N) -> (B,head_dim*geo_num_groups,T,N) -> (B,T,N,head_dim*geo_num_groups)
            geo_q = self.geo_q_conv(x_perm).permute(0, 2, 3, 1)
            geo_k = self.geo_k_conv(x_perm).permute(0, 2, 3, 1)
            geo_v = self.geo_v_conv(x_perm).permute(0, 2, 3, 1)
            
            # Query重塑为多头: (B,T,N,geo_channels) -> (B,T,N,H,dh) -> (B,T,H,N,dh)
            geo_q = geo_q.reshape(B, T, N, self.geo_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            
            # GQA核心：Key/Value重塑为分组，然后扩展到所有头
            # (B,T,N,G*dh) -> (B,T,N,G,dh) -> (B,T,G,N,dh)
            geo_k_grouped = geo_k.reshape(B, T, N, self.geo_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)
            geo_v_grouped = geo_v.reshape(B, T, N, self.geo_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)
            
            # 扩展K/V：每组重复(geo_num_heads // geo_num_groups)次
            # (B,T,G,N,dh) -> (B,T,G,heads_per_group,N,dh) -> (B,T,H,N,dh)
            heads_per_group = self.geo_num_heads // self.geo_num_groups
            geo_k = geo_k_grouped.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1, -1).reshape(B, T, self.geo_num_heads, N, self.head_dim)
            geo_v = geo_v_grouped.unsqueeze(3).expand(-1, -1, -1, heads_per_group, -1, -1).reshape(B, T, self.geo_num_heads, N, self.head_dim)
            
            # 注意力计算: QK^T/√d, shape (B,T,H,N,N)
            geo_attn = (geo_q @ geo_k.transpose(-2, -1)) * self.scale
            # 应用地理掩码: 屏蔽距离过远的节点对
            if geo_mask is not None:
                # 扩展mask维度: (N, N) -> (B, T, H, N, N)
                geo_mask_expanded = geo_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, self.geo_num_heads, -1, -1)
                geo_attn.masked_fill_(geo_mask_expanded, float('-inf'))
            geo_attn = geo_attn.softmax(dim=-1)
            geo_attn = self.geo_attn_drop(geo_attn)
            # 加权求和: (B,T,H,N,N) @ (B,T,H,N,dh) -> (B,T,H,N,dh)
            # 恢复形状: (B,T,H,N,dh) -> (B,T,N,H,dh) -> (B,T,N,geo_channels)
            geo_x = (geo_attn @ geo_v).transpose(2, 3).reshape(B, T, N, self.geo_channels)
        else:
            # 如果geo_num_heads=0，创建零张量
            geo_x = torch.zeros(B, T, N, self.geo_channels, device=x.device, dtype=x.dtype)
            geo_attn = None

        # ==================== Semantic Head (GQA) ====================
        # 捕获基于功能相似性（DTW）的空间依赖，使用分组查询注意力（GQA）
        if self.sem_num_heads > 0:
            # 生成Query（所有头）和Key/Value（分组）
            # Q: (B,D,T,N) -> (B,sem_channels,T,N) -> (B,T,N,sem_channels)
            # K/V: (B,D,T,N) -> (B,head_dim*sem_num_groups,T,N) -> (B,T,N,head_dim*sem_num_groups)
            sem_q = self.sem_q_conv(x_perm).permute(0, 2, 3, 1)
            sem_k = self.sem_k_conv(x_perm).permute(0, 2, 3, 1)
            sem_v = self.sem_v_conv(x_perm).permute(0, 2, 3, 1)
            
            # Query重塑为多头: (B,T,N,sem_channels) -> (B,T,N,H,dh) -> (B,T,H,N,dh)
            sem_q = sem_q.reshape(B, T, N, self.sem_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            
            # GQA核心：Key/Value重塑为分组，然后扩展到所有头
            # (B,T,N,G*dh) -> (B,T,N,G,dh) -> (B,T,G,N,dh)
            sem_k_grouped = sem_k.reshape(B, T, N, self.sem_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)
            sem_v_grouped = sem_v.reshape(B, T, N, self.sem_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)
            
            # 扩展K/V：每组重复(sem_num_heads // sem_num_groups)次
            # (B,T,G,N,dh) -> (B,T,G,heads_per_group,N,dh) -> (B,T,H,N,dh)
            sem_heads_per_group = self.sem_num_heads // self.sem_num_groups
            sem_k = sem_k_grouped.unsqueeze(3).expand(-1, -1, -1, sem_heads_per_group, -1, -1).reshape(B, T, self.sem_num_heads, N, self.head_dim)
            sem_v = sem_v_grouped.unsqueeze(3).expand(-1, -1, -1, sem_heads_per_group, -1, -1).reshape(B, T, self.sem_num_heads, N, self.head_dim)
            
            # 注意力计算: QK^T/√d, shape (B,T,H,N,N)
            sem_attn = (sem_q @ sem_k.transpose(-2, -1)) * self.scale
            # 应用语义掩码: 屏蔽功能不相似的节点对
            if sem_mask is not None:
                # 扩展mask维度: (N, N) -> (B, T, H, N, N)
                sem_mask_expanded = sem_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, self.sem_num_heads, -1, -1)
                sem_attn.masked_fill_(sem_mask_expanded, float('-inf'))
            sem_attn = sem_attn.softmax(dim=-1)
            sem_attn = self.sem_attn_drop(sem_attn)
            # 加权求和并恢复形状: (B,T,H,N,dh) -> (B,T,N,sem_channels)
            sem_x = (sem_attn @ sem_v).transpose(2, 3).reshape(B, T, N, self.sem_channels)
        else:
            # 如果sem_num_heads=0，创建零张量
            sem_x = torch.zeros(B, T, N, self.sem_channels, device=x.device, dtype=x.dtype)
            sem_attn = None

        # ==================== Adaptive Graph Head (GQA) ====================
        # 改进的动态图生成：时空联合建模
        if self.adp_num_heads > 0:
            # 1. 时间加权聚合（而非简单平均）
            time_weights = F.softmax(torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(0), dim=-1)
            time_weights = time_weights.view(1, T, 1, 1).expand(B, T, N, D)
            x_node = (x * time_weights).sum(dim=1)  # (B, N, D) - 近期时刻权重更大
            
            # 2. 多尺度节点嵌入
            node_embed = self.graph_generator(x_node)  # (B, N, D//4)
            node_embed_norm = F.normalize(node_embed, p=2, dim=-1)
            
            # 3. 相似度计算：余弦相似度
            graph_sim = torch.bmm(node_embed_norm, node_embed_norm.transpose(1, 2))  # (B, N, N)
            
            # 4. 自适应Top-K选择（基于图密度）
            K = min(self.adp_topk, N)
            # 使用Gumbel-Softmax实现可微分的Top-K选择
            _, topk_indices = torch.topk(graph_sim, k=K, dim=-1)
            keep = torch.zeros(B, N, N, device=x.device, dtype=torch.bool)
            batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, K)
            node_idx = torch.arange(N, device=x.device).view(1, N, 1).expand(B, N, K)
            keep[batch_idx, node_idx, topk_indices] = True
            
            # 对称化 + 保留自连接
            keep = keep | keep.transpose(1, 2)
            keep[:, range(N), range(N)] = True  # 确保自连接
            adp_mask = ~keep

            # 生成QKV并应用GQA
            adp_q = self.adp_q_conv(x_perm).permute(0, 2, 3, 1)
            adp_k = self.adp_k_conv(x_perm).permute(0, 2, 3, 1)
            adp_v = self.adp_v_conv(x_perm).permute(0, 2, 3, 1)

            adp_q = adp_q.reshape(B, T, N, self.adp_num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
            adp_k_grouped = adp_k.reshape(B, T, N, self.adp_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)
            adp_v_grouped = adp_v.reshape(B, T, N, self.adp_num_groups, self.head_dim).permute(0, 1, 3, 2, 4)

            adp_heads_per_group = self.adp_num_heads // self.adp_num_groups
            adp_k = adp_k_grouped.unsqueeze(3).expand(-1, -1, -1, adp_heads_per_group, -1, -1).reshape(B, T, self.adp_num_heads, N, self.head_dim)
            adp_v = adp_v_grouped.unsqueeze(3).expand(-1, -1, -1, adp_heads_per_group, -1, -1).reshape(B, T, self.adp_num_heads, N, self.head_dim)

            adp_attn = (adp_q @ adp_k.transpose(-2, -1)) * self.scale
            adp_mask_expanded = adp_mask.unsqueeze(1).unsqueeze(1).expand(B, T, self.adp_num_heads, N, N)
            adp_attn.masked_fill_(adp_mask_expanded, float('-inf'))
            adp_attn = adp_attn.softmax(dim=-1)
            adp_attn = self.adp_attn_drop(adp_attn)
            adp_x = (adp_attn @ adp_v).transpose(2, 3).reshape(B, T, N, self.adp_channels)
        else:
            # 如果adp_num_heads=0，创建零张量
            adp_x = torch.zeros(B, T, N, self.adp_channels, device=x.device, dtype=x.dtype)
            adp_attn = None

        # ==================== 拼接并投影 ====================
        # 拼接四个分支的输出: 维度 (B,T,N,t+geo+sem+adp) = (B,T,N,D)
        # 投影回原维度: (B,T,N,D) -> (B,T,N,D)
        x = self.proj(torch.cat([t_x, geo_x, sem_x, adp_x], dim=-1))
        x = self.proj_drop(x)
        
        if return_attention:
            # 返回注意力权重用于可视化
            # temporal: (B*N, H, T, T) -> 取平均为 (T, T)
            # spatial: (B, T, H, N, N) -> 取平均为 (N, N)
            attention_weights = {
                'temporal': t_attn.mean(dim=(0, 1)).detach().cpu() if t_attn is not None else None,
                'geographic': geo_attn.mean(dim=(0, 1, 2)).detach().cpu() if geo_attn is not None else None,
                'semantic': sem_attn.mean(dim=(0, 1, 2)).detach().cpu() if sem_attn is not None else None,
                'adaptive': adp_attn.mean(dim=(0, 1, 2)).detach().cpu() if adp_attn is not None else None,
            }
            return x, attention_weights
        return x


class Mlp(nn.Module):
    """
    标准多层感知机 (MLP) 前馈网络
    
    Transformer 中的标准前馈网络结构：
        FFN(x) = W2(Activation(W1(x)))
        
    通常使用两层全连接层，中间层维度是输入维度的 4 倍（mlp_ratio=4），
    使用 GELU 激活函数，并在每层后应用 Dropout。
    
    Args:
        in_features (int): 输入特征维度
        hidden_features (int, optional): 隐藏层维度，默认等于 in_features
        out_features (int, optional): 输出特征维度，默认等于 in_features
        act_layer: 激活函数类，默认 nn.GELU
        drop (float): Dropout 概率
        
    Shape:
        - Input: (..., in_features)
        - Output: (..., out_features)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        # 默认输出维度等于输入维度
        out_features = out_features or in_features
        # 默认隐藏层维度等于输入维度（实际使用时通常设置为 in_features * mlp_ratio）
        hidden_features = hidden_features or in_features
        
        # 第一层：扩展维度（通常扩展 4 倍）
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二层：压缩回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout 正则化
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x (Tensor): 输入特征
            
        Returns:
            Tensor: MLP 输出特征
        """
        x = self.fc1(x)  # 第一层线性变换
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # Dropout
        x = self.fc2(x)  # 第二层线性变换
        x = self.drop(x)  # Dropout
        return x


class GatedMlp(nn.Module):
    """
    SwiGLU 门控前馈网络 (Swish-Gated Linear Unit)
    
    公式：y = W2( SiLU(Vx) ⊙ Ux )
    
    相比标准 MLP，SwiGLU 使用门控机制：
        - Ux: 主路径（类似标准 MLP 的第一层）
        - Vx: 门控路径（控制信息流）
        - SiLU(Vx) ⊙ Ux: 门控激活，SiLU 作为门控函数
        - W2: 输出投影
        
    优势（参考 Shazeer 2020）：
        - 更强的表示能力：门控机制允许模型学习更复杂的非线性变换
        - 更好的梯度流：SiLU 激活函数（Swish）提供平滑的梯度
        - 在 Transformer 中表现优于标准 MLP
        
    SiLU (Swish) 激活函数：
        SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
        
    Args:
        in_features (int): 输入特征维度
        hidden_features (int, optional): 隐藏层维度，默认等于 in_features
        out_features (int, optional): 输出特征维度，默认等于 in_features
        drop (float): Dropout 概率
        
    Shape:
        - Input: (..., in_features)
        - Output: (..., out_features)
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # 主路径投影：Ux
        self.w_proj = nn.Linear(in_features, hidden_features)
        # 门控路径投影：Vx
        self.v_proj = nn.Linear(in_features, hidden_features)
        # 输出投影：W2
        self.out_proj = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        前向传播：SwiGLU 门控机制
        
        Args:
            x (Tensor): 输入特征
            
        Returns:
            Tensor: SwiGLU 输出特征
        """
        # 主路径：Ux
        u = self.w_proj(x)
        # 门控路径：Vx
        v = self.v_proj(x)
        # 门控激活：SiLU(Vx) ⊙ Ux
        # SiLU 是平滑的、非单调的激活函数，提供更好的梯度流
        x = F.silu(v) * u  # 逐元素相乘（门控）
        x = self.drop(x)
        # 输出投影：W2
        x = self.out_proj(x)
        x = self.drop(x)
        return x


class GatedSkipConnection(nn.Module):
    """
    门控 Skip 连接融合模块
    
    自适应加权融合多层编码器特征，使用可学习的门控机制动态选择最重要的层特征。
    相比简单的累加或拼接，门控机制能够根据输入自适应地调整每层的贡献。
    
    工作原理：
        1. 将每层编码器输出投影到统一的 skip_dim 维度
        2. 基于全局池化特征学习每层的重要性权重
        3. 使用 softmax 归一化权重，确保权重和为 1
        4. 加权融合所有层特征
        
    优势：
        - 自适应选择：模型可以学习哪些层对当前输入最重要
        - 多尺度信息：融合浅层细节和深层语义信息
        - 梯度友好：门控机制提供更好的梯度流
        
    Args:
        num_layers (int): 编码器层数
        embed_dim (int): 编码器嵌入维度
        skip_dim (int): Skip 连接维度（融合后的特征维度）
        
    Shape:
        - Input: List of (B, T, N, embed_dim) tensors
        - Output: (B, skip_dim, N, T)
    """
    def __init__(self, num_layers, embed_dim, skip_dim):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.skip_dim = skip_dim
        
        # 每层的投影：将不同层的特征投影到统一的 skip_dim 维度
        self.layer_projs = nn.ModuleList([
            nn.Conv2d(embed_dim, skip_dim, kernel_size=1) for _ in range(num_layers)
        ])
        
        # 门控网络：学习每层的重要性权重
        # 输入：全局池化后的特征 (skip_dim)
        # 输出：每层的权重 (num_layers)
        self.gate_net = nn.Sequential(
            nn.Linear(skip_dim, skip_dim // 4),  # 降维
            nn.ReLU(),
            nn.Linear(skip_dim // 4, num_layers),  # 输出每层权重
        )
        
    def forward(self, layer_outputs):
        """
        前向传播：门控融合多层特征
        
        Args:
            layer_outputs: 编码器各层输出列表，每个元素 shape (B, T, N, embed_dim)
            
        Returns:
            Tensor: 融合后的特征，shape (B, skip_dim, N, T)
        """
        B, T, N, D = layer_outputs[0].shape
        
        # ========== 步骤1: 投影所有层特征 ==========
        # 将每层特征投影到统一的 skip_dim 维度
        projected = []
        for i, layer_out in enumerate(layer_outputs):
            # 转置为 Conv2d 格式：(B, T, N, D) -> (B, D, N, T)
            # 然后投影：(B, D, N, T) -> (B, skip_dim, N, T)
            proj = self.layer_projs[i](layer_out.permute(0, 3, 2, 1))
            projected.append(proj)
        
        # ========== 步骤2: 堆叠所有层特征 ==========
        # (num_layers, B, skip_dim, N, T)
        stacked = torch.stack(projected, dim=0)
        
        # ========== 步骤3: 计算门控权重 ==========
        # 基于全局特征学习每层的重要性
        
        # 对空间和时间维度池化：(num_layers, B, skip_dim, N, T) -> (B, num_layers, skip_dim)
        stacked_permuted = stacked.permute(1, 0, 2, 3, 4)  # (B, num_layers, skip_dim, N, T)
        pooled = stacked_permuted.mean(dim=[3, 4])  # 空间和时间维度平均：(B, num_layers, skip_dim)
        pooled = pooled.mean(dim=1)  # 平均所有层：(B, skip_dim) - 全局特征
        
        # 通过门控网络生成每层权重：(B, skip_dim) -> (B, num_layers)
        gate_weights = self.gate_net(pooled)  # (B, num_layers)
        # Softmax 归一化：确保权重和为 1，且所有权重非负
        gate_weights = F.softmax(gate_weights, dim=1)  # (B, num_layers)
        
        # ========== 步骤4: 加权融合 ==========
        # 扩展权重维度以匹配 stacked 的形状
        gate_weights = gate_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, num_layers, 1, 1, 1)
        gate_weights = gate_weights.transpose(0, 1)  # (num_layers, B, 1, 1, 1)
        
        # 加权求和：逐元素相乘后求和
        # (num_layers, B, skip_dim, N, T) * (num_layers, B, 1, 1, 1) -> (B, skip_dim, N, T)
        fused = (stacked * gate_weights).sum(dim=0)  # 在层维度求和
        
        return fused


# ============================================================================
# 时空编码器块 (Spatiotemporal Encoder Block)
# ============================================================================
class STEncoderBlock(nn.Module):
    """
    Spatiotemporal Encoder Block
    
    架构设计：
        输入 → STSelfAttention → MLP → 输出
        每个模块都使用残差连接和层归一化
        
    架构特点：
        - STSelfAttention: 四分支自注意力（GQA优化）捕获多粒度时空依赖
        - MLP: SwiGLU或标准MLP进行特征变换
        - DropPath: 随机深度正则化
        
    Args:
        dim (int): 特征维度
        s_attn_size (int): 空间注意力窗口大小
        t_attn_size (int): 时间注意力窗口大小
        geo_num_heads (int): 地理注意力头数
        sem_num_heads (int): 语义注意力头数
        t_num_heads (int): 时间注意力头数
        geo_num_groups (int): 地理注意力GQA分组数
        sem_num_groups (int): 语义注意力GQA分组数
        mlp_ratio (float): MLP隐藏层扩张比例
        qkv_bias (bool): QKV投影是否使用偏置
        drop (float): Dropout概率
        attn_drop (float): 注意力Dropout概率
        drop_path (float): DropPath概率
        act_layer: 激活函数类
        norm_layer: 归一化层类
        device: 计算设备
        type_ln (str): 归一化类型，'pre'或'post'
        output_dim (int): 输出维度
        use_swiglu_ffn (bool): 是否使用SwiGLU门控FFN
    """
    def __init__(
        self, dim, s_attn_size, t_attn_size, geo_num_heads=4, sem_num_heads=2, adp_num_heads=2, t_num_heads=2, 
        geo_num_groups=2, sem_num_groups=1, adp_num_groups=1, adp_topk=10,
        mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
        act_layer=nn.GELU, norm_layer=nn.LayerNorm, device=torch.device('cpu'), 
        type_ln="pre", output_dim=1, use_swiglu_ffn=True, use_layerscale=False, ls_init_value=1e-6,
        use_time_guide=False, time_guide_dilations=(1, 2, 4, 8)
    ):
        super().__init__()
        self.type_ln = type_ln
        self.use_swiglu_ffn = use_swiglu_ffn
        self.use_time_guide = use_time_guide
        
        # ========== 第一阶段：时空自注意力（GQA优化）==========
        self.norm1 = norm_layer(dim)
        self.st_attn = STSelfAttention(
            dim, s_attn_size, t_attn_size, geo_num_heads=geo_num_heads, 
            sem_num_heads=sem_num_heads, adp_num_heads=adp_num_heads, t_num_heads=t_num_heads,
            geo_num_groups=geo_num_groups, sem_num_groups=sem_num_groups, adp_num_groups=adp_num_groups, adp_topk=adp_topk,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, 
            device=device, output_dim=output_dim,
        )
        
        # ========== 第二阶段：多尺度时间精炼器（可选）==========
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.use_time_guide:
            self.norm2 = norm_layer(dim)
            self.time_refiner = MultiScaleTemporalRefiner(
                channels=dim,
                dilations=time_guide_dilations,
                kernel_size=5,
                dropout=drop
            )
        
        # ========== 第三阶段：MLP前馈网络 ==========
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if self.use_swiglu_ffn:
            self.mlp = GatedMlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        # LayerScale: 提升稳定性与精度
        self.use_layerscale = use_layerscale
        if self.use_layerscale:
            self.gamma1 = nn.Parameter(torch.ones(dim) * ls_init_value)  # STSelfAttention
            if self.use_time_guide:
                self.gamma2 = nn.Parameter(torch.ones(dim) * ls_init_value)  # MultiScaleTemporalRefiner
            self.gamma3 = nn.Parameter(torch.ones(dim) * ls_init_value)  # MLP
        else:
            self.gamma1 = None
            if self.use_time_guide:
                self.gamma2 = None
            self.gamma3 = None

    def forward(self, x, geo_mask=None, sem_mask=None):
        """
        前向传播：三阶段处理流程
        
        架构流程：
            STSelfAttention → [MultiScaleTemporalRefiner] → MLP
        
        归一化位置：
            - 'pre': 归一化在模块之前（Pre-LN），更稳定，适合深层网络
            - 'post': 归一化在模块之后（Post-LN），原始 Transformer 设计
        
        Args:
            x (Tensor): 输入特征，shape (B, T, N, dim)
            geo_mask (Tensor, optional): 地理掩码，shape (N, N)，屏蔽远距离节点
            sem_mask (Tensor, optional): 语义掩码，shape (N, N)，屏蔽不相似节点
            
        Returns:
            Tensor: 输出特征，shape (B, T, N, dim)
        """
        if self.type_ln == 'pre':
            # Pre-LN 架构：归一化在模块之前，更稳定
            
            # ========== 第一阶段：时空自注意力 ==========
            # Pre-LN: 先归一化，再计算注意力
            st_out = self.st_attn(self.norm1(x), geo_mask=geo_mask, sem_mask=sem_mask)
            # LayerScale: 可选的缩放因子，提升训练稳定性
            if self.use_layerscale:
                st_out = st_out * self.gamma1
            # 残差连接 + DropPath
            x = x + self.drop_path(st_out)
            
            # ========== 第二阶段：多尺度时间精炼器（可选）==========
            if self.use_time_guide:
                # Pre-LN: 先归一化，再精炼
                time_out = self.time_refiner(self.norm2(x))
                if self.use_layerscale:
                    time_out = time_out * self.gamma2
                x = x + self.drop_path(time_out)
            
            # ========== 第三阶段：MLP前馈网络 ==========
            # Pre-LN: 先归一化，再前馈
            mlp_out = self.mlp(self.norm3(x))
            if self.use_layerscale:
                mlp_out = mlp_out * self.gamma3
            x = x + self.drop_path(mlp_out)
            
        elif self.type_ln == 'post':
            # Post-LN 架构：归一化在模块之后，原始 Transformer 设计
            
            # ========== 第一阶段：时空自注意力 ==========
            # Post-LN: 先计算注意力，再归一化
            st_out = self.st_attn(x, geo_mask=geo_mask, sem_mask=sem_mask)
            if self.use_layerscale:
                st_out = st_out * self.gamma1
            # 残差连接 + DropPath + 归一化
            x = self.norm1(x + self.drop_path(st_out))
            
            # ========== 第二阶段：多尺度时间精炼器（可选）==========
            if self.use_time_guide:
                # Post-LN: 先精炼，再归一化
                time_out = self.time_refiner(x)
                if self.use_layerscale:
                    time_out = time_out * self.gamma2
                x = self.norm2(x + self.drop_path(time_out))
            
            # ========== 第三阶段：MLP前馈网络 ==========
            # Post-LN: 先前馈，再归一化
            mlp_out = self.mlp(x)
            if self.use_layerscale:
                mlp_out = mlp_out * self.gamma3
            x = self.norm3(x + self.drop_path(mlp_out))
            
        return x


# ============================================================================
# TrafficFormer 主模型 (Main Model)
# ============================================================================
class TrafficFormer(AbstractTrafficStateModel):
    """
    TrafficFormer: Spatiotemporal Transformer for Traffic Flow Prediction
    
    模型架构：
        输入嵌入 → [STEncoderBlock × N] → Skip连接融合 → 输出投影
        
    每个STEncoderBlock包含：
        - STSelfAttention: 四分支自注意力(Temporal, Geographic, Semantic, Adaptive) with GQA优化
        - MLP: SwiGLU或标准前馈网络
        
    主要特性：
        1. 多粒度时空建模：同时捕获时间、地理和语义依赖
        2. GQA优化注意力：Geographic/Semantic使用分组查询注意力，Temporal使用多查询注意力
        3. 四分支注意力架构：STSelfAttention (Temporal + Geographic + Semantic + Adaptive) → MLP
        4. Skip连接：融合所有编码器层的输出
        5. 辅助损失：中间层预测头提升表示学习
        6. 复合损失：支持MAE+MAPE, MAE+RMSE等组合
        
    GQA优势（参考论文：GQA: Training Generalized Multi-Query Transformer Models）：
        - 减少KV缓存：Geographic/Semantic的K/V参数量减少50%（默认配置）
        - 提升推理速度：减少内存带宽需求，加速自回归生成
        - 保持模型质量：相比MQA，GQA保留更多表达能力
        
    输入输出：
        Input: (B, T_in, N, F) - 历史时空观测
            B: batch size
            T_in: 输入时间步数
            N: 空间节点数  
            F: 特征维度(流量+时间特征)
        Output: (B, T_out, N, C) - 未来时空预测
            T_out: 输出时间步数
            C: 输出通道数(如inflow/outflow)
    
    Args:
        config: 配置字典
        data_feature: 数据特征字典
    """
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get("num_nodes", 1)
        self.feature_dim = self.data_feature.get("feature_dim", 1)
        self.ext_dim = self.data_feature.get("ext_dim", 0)
        self.num_batches = self.data_feature.get('num_batches', 1)
        self.dtw_matrix = self.data_feature.get('dtw_matrix')
        self.adj_mx = data_feature.get('adj_mx')
        sd_mx = data_feature.get('sd_mx')
        sh_mx = data_feature.get('sh_mx')
        self._logger = getLogger(__name__)
        self.dataset = config.get('dataset')

        self.embed_dim = config.get('embed_dim', 80)
        self.skip_dim = config.get("skip_dim", 256)
        lape_dim = config.get('lape_dim', 8)
        geo_num_heads = config.get('geo_num_heads', 4)
        sem_num_heads = config.get('sem_num_heads', 2)
        adp_num_heads = config.get('adp_num_heads', 2)
        t_num_heads = config.get('t_num_heads', 2)
        # GQA配置：分组查询注意力
        geo_num_groups = config.get('geo_num_groups', 2)  # Geographic分组数，默认2（4头分成2组）
        sem_num_groups = config.get('sem_num_groups', 1)  # Semantic分组数，默认1（2头共享1组KV，等价于MQA）
        adp_num_groups = config.get('adp_num_groups', 1)  # Adaptive分组数，默认1（等价于MQA）
        adp_topk = config.get('adp_topk', 10)  # 自适应图Top-K邻居数
        mlp_ratio = config.get("mlp_ratio", 4)
        qkv_bias = config.get("qkv_bias", True)
        drop = config.get("drop", 0.)
        attn_drop = config.get("attn_drop", 0.)
        drop_path = config.get("drop_path", 0.3)
        self.s_attn_size = config.get("s_attn_size", 3)
        self.t_attn_size = config.get("t_attn_size", 1)
        enc_depth = config.get("enc_depth", 6)
        type_ln = config.get("type_ln", "post")
        self.type_short_path = config.get("type_short_path", "hop")

        self.output_dim = config.get('output_dim', 1)
        self.input_window = config.get("input_window", 12)
        self.output_window = config.get('output_window', 12)
        add_time_in_day = config.get("add_time_in_day", True)
        add_day_in_week = config.get("add_day_in_week", True)
        self.device = config.get('device', torch.device('cpu'))
        self.world_size = config.get('world_size', 1)
        self.huber_delta = config.get('huber_delta', 1)
        self.quan_delta = config.get('quan_delta', 0.25)
        self.far_mask_delta = config.get('far_mask_delta', 5)
        self.dtw_delta = config.get('dtw_delta', 5)
        
        # 复合损失函数权重配置
        self.loss_w_mae = config.get('loss_w_mae', 0.6)
        self.loss_w_rmse = config.get('loss_w_rmse', 0.3)
        self.loss_w_mape = config.get('loss_w_mape', 0.4)

        self.max_epoch = config.get('max_epoch', 200)
        
        # 模型架构配置：四分支注意力结构 STSelfAttention (GQA) → MLP
        self.use_swiglu_ffn = config.get('use_swiglu_ffn', True)
        self.use_layerscale = config.get('use_layerscale', False)
        self.ls_init_value = config.get('ls_init_value', 1e-6)
        
        # 多尺度时间精炼器配置
        self.use_time_guide = config.get('use_time_guide', False)
        self.time_guide_dilations = config.get('time_guide_dilations', (1, 2, 4, 8))
        self._logger.info('⚡ 使用四分支注意力架构（GQA优化）：STSelfAttention (Temporal + Geographic + Semantic + Adaptive)')
        self._logger.info(f'📊 GQA配置: Geographic({geo_num_heads}头/{geo_num_groups}组), Semantic({sem_num_heads}头/{sem_num_groups}组), Adaptive({adp_num_heads}头/{adp_num_groups}组), Temporal(MQA)')
        self._logger.info(f'🔗 自适应图配置: Top-K={adp_topk} (动态学习节点关系)')
        if self.use_swiglu_ffn:
            self._logger.info('🟢 使用 SwiGLU GatedMlp 作为前馈网络')
        else:
            self._logger.info('🟠 使用标准 MLP 作为前馈网络')
        if self.use_time_guide:
            self._logger.info(f'🌊 启用多尺度时间精炼器: dilations={self.time_guide_dilations}')
        else:
            self._logger.info('⏱️ 未启用多尺度时间精炼器')

        if self.type_short_path == "dist":
            distances = sd_mx[~np.isinf(sd_mx)].flatten()
            std = distances.std()
            sd_mx = np.exp(-np.square(sd_mx / std))
            self.far_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.far_mask[sd_mx < self.far_mask_delta] = 1
            self.far_mask = self.far_mask.bool()
            # dist 模式下不使用 geo/sem 掩码
            self.geo_mask = None
            self.sem_mask = None
        else:
            sh_mx = sh_mx.T
            self.geo_mask = torch.zeros(self.num_nodes, self.num_nodes).to(self.device)
            self.geo_mask[sh_mx >= self.far_mask_delta] = 1
            self.geo_mask = self.geo_mask.bool()
            self.sem_mask = torch.ones(self.num_nodes, self.num_nodes).to(self.device)
            sem_mask = self.dtw_matrix.argsort(axis=1)[:, :self.dtw_delta]
            for i in range(self.sem_mask.shape[0]):
                self.sem_mask[i][sem_mask[i]] = 0
            self.sem_mask = self.sem_mask.bool()

        self.enc_embed_layer = DataEmbedding(
            self.feature_dim - self.ext_dim, self.embed_dim, lape_dim, self.adj_mx, drop=drop,
            add_time_in_day=add_time_in_day, add_day_in_week=add_day_in_week, device=self.device,
        )

        # ========== 构建编码器层（四分支注意力架构 + GQA优化 + 多尺度时间引导）==========
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path, enc_depth)]
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                dim=self.embed_dim, s_attn_size=self.s_attn_size, t_attn_size=self.t_attn_size, 
                geo_num_heads=geo_num_heads, sem_num_heads=sem_num_heads, adp_num_heads=adp_num_heads, t_num_heads=t_num_heads,
                geo_num_groups=geo_num_groups, sem_num_groups=sem_num_groups, adp_num_groups=adp_num_groups, adp_topk=adp_topk,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop, 
                drop_path=enc_dpr[i], act_layer=nn.GELU,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), device=self.device, 
                type_ln=type_ln, output_dim=self.output_dim, use_swiglu_ffn=self.use_swiglu_ffn,
                use_layerscale=self.use_layerscale, ls_init_value=self.ls_init_value,
                use_time_guide=self.use_time_guide, time_guide_dilations=self.time_guide_dilations
            ) for i in range(enc_depth)
        ])

        # ========== Skip连接 ==========
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=self.embed_dim, out_channels=self.skip_dim, kernel_size=1,
            ) for _ in range(enc_depth)
        ])

        # ========== 输出解码器 ==========
        self.end_conv1 = nn.Conv2d(
            in_channels=self.input_window, out_channels=self.output_window, kernel_size=1, bias=True,
        )
        self.end_conv2 = nn.Conv2d(
            in_channels=self.skip_dim, out_channels=self.output_dim, kernel_size=1, bias=True,
        )

    def forward(self, batch, lap_mx=None):
        """
        TrafficFormer 前向传播
        
        完整的预测流程：
            1. 输入嵌入：将原始特征转换为嵌入表示（值+位置+时间+空间）
            2. 多层编码器：通过堆叠的 STEncoderBlock 提取时空特征
            3. Skip 连接融合：累加所有编码器层的输出，保留多尺度信息
            4. 时间投影：将输入时间步映射到输出时间步
            5. 通道投影：将特征维度映射到输出维度（如流量通道数）
            
        Args:
            batch (dict): 批次数据字典
                - 'X': 输入特征，shape (B, T_in, N, F)
                    B - batch size
                    T_in - 输入时间步数（历史观测长度）
                    N - 节点数（空间区域数）
                    F - 特征维度（流量特征 + 时间特征）
            lap_mx (Tensor, optional): 拉普拉斯特征矩阵，shape (N, lape_dim)
                用于空间位置编码，如果为 None 则不使用空间嵌入
                
        Returns:
            Tensor: 预测结果，shape (B, T_out, N, C)
                T_out - 输出时间步数（预测长度）
                C - 输出通道数（如 inflow/outflow）
        """
        x = batch['X']  # 提取输入特征
        
        # ========== 步骤1: 数据嵌入 ==========
        # 整合多种嵌入信息：
        #   - 值嵌入：交通流量等数值特征
        #   - 位置编码：时间序列位置信息
        #   - 时间嵌入：一天中的时刻（可选）
        #   - 星期嵌入：星期几（可选）
        #   - 空间嵌入：节点的图结构位置（如果提供 lap_mx）
        # 维度变换：(B, T_in, N, F) -> (B, T_in, N, embed_dim)
        enc = self.enc_embed_layer(x, lap_mx)
        
        # ========== 步骤2: 编码器堆叠 + Skip 连接 ==========
        # 通过多层编码器提取特征，同时累加每层的输出用于 Skip 连接
        # Skip 连接能够保留浅层的细节信息和深层的语义信息
        skip = 0  # 初始化 Skip 连接累加器
        for i, encoder_block in enumerate(self.encoder_blocks):
            # 编码器块处理：STSelfAttention → [MultiScaleTemporalRefiner] → MLP
            enc = encoder_block(enc, self.geo_mask, self.sem_mask)
            
            # 累加 Skip 连接：
            # 1. 转置为 Conv2d 格式：(B, T, N, embed_dim) -> (B, embed_dim, N, T)
            # 2. 投影到 skip_dim：(B, embed_dim, N, T) -> (B, skip_dim, N, T)
            # 3. 累加到 skip 变量
            skip = skip + self.skip_convs[i](enc.permute(0, 3, 2, 1))
        
        # ========== 步骤3: 解码生成预测 ==========
        # 使用两个卷积层完成时间维度和通道维度的投影
        
        # 时间投影：将输入时间步映射到输出时间步
        # 1. 转置：(B, skip_dim, N, T_in) -> (B, T_in, N, skip_dim)
        # 2. 时间维度投影：(B, T_in, N, skip_dim) -> (B, T_out, N, skip_dim)
        #    使用 1x1 卷积在时间维度上投影（实际上是通过通道维度实现）
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        
        # 通道投影：将特征维度映射到输出维度
        # 1. 转置：(B, T_out, N, skip_dim) -> (B, skip_dim, N, T_out)
        # 2. 通道投影：(B, skip_dim, N, T_out) -> (B, output_dim, N, T_out)
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        
        # 恢复时空维度：转置为最终输出格式
        # (B, output_dim, N, T_out) -> (B, T_out, N, output_dim)
        output = skip.permute(0, 3, 2, 1)
        
        return output

    def get_loss_func(self, set_loss):
        """
        损失函数
        
        支持单一损失和复合损失函数
        
        Args:
            set_loss (str): 损失函数名称
            
        Returns:
            loss_func: 损失函数对象
        """
        # 支持的损失函数列表
        supported_losses = ['mae', 'mse', 'rmse', 'mape', 'logcosh', 'huber', 'quantile', 'masked_mae',
                           'masked_mse', 'masked_rmse', 'masked_mape', 'masked_huber', 'r2', 'evar',
                           'mae_rmse', 'composite_mae_rmse', 'mae_rmse_log1p', 'composite_mae_rmse_log1p',
                           'mae_mape', 'composite_mae_mape']
        
        if set_loss.lower() not in supported_losses:
            self._logger.warning('Received unrecognized train loss function, set default mae loss func.')
        
        if set_loss.lower() == 'mae':
            lf = loss.masked_mae_torch
        elif set_loss.lower() == 'mse':
            lf = loss.masked_mse_torch
        elif set_loss.lower() == 'rmse':
            lf = loss.masked_rmse_torch
        elif set_loss.lower() == 'mape':
            lf = loss.masked_mape_torch
        elif set_loss.lower() == 'logcosh':
            lf = loss.log_cosh_loss
        elif set_loss.lower() == 'huber':
            lf = partial(loss.huber_loss, delta=self.huber_delta)
        elif set_loss.lower() == 'quantile':
            lf = partial(loss.quantile_loss, delta=self.quan_delta)
        elif set_loss.lower() == 'masked_mae':
            lf = partial(loss.masked_mae_torch, null_val=0)
        elif set_loss.lower() == 'masked_mse':
            lf = partial(loss.masked_mse_torch, null_val=0)
        elif set_loss.lower() == 'masked_rmse':
            lf = partial(loss.masked_rmse_torch, null_val=0)
        elif set_loss.lower() == 'masked_mape':
            lf = partial(loss.masked_mape_torch, null_val=0)
        elif set_loss.lower() == 'masked_huber':
            lf = partial(loss.masked_huber_loss, delta=self.huber_delta, null_val=0)
        elif set_loss.lower() == 'r2':
            lf = loss.r2_score_torch
        elif set_loss.lower() == 'evar':
            lf = loss.explained_variance_score_torch
        # E1方案：复合损失函数
        elif set_loss.lower() in ("mae_rmse", "composite_mae_rmse"):
            lf = partial(loss.composite_mae_rmse_torch, 
                        w_mae=self.loss_w_mae, w_rmse=self.loss_w_rmse, null_val=0)
        elif set_loss.lower() in ("mae_rmse_log1p", "composite_mae_rmse_log1p"):
            lf = partial(loss.composite_mae_rmse_log1p_torch, 
                        w_mae=self.loss_w_mae, w_rmse=self.loss_w_rmse, null_val=0)
        elif set_loss.lower() in ("mae_mape", "composite_mae_mape"):
            # 直接优化MAPE的复合损失
            lf = partial(loss.composite_mae_mape_torch, 
                        w_mae=self.loss_w_mae, w_mape=self.loss_w_mape, null_val=0)
        else:
            lf = loss.masked_mae_torch
        return lf

    def calculate_loss_without_predict(self, y_true, y_predicted, batches_seen=None, set_loss='masked_mae'):
        """
        计算损失
        
        Args:
            y_true: 真实值
            y_predicted: 预测值
            batches_seen: 已训练批次数（保留参数以兼容调用接口）
            set_loss: 损失函数名称
        """
        lf = self.get_loss_func(set_loss=set_loss)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return lf(y_predicted, y_true)

    def calculate_loss(self, batch, batches_seen=None, lap_mx=None):
        """计算损失"""
        y_true = batch['y']
        y_predicted = self.forward(batch, lap_mx)
        return self.calculate_loss_without_predict(y_true, y_predicted, batches_seen)

    def predict(self, batch, lap_mx=None):
        return self.forward(batch, lap_mx)
