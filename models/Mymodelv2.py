import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x):
        # x: [B, L, D]
        # 提取低频趋势项
        trend = self.moving_avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        # 提取高频波动项
        res = x - trend
        return res, trend

# --- 2. 核心改进：空洞卷积精炼器 (Dilated Inception Refiner) ---
# 专门用于降低 MAE，增强对功率跳变点的捕捉
class DilatedInceptionRefiner(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(DilatedInceptionRefiner, self).__init__()
        # 分支1：标准感受野
        self.branch1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, dilation=1),
            nn.GELU()
        )
        # 分支2：扩张感受野 (通过 dilation=2 捕捉更长的时间窗口)
        self.branch2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU()
        )


        # 特征融合
        self.conv_final = nn.Conv1d(d_model * 2, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D] -> [B, D, L]
        x_in = x.transpose(1, 2)
        b1 = self.branch1(x_in)
        b2 = self.branch2(x_in)

        
        # 通道拼接后投影回 d_model
        out = torch.cat([b1,b2], dim=1)
        out = self.conv_final(out).transpose(1, 2)
        
        # 残差连接
        return self.norm(self.dropout(out) + x)
    
# --- 1. 变量注意力机制 (Variable-wise Attention) ---
# 解决特征权重模糊问题
class VariableAttention(nn.Module):
    def __init__(self, enc_in):
        super(VariableAttention, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(enc_in, enc_in),
            nn.ReLU(),
            nn.Linear(enc_in, enc_in),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, L, enc_in]
        # 对变量维度计算注意力权重
        weights = self.gate(x) 
        return x * weights

# --- 2. 梯度增强模块 (Gradient Refiner) ---
# 捕捉功率变化的斜率特征
class GradientRefiner(nn.Module):
    def __init__(self, d_model):
        super(GradientRefiner, self).__init__()
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        # x: [B, L, D]
        # 计算一阶差分
        diff = torch.zeros_like(x)
        diff[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return x + self.proj(diff) # 残差注入

# --- 3. 完整的步进改进模型 ---
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.pred_len = configs.pred_len 
        self.seq_len = configs.seq_len
        # 基础层
        self.var_attn = VariableAttention(configs.enc_in)
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, 
                                          configs.embed_type, configs.freq, configs.dropout)
        self.grad_refiner = GradientRefiner(configs.d_model)  
        self.decomp = SeriesDecomposition(kernel_size=25)
        
        # 波动流 (Mamba + Dilated CNN)
        self.mamba_fluc = Mamba(d_model=configs.d_model, d_state=16, d_conv=4, expand=2)
        self.fluc_refiner = DilatedInceptionRefiner(configs.d_model, configs.dropout)
        
        # 趋势流与投影
        self.trend_projector = nn.Linear(configs.d_model, configs.d_model)
        self.predict_linear_fluc = nn.Linear(configs.seq_len, configs.pred_len)
        self.predict_linear_trend = nn.Linear(configs.seq_len, configs.pred_len)
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc):
        # RevIn 归一化 (逻辑同前)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # 核心改进流程
        x_enc = self.var_attn(x_enc)              # 1. 变量加权
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.grad_refiner(enc_out)       # 2. 梯度增强
        
        res_part, trend_part = self.decomp(enc_out)

        # 波动流与趋势流处理...
        m_out = self.mamba_fluc(res_part)
        res_refined = self.fluc_refiner(m_out)
        fluc_combined = res_refined + res_part

    
        trend_combined = self.trend_projector(trend_part) + trend_part

        # 映射与反归一化...
        fluc_pred = self.predict_linear_fluc(fluc_combined.transpose(1, 2)).transpose(1, 2)
        trend_pred = self.predict_linear_trend(trend_combined.transpose(1, 2)).transpose(1, 2)
        
        dec_out = self.projection(fluc_pred + trend_pred)
        
        # 反归一化 OT 
        dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
