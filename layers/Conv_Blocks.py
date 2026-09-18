import torch
import torch.nn as nn


import torch
import torch.nn as nn


class Inception_Block_V1_Improved(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3, init_weight=True):
        super(Inception_Block_V1_Improved, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        # 分支1：保留原3x3卷积（数量与原num_kernels一致，保证基础特征）
        base_kernels = []
        for i in range(num_kernels):
            # 卷积+BN（不改变维度，提升训练稳定性）
            base_kernels.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # padding=1保证H/W不变
                nn.BatchNorm2d(out_channels),  # 归一化不改变形状
                nn.ReLU(inplace=True)
            ))
        self.base_kernels = nn.ModuleList(base_kernels)

        # 分支2：1x1卷积补充小尺度特征（通道数与输出一致，不改变形状）
        self.small_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),  # 1x1不改变H/W
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 分支3：空洞3x3卷积扩展感受野（padding=dilation保证H/W不变）
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),  # 空洞率2，padding=2保证H/W不变
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 可学习权重（所有分支总数量 = num_kernels + 2，权重维度匹配）
        self.branch_weights = nn.Parameter(torch.ones(num_kernels + 2) / (num_kernels + 2))

        # 残差连接（适配通道数，保证形状一致）
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)  # 1x1调整通道，不改变H/W

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 保存原始输入（用于残差连接，保证形状一致）
        x_shortcut = self.shortcut(x)

        # 1. 基础3x3卷积分支（num_kernels个，与原模块一致）
        base_feats = [self.base_kernels[i](x) for i in range(self.num_kernels)]
        
        # 2. 补充多尺度分支（1x1 + 空洞3x3，不改变形状）
        small_feat = self.small_scale(x)
        dilated_feat = self.dilated_conv(x)
        
        # 合并所有分支
        all_feats = base_feats + [small_feat, dilated_feat]
        
        # 可学习权重融合（均值的升级版，不改变形状）
        weights = nn.functional.softmax(self.branch_weights, dim=0)
        res = 0.0
        for i, feat in enumerate(all_feats):
            res += weights[i] * feat
        
        # 残差连接（保证输出形状 = x_shortcut形状 = 原模块输出形状）
        res = res + x_shortcut
        return res






class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):





            k_size = 3 + i * 2





            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=3, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels // 2 * 2 + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    


class ChannelGating(nn.Module):
    #"""创新点1：多变量通道融合门控。动态调整气象因子权重"""
    def __init__(self, enc_in, d_model):
            super(ChannelGating, self).__init__()
            self.gate = nn.Sequential(
                nn.Linear(enc_in, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, enc_in),
                nn.Sigmoid()
            )

    def forward(self, x_raw):
            # x_raw: [B, L, enc_in]
            # 计算每个通道的全局重要性
            avg_pool = torch.mean(x_raw, dim=1) # [B, enc_in]
            gate_weights = self.gate(avg_pool).unsqueeze(1) # [B, 1, enc_in]
            return x_raw * gate_weights

class ProbabilisticHead(nn.Module):
    #"""创新点2：概率输出层。输出三个分位数：[0.05, 0.5, 0.95]"""
    def __init__(self, d_model, c_out):
        super(ProbabilisticHead, self).__init__()
        self.project = nn.Linear(d_model, c_out * 3) # 输出维度翻三倍

    def forward(self, x):
        # x: [B, L, d_model]
        out = self.project(x)
        return out # 返回后需要在外面reshape为 [B, L, C, 3]

