import torch 
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1 , Inception_Block_V2


class Multienc_Scale(nn.Module):
    def __init__(self, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_sampling_lauers = nn.ModuleList([
            nn.AvgPool1d(7,2**(i+1),3) for i in range(3)])
        self.upsampl = nn.ModuleList([
            nn.Upsample(size=seq_len, mode='linear') for i in range(3)
        ])

    def forward(self,x):
        out_list = []
        for i in range(3):
            out = self.down_sampling_lauers[i](x.permute(0,2,1)).permute(0,2,1)
            out = self.upsampl[i](out.permute(0,2,1)).permute(0,2,1)
            out_list.append(out)
        return out_list

class Multiori_Scale(nn.Module):
    def __init__(self, seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.down_sampling_lauers = nn.ModuleList([
            nn.AvgPool1d(7,2**i,3) for i in range(3)])

    def forward(self,x):
        out_list = []
        for i in range(3):
            out = self.down_sampling_lauers[i](x.permute(0,2,1)).permute(0,2,1)
            out_list.append(out)
        return out_list 


class Mamba_layer(nn.Module):
    def __init__(self,dim,state,conv, expand, seq_len, pred_len,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mamba = Mamba(d_model=dim, d_state=state, d_conv=conv, expand=expand)
        # self.mamba_pred = nn.Linear(seq_len, pred_len)
    def forward(self, x):
        x = self.mamba(x)
        return x 

class TimesBlock(nn.Module):
    def __init__(self, configs,in_channel):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(in_channel, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, in_channel,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        out = self.conv(x)
        res = out + x
        return res

class Trans22D_layer(nn.Module):
    def __init__(self, fft_dim_max, c_in, d_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_pair = nn.ModuleList([nn.Linear(c_in, d_model) for i in range(3)])

    def forward(self, fft_x_list, mutli_list):
        data_fusion_list = []
        for i in range(3):
#            fft_x_list[i] = self.embed_pair[i](self.dim_pair[i](fft_x_list[i].permute(0,2,1)).permute(0,2,1))
            fft_x_list[i] = self.embed_pair[i](fft_x_list[i])
            data_fusion_list.append(torch.stack((fft_x_list[i],mutli_list[i]),dim=1))

        return data_fusion_list


class Conv2_block(nn.Module):
    def __init__(self, pred_len, en_c, CH,num_kernels=6,init_weight=True):
        super(Conv2_block, self).__init__()
        self.pred_len = pred_len
        self.num_kernels = num_kernels
        self.Linear_ch = nn.Linear(en_c, CH)
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(1, pred_len, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()
        self.Conv2d = nn.Sequential(nn.Conv2d(in_channels=pred_len, out_channels=pred_len, kernel_size=3,stride=2,padding=1),
                                     nn.ReLU(),
                                     nn.Conv2d(pred_len,pred_len,3,2,1))                                     

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.Linear_ch(x)[:, None, :, :]
        B,C,H,W = x.shape
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        res = self.Conv2d(res).reshape(B,self.pred_len,-1)
        return res



class Model(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_len = config.pred_len

        self.embed_target = DataEmbedding(1,config.d_model, embed_type='fixed', freq=config.freq)
        self.embed = DataEmbedding(config.enc_in, config.d_model,embed_type='fixed', freq=config.freq)
        self.embed_multi_scale = Multienc_Scale(config.seq_len)
        self.model = nn.ModuleList([TimesBlock(config,in_channel=3)
                                    for _ in range(config.e_layers)])
        self.layer_norm = nn.LayerNorm((3,config.seq_len,config.d_model))
        self.layer = config.e_layers
        
        self.linear2d_model = nn.Linear(
            3*config.d_model, config.d_model)
        self.lanorm_dmodel = nn.LayerNorm(config.d_model)
        self.act = nn.ReLU()

        self.mamba_model = Mamba_layer(dim=config.d_model, state=32, conv=3, expand=1, seq_len=config.seq_len, pred_len=config.pred_len)
        self.predict_linear = nn.Linear(
            config.seq_len*2, config.pred_len)
        self.projection = nn.Linear(
            config.d_model, config.c_out, bias=True)
        

    def forward(self,x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev


        target_data = x_enc[:,:,-1].unsqueeze(-1)

        #embed multiscal
        x = self.embed_target(target_data, x_mark_enc)
        x_mamba = self.embed(x_enc,x_mark_enc)
        mamba_out = self.mamba_model(x_mamba)
        
        #多尺度处理
        mutli_list = self.embed_multi_scale(x)
        mutli_x = torch.stack(mutli_list,dim=1)

        #获取频率特征
        # freq_trea = self.FFT_for_Period(x)
        # freq_trea = torch.cat((freq_trea,torch.flip(freq_trea, dims=(1,))),dim=1)

        # inx_list = []
        # for i in range(3):
        #     inx_list.append(torch.stack((mutli_list[i],freq_trea),dim=1) )

        # enc_out = []
        for i in range(self.layer):
            # enc_out.append(self.layer_norm(self.model[i](mutli_x)))
            mutli_x = self.layer_norm(self.model[i](mutli_x)) 
    
        # out = torch.cat(enc_out, dim=1)
        B, C, P, D = mutli_x.shape
        out = mutli_x.reshape(B, P, C*D)
        out = self.lanorm_dmodel(self.linear2d_model(out))
        out = self.act(out)

        ''' change ''' 
        # out = self. anorm_dmodel(self.mamba_model(out))
        out = torch.cat([out, mamba_out],dim=1)
        # out = out + mamba_out
        out = self.predict_linear(out.transpose(1,2)).transpose(1,2)
        dec_out = self.projection(out)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len, 1))
        return dec_out

    def FFT_for_Period(self,x, k=2):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf)
        frequency_list = frequency_list[:,1:,:]
        return frequency_list


