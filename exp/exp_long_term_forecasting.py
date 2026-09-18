from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.losses import PinballLoss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import torch.nn.functional as F
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import matplotlib.pyplot as plt
from models import Mymodel,Mymodelv3, Mymodelv2
warnings.filterwarnings('ignore')


def add_gaussian_noise(x, noise_level):

    noise = torch.randn_like(x)

    x_noisy = x + noise_level * noise

    return x_noisy



class PVSmartCriterion(nn.Module):
    """
    专门为光伏指标优化的智能损失函数
    组合了 Log-Cosh 的平滑性和 动态时段加权
    """
    def __init__(self, mask_threshold=0.01):
        super(PVSmartCriterion, self).__init__()
        self.mask_threshold = mask_threshold

    def forward(self, pred, true):
        # 1. 计算基础误差
        diff = pred - true
        
        # 2. 使用 Log-Cosh 替代 MSE (对离群点更鲁棒，比 Huber 更平滑)
        # log(cosh(x)) 在 x 较小时近似 0.5 * x^2，较大时近似 |x|
        base_loss = torch.log(torch.cosh(diff + 1e-12))
        
        # 3. 动态权重分配 (关键创新)
        # 判定是否为“有效发电时段”
        # 我们不使用硬截断，而是使用一个平滑的权重曲线
        weight = torch.where(true > self.mask_threshold, 
                             1.2, # 发电时段权重稍高，提升 R2
                             0.5) # 非发电时段权重降低，过滤传感器噪声
        
        # 4. 结合权重
        weighted_loss = weight * base_loss
        
        return torch.mean(weighted_loss)

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, true):
        ey_t = pred - true
        # 数值稳定的 log(cosh(x)) 实现
        loss = torch.abs(ey_t) + F.softplus(-2. * torch.abs(ey_t)) - torch.log(torch.tensor(2.0))
        return torch.mean(loss)


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = Mymodelv2.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
  
    def _select_criterion(self):
        criterion = nn.L1Loss()
        return criterion


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs =  self.model(batch_x, batch_x_mark)
                else:
                    outputs =  self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

               
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark)
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:, :]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark)




                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # 防止 Mamba 引起的爆炸
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0, noise_level = 0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    
                batch_x = batch_x.float().to(self.device)
                # ==========================
# Add Gaussian Noise
# ==========================
                if noise_level > 0:

                    batch_x = add_gaussian_noise(
                        batch_x,
                        noise_level
                    )

                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder


                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs =  self.model(batch_x, batch_x_mark)
                else: 
                    outputs =  self.model(batch_x, batch_x_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                







                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                batch_x = batch_x.detach().cpu().numpy()
                #outputs_np = outputs.detach().cpu().numpy()
                #batch_y_np = batch_y.detach().cpu().numpy()
                #outputs = outputs_np
                #batch_y = batch_y_np


                outputs_inv = outputs
                batch_y_inv = batch_y
                batch_x_inv = batch_x


                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)





                    input_shape = batch_x.shape
                    batch_x_inv = test_data.inverse_transform(batch_x.reshape(-1, input_shape[-1])).reshape(input_shape)
                else:
                    outputs_inv = outputs
                    batch_y_inv = batch_y
                    batch_x_inv = batch_x




                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]




                batch_x_target = batch_x_inv





                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)




                inputs.append(batch_x_target)




                if i % 20 == 0:
                    input = batch_x
                    #input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)




        inputs = np.concatenate(inputs, axis=0) # [total_samples, seq_len, 1]









        print('test shape:', preds.shape, trues.shape)
        preds_flat = preds.flatten()
        trues_flat = trues.flatten()
        #preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        #trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        #print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        mean = np.mean(trues)
        rss = np.sum((trues - preds) ** 2)
        tss = np.sum((trues - mean) ** 2)
        r2 = 1 - (rss / tss)
        print('mae:{}, mse:{},rmse:{}, mape:{}, mspe:{}, r2:{},dtw:{}'.format(mae, mse, rmse, mape, mspe, r2,dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mae:{}, mse:{},rmse:{}, mape:{}, mspe:{}, r2:{},dtw:{}'.format(mae, mse, rmse, mape, mspe, r2,dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        #np.save(folder_path + 'pred.npy', preds)
        #np.save(folder_path + 'true.npy', trues)






        inputs = np.array(inputs)








        return mae, mse, rmse, mape, r2
    
    
    def parament_caluate(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"参数总数: {total_params:,}")

        # 区分可训练参数和不可训练参数
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"可训练参数数量: {trainable_params:,}")
    def model_complexity_analysis(self):
        print("\n" + "="*50)
        print(">>>>>>> Model Complexity & Efficiency Analysis >>>>>>>")

        # 1. 精确计算 Parameters（以 K 为单位）
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Parameters: {total_params / 1e3:.2f} K (Trainable: {trainable_params / 1e3:.2f} K)')

        # 2. 理论计算 FLOPs
        seq_len = self.args.seq_len
        d_model = self.args.d_model
        e_layers = getattr(self.args, 'e_layers', 2)
        mixer_flops = e_layers * (seq_len ** 2) * d_model
        total_flops = mixer_flops * 2  # MACs 转 FLOPs
        print(f'FLOPs: {total_flops / 1e6:.2f} M')

        # 3. 评估单次前向推理延迟 (Inference Latency) —— 使用真实 DataLoader 批次
        eval_device = torch.device(f'cuda:{self.args.gpu}' if self.args.use_gpu else 'cpu')
        test_data, test_loader = self._get_data(flag='test')
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))

        self.model.eval()
        with torch.no_grad():
            # 预热 GPU (调用原生的 _process_one_batch，100% 契合模型签名)
            for _ in range(10):
                _ = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            for _ in range(100):
                _ = self._process_one_batch(test_data, batch_x, batch_y, batch_x_mark)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) / 100 * 1000  # 毫秒 (ms)

        print(f'Inference Latency (ms): {inference_time:.2f} ms')
        print('>' * 50)

    def calculate_model_size(self, model, dtype=torch.float32):
        # 获取数据类型的字节数
        if dtype == torch.float32:
            bytes_per_param = 4  # 32位浮点数为4字节
        elif dtype == torch.float64:
            bytes_per_param = 8  # 64位浮点数为8字节
        elif dtype == torch.float16:
            bytes_per_param = 2  # 16位浮点数为2字节
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")
        
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = total_params * bytes_per_param
        total_mb = total_bytes / (1024 * 1024)  # 转换为MB
        print(f"模型参数占用内存: {total_mb:.2f} MB")

    
    def my_Model_EXP(self,settings,path):
        test_data, test_loader = self._get_data(flag='test')
        # best_model_path = "/home/qihui/EXP/Time-Series-Library-main/checkpoints"+settings+"/checkpoint.pth"
        # self.model.load_state_dict(torch.load(best_model_path))
        self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + path, 'checkpoint.pth')))
        self.model.eval()
        preds = []
        trues = []
        inputs = []
        for i in range(10,481,24):
            batch_x,batch_y,batch_x_mark,batch_y_mark = test_data[i]
            batch_x = torch.from_numpy(batch_x).to("cuda:0").unsqueeze(0).float()
            batch_y = torch.from_numpy(batch_y).to("cuda:0").unsqueeze(0).float()
            batch_x_mark = torch.from_numpy(batch_x_mark).to("cuda:0").unsqueeze(0).float()
            batch_y_mark = torch.from_numpy(batch_y_mark).to("cuda:0").unsqueeze(0).float()
            # (pred,season,trend), (true,res,trend) = self._process_one_batch(
            #         test_data,batch_x,batch_y,batch_x_mark)
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :])
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            # outputs = outputs.detach().cpu().numpy()
            # batch_y = batch_y.detach().cpu().numpy()
            # f_dim = -1 if self.args.features == 'MS' else 0
            # outputs= test_data.inverse_transform(outputs.squeeze(0).to("cpu").detach())[:,-1].flatten()
            # batch_y = test_data.inverse_transform(batch_y.squeeze(0).to("cpu").detach())[:,-1].flatten()
            # outputs = outputs[:, -self.args.pred_len:, :]
            # batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            outputs = outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()
            if test_data.scale and self.args.inverse:
                shape = batch_y.shape
                if outputs.shape[-1] != batch_y.shape[-1]:
                    outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs = outputs[:, :, -1:].flatten()/2
            batch_y = batch_y[:, :, -1:].flatten()/2

            # pred = outputs
            # true = batch_y            
            # pred = test_data.inverse_transform(pred.squeeze(0))[:,-1].to("cpu").detach().numpy().flatten()
            # true = test_data.inverse_transform(true.squeeze(0))[:,-1].to("cpu").detach().numpy().flatten()
            preds.extend(outputs)
            trues.extend(batch_y)

            inputs.extend(batch_x.cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        pic, ax = plt.subplots()
        ax.plot(preds, label='pred')
        ax.plot(trues, label='true') 
        ax.legend(loc='upper right')
        plt.show()



