import argparse
#from thop import profile
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# from exp.exp_imputation import Exp_Imputation
# from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
# from exp.exp_anomaly_detection import Exp_Anomaly_Detection
# from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import time
from models import Mymodel,Mymodelv3, Mymodelv2
import numpy as np

if __name__ == '__main__':
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed) 

    parser = argparse.ArgumentParser(description='zzmodel')

   # input_x = torch.randn(64, 96, 8).to(device)
   # input_mark = torch.randn(64, 96, 4).to(device) # 假设时间特征是 4 维

    # 2. 调用 profile 计算
    # model 是你实例化后的 Mymodelv2
   # flops, params = profile(Mymodelv2, inputs=(input_x, input_mark))

   # print(f"Total Params: {params / 1e6:.3f} M") # 以百万为单位
   # print(f"Total FLOPs: {flops / 1e9:.3f} G")   # 以十亿为单位

    # basic config
    parser.add_argument('--is_training', type=int,  default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test_v2', help='model id')
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--scale', action='store_false', help='whether to scale data', default=True)
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='XJ.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='t',
                      help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    #parser.add_argument('--inverse', type=str, default=False, help='inverse output data',)
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    parser.add_argument('--embed_type', type=str, default='timeF', help='text embedding type')
    # model define
    parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layrs')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--d_state', type=int, default=9, help='Mamba state space dimension')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=4, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0055732049461341545, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description') 
    parser.add_argument('--loss', type=str, default='MAE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU  
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=True, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=True, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    Exp = Exp_Long_Term_Forecast
    

    if args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                exp = Exp(args)  # set experiments
    
    
    
        #        if ii == 0:  # 仅在第一次迭代时打印，避免重复输出
         #           print('>>>>>>> Model Complexity Analysis >>>>>>>')
          #          # 模拟输入数据：[Batch_size, Seq_len, Features]
           #         # 根据你的 args，Features 是 enc_in (8)，Seq_len 是 96
           #         random_input = torch.randn(1, args.seq_len, args.enc_in).to(args.device)
         #            # 模拟时间特征：[Batch_size, Seq_len, Time_features]
           #          # 默认 timeF 编码通常为 4 维（月、日、周、时）
         #            random_mark = torch.randn(1, args.seq_len, 5).to(args.device) 
                    
         #            try:
          #               flops, params = profile(exp.model, inputs=(random_input, random_mark, ), verbose=False)
         #                print(f'Total Params: {params / 1e6:.4f} M')
          #               print(f'Total FLOPs: {flops / 1e9:.4f} G (Batch=1)')
          #           except Exception as e:
         # #                print(f'Complexity analysis failed: {e}')
         #            print('>' * 40)
    
    
                setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                    args.model_id,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.pred_len,
                    args.d_model,
                    args.d_ff,
                    args.d_conv, ii)
    
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.train(setting)
    
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)
                #siple = "{}_{}_{}".format(args.model_id, args.data_path, args.pred_len)
                # exp.my_Model_EXP(settings=simple,path=setting)
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.model_id,
                args.data,
                args.features,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.d_conv)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    
            if args.use_gpu:
                    torch.cuda.synchronize() # 确保之前所有的 GPU 任务已完成
                
            test_start = time.time()
                
            exp.test(setting) # 执行测试逻辑
                
            if args.use_gpu:
                    torch.cuda.synchronize() # 确保测试任务全部完成
                
            test_end = time.time()
                
            total_test_time = test_end - test_start
                # 这里的 test_loader 长度可以从 exp 对象中获取（取决于你的代码实现）
                # 通常 Runtime 在论文中指：整个测试集跑完的总秒数
            print(f'>>>>>>> Test Runtime: {total_test_time:.4f} s')
    
    
            exp.test(setting, test=1)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    
    
        
    