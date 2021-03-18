import torch
import argparse

device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    # 1. arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--iteration', type=int, default=300000)
    parser.add_argument('--port', type=str, default='8097')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--eval_start_step', type=int, default=200000)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--vis_step', type=int, default=10)

    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='DualGradDeSnow')                         # FIXME

    # FIXME choose your dataset root
    parser.add_argument('--root', type=str, default='D:\data\Snow100K')
    # parser.add_argument('--root', type=str, default='D:\data\SRRS')
    # parser.add_argument('--root', type=str, default='/home/cvmlserver3/Sungmin/data/Snow100K')
    # parser.add_argument('--root', type=str, default='/home/cvmlserver4/Sungmin/data/Snow/SRRS')

    parser.add_argument('--data_type', type=str, default='snow100k', help='snow100k or srrs')               # FIXME

    parser.set_defaults(visualization=False)
    parser.add_argument('--vis', dest='visualization', action='store_true')

    opts = parser.parse_args(args)
    return opts