import os
import argparse
# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        # print(torch.cuda.memory_summary())
        solver.test()



    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=int, default=2)
    parser.add_argument('--mask_dot', type=int, default=6)
    parser.add_argument('--series_mask', type=int, default=2)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=25)
    parser.add_argument('--output_c', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pretrained_model', type=str, default=20)
    parser.add_argument('--dataset', type=str, default='PSM')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='dataset/PSM')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=1.00)


    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    main(config)

