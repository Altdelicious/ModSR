import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.utils.checkpoint as cp
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import numpy as np
import os
import copy
from scipy.io import savemat
from tqdm import tqdm
import cv2
import time
import re
from collections import OrderedDict

from meta import MetaLearner
from naive import VDSR_mod, VDSR
from dataGenerator import dataGenerator
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, calc_ssim, AverageMeter, GradLoss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', type=str, default='../../data/DIV2K_train/HR')
parser.add_argument('--weights_file', type=str, default='../VDSR-pytorch/pretrained_models/vdsr_x2.pth')
parser.add_argument('--meta_batch_size', type=int, default=3)
parser.add_argument('--meta_lr', type=float, default=1e-3)
parser.add_argument('--num_updates', type=int, default=5)
parser.add_argument('--task_batch_size', type=int, default=5)
parser.add_argument('--crop_size', type=int, default=96)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--num_iter', type=int, default=5000)
parser.add_argument('--output_file', type=str, default='output.pth')
args = parser.parse_args()


batch_size = args.meta_batch_size * args.task_batch_size * 2
device0, device1 = 'cuda:0', 'cuda:2'

train_set = dataGenerator(args.dataset_dir, args.meta_batch_size, args.task_batch_size, args.crop_size, args.scale)
meta = MetaLearner(VDSR, args.weights_file, meta_batchsz=args.meta_batch_size, beta=args.meta_lr, num_updates=args.num_updates,
                   device0=device0, device1=device1, output_file = args.output_file)

for episode_num in range(args.num_iter):
    inputa, labela, inputb, labelb = train_set.__getitem__()
    inputa, inputb = inputa.to(device0), inputb.to(device0)
    labela, labelb = labela.to(device0), labelb.to(device0)
    
    loss = meta(inputa, labela, inputb, labelb)
    print('episode_num:{}, dummy-loss:{:.4f}'.format(episode_num, loss))
