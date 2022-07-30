import torch
import torch.backends.cudnn as cudnn
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import numpy as np
import PIL.Image as pil_image
import os
import copy
from scipy.io import savemat
from tqdm import tqdm
import cv2
import random
import time

from dataset import TrainDataset, EvalDataset
from datasets import SRDataset

from models import VDSR, VDSR_mod
from imresize import imresize
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr, calc_ssim, AverageMeter, preprocess_kernels, GradLoss

# % trainable parameters in VDSR_mod = 18.39

import argparse

def loss_fn(out, gt):
    
    l1_loss = nn.L1Loss()(out, gt)
    
    Grad_Loss = GradLoss()
    Grad_Loss = Grad_Loss.to(device)
    grad_loss = Grad_Loss(out, gt)
    
    a, b = 1.0, 0.1
    pixel_loss = a*l1_loss + b*grad_loss
    
    return pixel_loss

seed = 21
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

def train_single():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = VDSR_mod()
        state_dict = model.state_dict()

        #pre-trained weights loaded in model

        for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
                #if ('block' not in n):
                #    state_dict[n].requires_grad = False
        
        #freezing the weights
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d) and ('block' not in name):
                m.weight.requires_grad = False
                #m.bias.requires_grad = False
        
        #num_total_params = count_total_params(model)
        #num_train_params = count_train_params(model)
        #print("% of trainable parameters:", round((num_train_params*100/num_total_params), 2), '%')
        
        
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    miles = [5]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=miles, gamma=10, last_epoch=-1, verbose=False)

    # Custom dataloaders
    #train_dataset = TrainDataset(train_file, scale, crop_size, kernel_file)
    #train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    #                          drop_last=True)
    
    hr_img = cv2.cvtColor(cv2.imread(train_file), cv2.COLOR_BGR2RGB)
    hr_img = hr_img.astype(np.float32)
    hr_img = imresize(hr_img, scale=1./scale, kernel=Kernel)
    lr_img = imresize(hr_img, scale=1./scale, kernel=Kernel)
    lr_img = imresize(lr_img, scale=scale, output_shape=hr_img.shape, kernel='cubic')
    lr_img = lr_img.astype(np.float32)
    
    hr_img = convert_rgb_to_ycbcr(hr_img)
    hr_img = hr_img[..., 0]
    hr_img /= 255.
    hr_img = torch.from_numpy(hr_img)
    hr_img = hr_img.unsqueeze(0).unsqueeze(0)
    
    #lr = np.array(lr).astype(np.float32)
    lr_img = convert_rgb_to_ycbcr(lr_img)
    lr_img = lr_img[..., 0]
    lr_img /= 255.
    lr_img = torch.from_numpy(lr_img)
    lr_img = lr_img.unsqueeze(0).unsqueeze(0)

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train(lr_img, hr_img, model=model, optimizer=optimizer, epoch=epoch)
        torch.save(model.state_dict(), os.path.join(outputs_dir, args.output_weights_file))
        
        #scheduler.step()
        
        if (epoch==0):
            lst_psnr[idx,0], lst_ssim[idx, 0] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==9):
            lst_psnr[idx,1], lst_ssim[idx, 1] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==19):
            lst_psnr[idx,2], lst_ssim[idx, 2] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==29):
            lst_psnr[idx,3], lst_ssim[idx, 3] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==39):
            lst_psnr[idx,4], lst_ssim[idx, 4] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==49):
            lst_psnr[idx,5], lst_ssim[idx, 5] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==59):
            lst_psnr[idx,6], lst_ssim[idx, 6] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==69):
            lst_psnr[idx,7], lst_ssim[idx, 7] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==79):
            lst_psnr[idx,8], lst_ssim[idx, 8] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==89):
            lst_psnr[idx,9], lst_ssim[idx, 9] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        if (epoch==99):
            lst_psnr[idx,10], lst_ssim[idx, 10] = evaluate_single(dataset_dir, images[idx], model, outputs_dir, scale, Kernel, device)
        
        
    return
        
def train(lr_imgs, hr_imgs, model, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    #for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
    data_time.update(time.time() - start)

    # Move to default device
    lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24)
    hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96)

    # Forward prop.
    sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

    # Loss
    loss = loss_fn(sr_imgs, hr_imgs)  # scalar

    # Backward prop.
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients, if necessary
    if grad_clip is not None:
        clip_gradient(optimizer, grad_clip)

    # Update model
    optimizer.step()

    # Keep track of loss
    losses.update(loss.item(), lr_imgs.size(0))

    # Keep track of batch time
    batch_time.update(time.time() - start)

    # Reset start time
    start = time.time()
    
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    
def evaluate_single(dataset_dir, image, model, output_dir, scale, Kernel, device):
    hr_file = dataset_dir + '/{}'.format(image)
    # = 'data/Set5/LR_noncubic/X2/{}x2.png'.format(image)
    output_file = output_dir + '/{}_x2.png'.format(image[:-4])
    
    hr = pil_image.open(hr_file).convert('RGB')
    hr = np.array(hr).astype(np.float32)
    lr = imresize(hr, scale=1./scale, kernel=Kernel)
    lr = imresize(lr, scale=scale, output_shape=hr.shape, kernel='cubic')
    
    ycbcr_lr = convert_rgb_to_ycbcr(lr)
    
    y_lr = ycbcr_lr[..., 0]
    y_lr /= 255.
    y_lr = torch.from_numpy(y_lr).to(device)
    y_lr = y_lr.unsqueeze(0).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        preds = model(y_lr.float()).clamp(0.0, 1.0)
    
    ycbcr_hr = convert_rgb_to_ycbcr(hr)
    y_hr = ycbcr_hr[..., 0]
    y_hr /= 255.
    y_hr = torch.from_numpy(y_hr).to(device)
    y_hr = y_hr.unsqueeze(0).unsqueeze(0)
    
    psnr = calc_psnr(y_hr, preds)
    psnr = psnr.cpu().numpy()
    
    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    y_hr = y_hr.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
    ssim = calc_ssim(y_hr, preds, scale=2)
    
    if (args.save):
        output = np.array([preds, ycbcr_lr[..., 1], ycbcr_lr[..., 2]]).transpose([1, 2, 0])
        output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
        output = pil_image.fromarray(output)
        output.save(output_file)
    
    return (psnr, ssim)


def count_train_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_total_params(model):
    return sum(p.numel() for p in model.parameters())

#Lrate (with adapters) -> 1e-3 for non-meta-learning, (1e-3)-(1e-2, 5 epochs) for meta-learning

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Urban100', help='name of test dataset from the data directory')
parser.add_argument('--meta_weights_file', type=str, default='../MAML-simple/aniso_5000.pth', help='path to meta-weights file')
parser.add_argument('--scale', type=int, default=2, help='scale of SR')
parser.add_argument('--output_weights_file', type=str, default='best.pth', help='name of output-weights file')
parser.add_argument('--seed', type=int, default=21, help='random seed value')
parser.add_argument('--epochs', type=int, default=100, help='no. of fine-tuning iterations for adapters')
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for adapters, consult the mail file for proper values')
parser.add_argument('--kernel_file', type=str, default='../../data/Set5/LR_noncubic/aniso_1.mat', help='path to blur kernel file for downsampling')
parser.add_argument('--save', type=bool, default=False, help='True or False to save output image after final epoch')
args = parser.parse_args()


dataset = args.dataset
dataset_dir = '../../data/{}/HR'.format(dataset)
weights_file = args.meta_weights_file
outputs_dir = 'output/{}_mod'.format(dataset)
scale = args.scale

kernel_file = args.kernel_file

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 1
start_epoch = 0  # start at this epoch
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True

images = sorted(os.listdir(dataset_dir))
num_test = len(images)
Kernel = preprocess_kernels(kernel_file, sf=scale)

lst_psnr = np.zeros((num_test, 11))       # (num_testx11)
lst_ssim = np.zeros((num_test, 11))

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
    
total_time = 0.0
    
for i in range(num_test):
    train_file = dataset_dir + '/{}'.format(images[i])
    idx = i
    
    start = time.time()
    train_single()
    end = time.time()
    
    #psnr100, ssim100 = evaluate_single(dataset_dir, images[i], model, outputs_dir, scale, Kernel, device)
    if (i!=0):
        total_time = total_time + (end-start)
    
    print('image {} done, time elapsed: {:.3f} seconds'.format(idx, end-start))
    
print('DONE!')
print('Metrics for 1 epoch: {:.4f}, {:.4f}'.format(np.mean(lst_psnr[:,0]), np.mean(lst_ssim[:,0])))
print('Metrics for 20 epoch: {:.4f}, {:.4f}'.format(np.mean(lst_psnr[:,2]), np.mean(lst_ssim[:,2])))
print('Metrics for 50 epoch: {:.4f}, {:.4f}'.format(np.mean(lst_psnr[:,5]), np.mean(lst_ssim[:,5])))
print('Metrics for 100 epoch: {:.4f}, {:.4f}'.format(np.mean(lst_psnr[:,10]), np.mean(lst_ssim[:,10])))
print('Average time: {:.2f}'.format(total_time/num_test))

print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(np.mean(lst_psnr[:,0]), np.mean(lst_psnr[:,1]), np.mean(lst_psnr[:,2]), np.mean(lst_psnr[:,3]), np.mean(lst_psnr[:,4]), np.mean(lst_psnr[:,5]), np.mean(lst_psnr[:,6]), np.mean(lst_psnr[:,7]), np.mean(lst_psnr[:,8]), np.mean(lst_psnr[:,9]), np.mean(lst_psnr[:,10])))

print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}'.format(np.mean(lst_ssim[:,0]), np.mean(lst_ssim[:,1]), np.mean(lst_ssim[:,2]), np.mean(lst_ssim[:,3]), np.mean(lst_ssim[:,4]), np.mean(lst_ssim[:,5]), np.mean(lst_ssim[:,6]), np.mean(lst_ssim[:,7]), np.mean(lst_ssim[:,8]), np.mean(lst_ssim[:,9]), np.mean(lst_ssim[:,10])))
