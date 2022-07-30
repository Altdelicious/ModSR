import numpy as np
import cv2
import torch
import os
import random
from torch.utils.data import Dataset
from utils import convert_rgb_to_ycbcr, preprocess_kernels
import PIL.Image as pil_image
from imresize import imresize
from gkernel import generate_kernel

#Code source (in TF): https://github.com/JWSoh/MZSR/blob/master/dataGenerator.py

class dataGenerator(Dataset):
    def __init__(self, dataset_dir, meta_batch_size, task_batch_size, crop_size, scale):
        super(dataGenerator, self).__init__()
        
        self.dataset_dir = dataset_dir
        self.images = os.listdir(dataset_dir)
        self.task_batch_size = task_batch_size
        self.meta_batch_size = meta_batch_size
        self.scale = scale
        self.crop_size = crop_size
        
    def make_data_tensor(self):
        input_meta, label_meta = [], []
        
        for i in range(self.meta_batch_size):
            input_task, label_task = [], []
            scale = self.scale
            Kernel = generate_kernel(k1=scale*2.5, ksize=15)
            
            for j in range(self.task_batch_size*2):
                idx = i*self.task_batch_size*2 + j
                #print(idx)
                img_file = os.path.join(self.dataset_dir, self.images[idx])
                img_lr, img_hr = self.load_hr_lr(img_file, scale, Kernel, self.crop_size)
                
                input_task.append(img_lr.unsqueeze(0))
                label_task.append(img_hr.unsqueeze(0))
                
            #print(type(input_task[0]), (input_task[0].size()))
                
            input_meta.append(torch.cat(input_task).unsqueeze(0))
            label_meta.append(torch.cat(label_task).unsqueeze(0))
            
            
        input_meta = torch.cat(input_meta)
        label_meta = torch.cat(label_meta)
        #print(input_meta.size())
        
        inputa=input_meta[:,:self.task_batch_size,:,:,:]
        labela=label_meta[:,:self.task_batch_size,:,:,:]
        inputb=input_meta[:,self.task_batch_size:,:,:,:]
        labelb=label_meta[:,self.task_batch_size:,:,:,:]

        return (inputa, labela, inputb, labelb)
    
    def load_hr_lr(self, img_file, scale, Kernel, crop_size):
        #the following image loader is specific for SRCNN
        #this needs to be changed for other architectures
        
        hr = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        height, width, _ = hr.shape
        hr = hr.astype(np.float32)
        left = random.randint(1, width - crop_size)
        top = random.randint(1, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        hr = hr[top:bottom, left:right, :]
        lr = imresize(hr, scale=1./scale, kernel=Kernel)
        lr = imresize(lr, scale=scale, kernel='cubic')
        lr = lr.astype(np.float32)
        
        hr = convert_rgb_to_ycbcr(hr)
        hr = hr[..., 0]
        hr /= 255.
        hr = torch.from_numpy(hr)
        
        #lr = np.array(lr).astype(np.float32)
        lr = convert_rgb_to_ycbcr(lr)
        lr = lr[..., 0]
        lr /= 255.
        lr = torch.from_numpy(lr)
        return (lr.unsqueeze(0), hr.unsqueeze(0))
                

    def __getitem__(self):
        #To be added: the current method will output the same batch in every iteration
        
        random.shuffle(self.images)
        return self.make_data_tensor()
    
    def __len__(self):
        #with h5py.File(self.h5_file, 'r') as f:
        #    return len(f['lr'])
        return len(self.images)