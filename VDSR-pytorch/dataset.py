import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from utils import convert_rgb_to_ycbcr, preprocess_kernels
import PIL.Image as pil_image
from imresize import imresize

class TrainDataset(Dataset):
    def __init__(self, file, scale, crop_size, kernel):
        super(TrainDataset, self).__init__()
        #self.file = file
        self.scale = scale
        self.crop_size = crop_size
        self.Kernel = preprocess_kernels(kernel, sf=self.scale)
        self.hr = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx):
        
        hr = self.hr.astype(np.float32)
        #height, width, _ = self.hr.shape
        #left = random.randint(1, width - self.crop_size)
        #top = random.randint(1, height - self.crop_size)
        #right = left + self.crop_size
        #bottom = top + self.crop_size
        #hr = hr[top:bottom, left:right, :]
        hr = imresize(hr, scale=1./self.scale, kernel=self.Kernel)
        lr = imresize(hr, scale=1./self.scale, kernel=self.Kernel)
        lr = imresize(lr, scale=self.scale, output_shape=hr.shape, kernel='cubic')
        lr = lr.astype(np.float32)
        
        #hr = np.array(hr).astype(np.float32)
        #print(hr.shape)  [h, w, 3]
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

    def __len__(self):
        #with h5py.File(self.h5_file, 'r') as f:
        #    return len(f['lr'])
        return 1


class EvalDataset(Dataset):
    def __init__(self, eval_file, hr_file, scale):
        super(EvalDataset, self).__init__()
        self.lr = eval_file
        self.hr = hr_file
        self.scale = scale

    def __getitem__(self, idx):
        
        
        hr = cv2.cvtColor(cv2.imread(self.hr), cv2.COLOR_BGR2RGB)
        lr = cv2.cvtColor(cv2.imread(self.lr), cv2.COLOR_BGR2RGB)
        lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), cv2.INTER_CUBIC)
        hr = hr.astype(np.float32)
        lr = lr.astype(np.float32)
        
        #hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        #hr = hr.transpose([1, 2, 0])
        #lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        #lr = lr.transpose([1, 2, 0])
        
        #image = pil_image.open(self.file).convert('RGB')
        #image_width = (image.width // self.scale) * self.scale
        #image_height = (image.height // self.scale) * self.scale
        #hr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
        #lr = hr.resize((hr.width // self.scale, hr.height // self.scale), resample=pil_image.BICUBIC)
        #lr = lr.resize((lr.width * self.scale, lr.height * self.scale), resample=pil_image.BICUBIC)
        
        #hr = np.array(hr).astype(np.float32)
        #print(hr.shape)  [h, w, 3]
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

    def __len__(self):
        #with h5py.File(self.h5_file, 'r') as f:
        #    return len(f['lr'])
        return 1
