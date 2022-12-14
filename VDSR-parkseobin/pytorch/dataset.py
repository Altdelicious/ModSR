import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from imresize import ImresizeWrapper
import numpy as np
from skimage.color import rgb2ycbcr
import random
from time import time





class SRDatasetOnlyGT(Dataset):
    def __init__(self, GT_path, LR_transform=0.5, lazy_load=True, transform=None, dataset_size=-1, rgb=False):
        self.GT_path = GT_path
        self.lazy_load = lazy_load
        self.transform = transform
        self.LR_transform = ImresizeWrapper(scale_factor=LR_transform, vdsr_like=True) if isinstance(LR_transform, float) else LR_transform
        self.dataset_size = dataset_size
        self.rgb = rgb
        
        self.GT_img = sorted(os.listdir(GT_path))
        self.real_size = len(self.GT_img)
        
        if(not lazy_load):
            start_time = time()
            self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert('RGB')).astype(np.uint8) for gt in self.GT_img]
            print('[*] Time spent on non-lazy loading: {:.1f}s'.format(time() - start_time))
        

    def __len__(self):
        if(self.dataset_size == -1):
            return len(self.GT_img)
        else:
            return self.dataset_size
        

    def __getitem__(self, i):
        if(self.lazy_load):
            GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i % self.real_size])).convert('RGB'))
        else:
            if(i % self.real_size == 0):
                random.shuffle(self.GT_img)
            GT = self.GT_img[i % self.real_size].astype(np.float32)
            #GT = self.GT_img[np.random.randint(self.real_size)].astype(np.float32)

        if(self.transform is not None):
            for tr in self.transform:
                GT = tr(GT)

        LR = self.LR_transform(GT)
        if(not self.rgb):
            GT = GT.astype(np.float32) / 255.
            LR = LR.astype(np.float32) / 255.
            GT = rgb2ycbcr(GT)[:, :, :1]
            LR = rgb2ycbcr(LR)[:, :, :1]

        img_item = {}
        img_item['GT'] = GT.transpose(2, 0, 1).astype(np.float32) / 255.
        img_item['LR'] = LR.transpose(2, 0, 1).astype(np.float32) / 255.

        return img_item
    


class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size
        

    def __call__(self, *inputs):
        ih, iw = inputs[0].shape[:2]
        try:
            ix = random.randrange(0, iw - self.patch_size +1)
            iy = random.randrange(0, ih - self.patch_size +1)
        except(ValueError):
            print('>> patch size: {}'.format(self.patch_size))
            print('>> ih, iw: {}, {}'.format(ih, iw))
            exit()

        output_list = [] 
        for inp in inputs:
            output_list.append(inp[iy : iy + self.patch_size, ix : ix + self.patch_size])
        
        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]



class augmentation(object):
    def __call__(self, *inputs):

        hor_flip = random.randrange(0,2)
        ver_flip = random.randrange(0,2)
        rot = random.randrange(0,2)

        output_list = []
        for inp in inputs:
            if hor_flip:
                tmp_inp = np.fliplr(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if ver_flip:
                tmp_inp = np.flipud(inp)
                inp = tmp_inp.copy()
                del tmp_inp
            if rot:
                inp = inp.transpose(1, 0, 2)
            output_list.append(inp)

        if(len(output_list) > 1):
            return output_list
        else:
            return output_list[0]