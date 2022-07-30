import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from imresize import imresize
from utils import ImageTransforms
#from data import DataGenerator


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, split, crop_size, scaling_factor, lr_img_type, hr_img_type, kernel_file, test_data_name=None):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.test_data_name = test_data_name
        self.kernel_file = kernel_file
        self.num_iters = 100

        assert self.split in {'train', 'test'}
        if self.split == 'test' and self.test_data_name is None:
            raise ValueError("Please provide the name of the test dataset!")
        assert lr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        if self.split == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        #if self.split == 'train':
        #with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
        #    self.images = json.load(j)
        self.images = os.listdir(data_folder)
        #else:
        #with open(os.path.join(data_folder, self.test_data_name + '_test_images.json'), 'r') as j:
        #    self.images = json.load(j)
        
        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         kernel_file = self.kernel_file,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)
        
        #self.hr_lr_data = []
        #for i in range(len(self.images)):
        #    img_data = DataGenerator(self.data_folder+'/'+self.images[i], kernel_file)
        #    self.hr_lr_data.append(img_data)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img = Image.open(self.data_folder+'/'+self.images[i], mode='r')
        img = img.convert('RGB')
        if img.width <= 96 or img.height <= 96:
            print(self.images[i], img.width, img.height)
        
        lr_img, hr_img = self.transform(img)
        
        #idx = np.random.randint(self.num_iters)
        #hr_lr = self.hr_lr_data[i].getitem(idx)
        #hr_img, lr_img = hr_lr['HR'], hr_lr['LR']
        
        #return (lr_img.unsqueeze(0), hr_img.unsqueeze(0))
        return (lr_img.float(), hr_img.float())

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :return: size of this data (in number of images)
        """
        return len(self.images)
