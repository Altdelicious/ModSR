import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
from scipy import signal
from scipy.signal import convolve2d
import scipy.io as sio
from scipy.ndimage import measurements, interpolation
from scipy.io import loadmat
from imresize import imresize
import random
import imageio


def convert_rgb_to_y(img):
    if type(img) == np.ndarray:
        return 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        return 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
    else:
        raise Exception('Unknown Type', type(img))


def convert_rgb_to_ycbcr(img):
    if type(img) == np.ndarray:
        y = 16. + (64.738 * img[:, :, 0] + 129.057 * img[:, :, 1] + 25.064 * img[:, :, 2]) / 256.
        cb = 128. + (-37.945 * img[:, :, 0] - 74.494 * img[:, :, 1] + 112.439 * img[:, :, 2]) / 256.
        cr = 128. + (112.439 * img[:, :, 0] - 94.154 * img[:, :, 1] - 18.285 * img[:, :, 2]) / 256.
        return np.array([y, cb, cr]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        y = 16. + (64.738 * img[0, :, :] + 129.057 * img[1, :, :] + 25.064 * img[2, :, :]) / 256.
        cb = 128. + (-37.945 * img[0, :, :] - 74.494 * img[1, :, :] + 112.439 * img[2, :, :]) / 256.
        cr = 128. + (112.439 * img[0, :, :] - 94.154 * img[1, :, :] - 18.285 * img[2, :, :]) / 256.
        return torch.cat([y, cb, cr], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def convert_ycbcr_to_rgb(img):
    if type(img) == np.ndarray:
        r = 298.082 * img[:, :, 0] / 256. + 408.583 * img[:, :, 2] / 256. - 222.921
        g = 298.082 * img[:, :, 0] / 256. - 100.291 * img[:, :, 1] / 256. - 208.120 * img[:, :, 2] / 256. + 135.576
        b = 298.082 * img[:, :, 0] / 256. + 516.412 * img[:, :, 1] / 256. - 276.836
        return np.array([r, g, b]).transpose([1, 2, 0])
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        r = 298.082 * img[0, :, :] / 256. + 408.583 * img[2, :, :] / 256. - 222.921
        g = 298.082 * img[0, :, :] / 256. - 100.291 * img[1, :, :] / 256. - 208.120 * img[2, :, :] / 256. + 135.576
        b = 298.082 * img[0, :, :] / 256. + 516.412 * img[1, :, :] / 256. - 276.836
        return torch.cat([r, g, b], 0).permute(1, 2, 0)
    else:
        raise Exception('Unknown Type', type(img))


def calc_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    Acknowledgement : https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m)
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
        return h

def calc_ssim(X, Y, scale, dataset=None, sigma=1.5, K1=0.01, K2=0.03, R=255):
    '''
    X : y channel (i.e., luminance) of transformed YCbCr space of X
    Y : y channel (i.e., luminance) of transformed YCbCr space of Y
    Please follow the setting of psnr_ssim.m in EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool, 
    thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2. 
    '''
    gaussian_filter = matlab_style_gauss2D((11, 11), sigma)
    
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if X.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = X.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            X = X.mul(convert).sum(dim=1)
            Y = Y.mul(convert).sum(dim=1)
    else:
        shave = scale + 6
    
    #X = X[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64) 
    #Y = Y[..., shave:-shave, shave:-shave].squeeze().cpu().numpy().astype(np.float64)
    X = X[..., shave:-shave, shave:-shave].astype(np.float64)
    Y = Y[..., shave:-shave, shave:-shave].astype(np.float64)
    
    window = gaussian_filter
    
    ux = signal.convolve2d(X, window, mode='same', boundary='symm')
    uy = signal.convolve2d(Y, window, mode='same', boundary='symm')
    
    uxx = signal.convolve2d(X*X, window, mode='same', boundary='symm')
    uyy = signal.convolve2d(Y*Y, window, mode='same', boundary='symm')
    uxy = signal.convolve2d(X*Y, window, mode='same', boundary='symm')
    
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy
    
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    
    A1, A2, B1, B2 = ((2 * ux * uy + C1, 2 * vxy + C2, ux ** 2 + uy ** 2 + C1, vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D
    mssim = S.mean()
    
    return mssim


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class GradLayer(nn.Module):

    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = [[0, 1, 0],
                    [0, 0, 0],
                    [0, -1, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        # x_list = []
        # for i in range(x.shape[1]):
        #     x_i = x[:, i]
        #     x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
        #     x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
        #     x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
        #     x_list.append(x_i)

        # x = torch.cat(x_list, dim=1)
        if x.shape[1] == 3:
            x = self.get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

class GradLoss(nn.Module):

    def __init__(self):
        super(GradLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.grad_layer = GradLayer()

    def forward(self, output, gt_img):
        output_grad = self.grad_layer(output)
        gt_grad = self.grad_layer(gt_img)
        return self.loss(output_grad, gt_grad)
    

class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, split, crop_size, scaling_factor, kernel_file, lr_img_type, hr_img_type):
        """
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: LR images will be downsampled from the HR images by this factor
        :param lr_img_type: the target format for the LR image; see convert_image() above for available formats
        :param hr_img_type: the target format for the HR image; see convert_image() above for available formats
        """
        self.split = split.lower()
        self.crop_size = crop_size
        self.scaling_factor = scaling_factor
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type
        self.Kernel = preprocess_kernels(kernel_file, sf=self.scaling_factor)

        assert self.split in {'train', 'test'}

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """

        # Crop
        if self.split == 'train':
            # Take a random fixed-size crop of the image, which will serve as the high-resolution (HR) image
            left = random.randint(1, img.width - self.crop_size)
            top = random.randint(1, img.height - self.crop_size)
            right = left + self.crop_size
            bottom = top + self.crop_size
            hr_img = img.crop((left, top, right, bottom))
        else:
            # Take the largest possible center-crop of it such that its dimensions are perfectly divisible by the scaling factor
            x_remainder = img.width % self.scaling_factor
            y_remainder = img.height % self.scaling_factor
            left = x_remainder // 2
            top = y_remainder // 2
            right = left + (img.width - x_remainder)
            bottom = top + (img.height - y_remainder)
            hr_img = img.crop((left, top, right, bottom))

        # Downsize this crop to obtain a low-resolution version of it
        
        hr_img = np.array(hr_img).astype(np.float32)
        lr_img = imresize(hr_img, scale=1./self.scaling_factor, kernel=self.Kernel)
        lr_img = imresize(lr_img, scale=self.scaling_factor, kernel='cubic')
        #lr_img = cv2.resize(lr_img, (hr_img.shape[1], hr_img.shape[0]), cv2.INTER_CUBIC)
        
        #lr_img = hr_img.resize((int(hr_img.width / self.scaling_factor), int(hr_img.height / self.scaling_factor)),
        #                       Image.BICUBIC)

        # Sanity check
        #assert hr_img.width == lr_img.width * self.scaling_factor and hr_img.height == lr_img.height * self.scaling_factor

        # Convert the LR and HR image to the required type
        #lr_img = convert_image(lr_img, source='pil', target=self.lr_img_type)
        #hr_img = convert_image(hr_img, source='pil', target=self.hr_img_type)
        hr_img = convert_rgb_to_ycbcr(hr_img)
        hr_img = hr_img[..., 0]
        hr_img /= 255.
        hr_img = torch.from_numpy(hr_img)
        
        #lr = np.array(lr).astype(np.float32)
        lr_img = convert_rgb_to_ycbcr(lr_img)
        lr_img = lr_img[..., 0]
        lr_img /= 255.
        lr_img = torch.from_numpy(lr_img)
        return (lr_img.unsqueeze(0), hr_img.unsqueeze(0))
    
def preprocess_kernels(kernel, sf=2):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if (kernel == 'cubic'):
        return kernel
    else:
        return kernel_shift(loadmat(kernel)['kernel'], sf)

def move2cpu(d):
    """Move data from gpu to cpu"""
    return d.detach().cpu().float().numpy()


def tensor2im(im_t, normalize_en = False):
    """Copy the tensor to the cpu & convert to range [0,255]"""
    im_np = np.transpose(move2cpu(im_t[0]), (1, 2, 0))  
    if normalize_en:
        im_np = (im_np + 1.0) / 2.0
    im_np = np.clip(np.round(im_np * 255.0), 0, 255)
    return im_np.astype(np.uint8)


def im2tensor(im_np, normalize_en = False):
    """Copy the image to the gpu & converts to range [-1,1]"""
    rgb_range = 255
    #im_np = np.transpose(im_np, (2, 0, 1))
    tensor = torch.from_numpy(im_np).float()
    tensor.mul_(rgb_range / 255)
    #im_np = im_np / 255.0 if im_np.dtype == 'uint8' else im_np
    #if normalize_en:
    #    im_np = im_np * 2.0 - 1.0
    #return torch.FloatTensor(np.transpose(im_np, (2, 0, 1))).unsqueeze(0).cuda()
    return tensor.unsqueeze(0)


def resize_tensor_w_kernel(im_t, k, sf=None):
    """Convolves a tensor with a given bicubic kernel according to scale factor"""
    # Expand dimensions to fit convolution: [out_channels, in_channels, k_height, k_width]
    k = k.expand(im_t.shape[1], im_t.shape[1], k.shape[0], k.shape[1])
    # Calculate padding
    padding = (k.shape[-1] - 1) // 2
    return F.conv2d(im_t, k, stride=round(1 / sf), padding=padding)
    

def read_image(path):
    """Loads an image"""
    im = Image.open(path).convert('RGB')
    im = np.array(im, dtype=np.uint8)
    #im = imageio.imread(path)
    #im = np.ascontiguousarray(im) #(h, w, c) - RGB - (0, 255)
    return im


def rgb2gray(im):
    """Convert and RGB image to gray-scale"""
    return np.dot(im, [0.299, 0.587, 0.114]) if len(im.shape) == 3 else im


def make_1ch(im):
    s = im.shape
    assert s[1] == 3
    return im.reshape(s[0] * 3, 1, s[2], s[3])


def make_3ch(im):
    s = im.shape
    assert s[1] == 1
    return im.reshape(s[0] // 3, 3, s[2], s[3])


def shave_a2b(a, b):
    """Given a big image or tensor 'a', shave it symmetrically into b's shape"""
    # If dealing with a tensor should shave the 3rd & 4th dimension, o.w. the 1st and 2nd
    is_tensor = (type(a) == torch.Tensor)
    r = 2 if is_tensor else 0
    c = 3 if is_tensor else 1
    
    assert (a.shape[r] >= b.shape[r]) and (a.shape[c] >= b.shape[c])
    assert ((a.shape[r] - b.shape[r]) % 2 == 0) and ((a.shape[c] - b.shape[c]) % 2 == 0)
    # Calculate the shaving of each dimension
    shave_r, shave_c = max(0, a.shape[r] - b.shape[r]), max(0, a.shape[c] - b.shape[c])
    return a[:, :, shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2] if is_tensor \
        else a[shave_r // 2:a.shape[r] - shave_r // 2 - shave_r % 2, shave_c // 2:a.shape[c] - shave_c // 2 - shave_c % 2]


def create_gradient_map(im, window=5, percent=.97):
    """Create a gradient map of the image blurred with a rect of size window and clips extreme values"""
    # Calculate gradients
    gx, gy = np.gradient(rgb2gray(im))
    # Calculate gradient magnitude
    gmag, gx, gy = np.sqrt(gx ** 2 + gy ** 2), np.abs(gx), np.abs(gy)
    # Pad edges to avoid artifacts in the edge of the image
    gx_pad, gy_pad, gmag = pad_edges(gx, int(window)), pad_edges(gy, int(window)), pad_edges(gmag, int(window))
    lm_x, lm_y, lm_gmag = clip_extreme(gx_pad, percent), clip_extreme(gy_pad, percent), clip_extreme(gmag, percent)
    # Sum both gradient maps
    grads_comb = lm_x / lm_x.sum() + lm_y / lm_y.sum() + gmag / gmag.sum()
    # Blur the gradients and normalize to original values
    loss_map = convolve2d(grads_comb, np.ones(shape=(window, window)), 'same') / (window ** 2)
    # Normalizing: sum of map = numel
    return loss_map / np.mean(loss_map)


def create_probability_map(loss_map, crop):
    """Create a vector of probabilities corresponding to the loss map"""
    # Blur the gradients to get the sum of gradients in the crop
    blurred = convolve2d(loss_map, np.ones([crop // 2, crop // 2]), 'same') / ((crop // 2) ** 2)
    # Zero pad s.t. probabilities are NNZ only in valid crop centers
    prob_map = pad_edges(blurred, crop // 2)
    # Normalize to sum to 1
    prob_vec = prob_map.flatten() / prob_map.sum() if prob_map.sum() != 0 else np.ones_like(prob_map.flatten()) / prob_map.flatten().shape[0]
    return prob_vec


def pad_edges(im, edge):
    """Replace image boundaries with 0 without changing the size"""
    zero_padded = np.zeros_like(im)
    zero_padded[edge:-edge, edge:-edge] = im[edge:-edge, edge:-edge]
    return zero_padded


def clip_extreme(im, percent):
    """Zeroize values below the a threshold and clip all those above"""
    # Sort the image
    im_sorted = np.sort(im.flatten())
    # Choose a pivot index that holds the min value to be clipped
    pivot = int(percent * len(im_sorted))
    v_min = im_sorted[pivot]
    # max value will be the next value in the sorted array. if it is equal to the min, a threshold will be added
    v_max = im_sorted[pivot + 1] if im_sorted[pivot + 1] > v_min else v_min + 10e-6
    # Clip an zeroize all the lower values
    return np.clip(im, v_min, v_max) - v_min
    

def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def nn_interpolation(im, sf):
    """Nearest neighbour interpolation"""
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize((im.shape[1] * sf, im.shape[0] * sf), Image.NEAREST), dtype=im.dtype)
    
    #return cv2.resize(im, (im.shape[0]*sf, im.shape[1]*sf))

def preprocess_kernels(kernel, sf=2):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if (kernel == 'cubic'):
        return kernel
    else:
        return kernel_shift(loadmat(kernel)['kernel'], sf)

def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel :
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) // 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))
    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass
    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    #kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')
    # Finally shift the kernel and return
    kernel = interpolation.shift(kernel, shift_vec)

    return kernel


def save_final_kernel(k_2, conf):
    """saves the final kernel the results folder"""
    sio.savemat(os.path.join(conf.output_dir, '%s_kernel_x2.mat' % conf.abs_img_name), {'Kernel': k_2})


def calc_curr_k(G_DW_params):
    """given a generator network, the function calculates the kernel it is imitating"""
    curr_k = torch.Tensor([1.]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()
    for ind, w in enumerate(G_DW_params):
        curr_k = F.conv2d(curr_k, w, padding=w.shape[-1]-1) #if ind == 0 else F.conv2d(curr_k, w)
    curr_k = curr_k.squeeze().flip([0, 1])
    return curr_k

      
def downscale_with_kernel(hr_img, kernel, stride=2):
    hr_img = make_1ch(hr_img)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    #padding = (kernel.shape[-1] - 1) // 2
    lr_img = F.conv2d(hr_img, kernel, stride=stride, padding=0)
    lr_img = make_3ch(lr_img)
    return lr_img


def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad