from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader

from PIL import Image

def SSIM(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = 255.0*img1.astype(np.float64)
    img2 = 255.0*img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def PSNR(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = (img1 - img2)*255.0
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return SSIM(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(SSIM(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return SSIM(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')



def get_psnr_ssim(img1,img2):


    data_hazy = (img1.squeeze(0)).permute(1,2,0)
    clean_image = (img2.squeeze(0)).permute(1,2,0)
    data_hazy = data_hazy.cpu().detach().numpy()
    clean_image = clean_image.cpu().detach().numpy()


    ssim = calculate_ssim(data_hazy,clean_image)
    psnr = PSNR(data_hazy,clean_image)

    return psnr,ssim


def get_psnrs_ssims(dir,model_path):
    images = os.listdir(dir)
    psnrs = []
    ssims = []
    for image in images:
        image = os.path.join(dir,image)
        psnr,ssim= get_psnr_ssim(image,model_path)
        psnrs.append(psnr)
        ssims.append(ssim)
    Avarage_SSIM = sum(ssims) / len(ssims)
    Avarage_PSNR = sum(psnrs)/len(psnrs)
    return Avarage_PSNR,Avarage_SSIM





if __name__ == "__main__":
    pass
