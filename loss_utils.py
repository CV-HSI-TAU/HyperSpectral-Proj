# IMPORTS
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from output_visualization import tensor_to_rgb
from model import HyperNet
import matplotlib as cm
import matplotlib.pyplot as plt
from Dataset import AllDataset
from torchvision.transforms import transforms as trans
import torch
import pytorch_ssim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import time
#

def mse(pred, target):
    if torch.is_tensor(pred):
        return torch.mean((pred - target) ** 2)
    else:
        print("not tensor")

def spectral(pred, target): ##[Batch Size, C, H, W]
    pred_grad = pred[:, 2:, :, :] - pred[:, :-2, :, :]
    target_grad = target[:, 2:, :, :] - target[:, :-2, :, :]

    return torch.mean(torch.sum(torch.abs(pred_grad - target_grad)))

def rmse(pred, target, normalization='cube'):

    if pred.ndim == 1:
        pred = pred / pred.max()
        target = target / target.max()

    elif normalization == 'cube':
        if torch.is_tensor(pred):
            if pred.ndim == 3:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)

            pred = pred / torch.max(pred.reshape(pred.shape[0], -1).max(-1)[0][:, None, None, None],
                             torch.tensor(1e-7).to(pred.device).expand_as(pred))
            target = target / torch.max(target.reshape(target.shape[0], -1).max(-1)[0][:, None, None, None],
                               torch.tensor(1e-7).to(target.device).expand_as(target))
        else:
            print("not tensor")
            if pred.ndim == 3:
                pred = np.expand_dims(pred, 0)
                target = np.expand_dims(target, 0)
            pred = pred / pred.max(tuple(range(1, len(pred.shape))))[:, None, None, None]
            target = target / target.max(tuple(range(1, len(target.shape))))[:, None, None, None]

    elif normalization == 'pixel':
        assert torch.is_tensor(pred), 'Inputs must be Pytorch tensors in pixel normalization'
        pred = pred / torch.max(pred.max(1)[0].unsqueeze(1), torch.tensor(1e-7).to(pred.device).expand_as(pred))
        target = target / torch.max(target.max(1)[0].unsqueeze(1), torch.tensor(1e-7).to(target.device).expand_as(target))

    if torch.is_tensor(pred):
        return torch.sqrt(mse(pred, target))
    else:
        return np.sqrt(mse(pred, target))

def calcLoss(pred, target, weights,device):
    #ssim_loss=pytorch_ssim.SSIM(window_size=11)
    #pred_rgb, target_rgb = tensor_to_rgb(pred[0, :, :, :], target[0, :, :, :], False,device)
    #l1loss=nn.L1Loss()
    #rmse_loss = rmse(pred, target) if weights[0]!=0 else 0
    #rgb_loss = (1-ssim_loss(pred_rgb.unsqueeze(0), target_rgb.unsqueeze(0))) if weights[1]!=0 else 0
    #spectral_loss = (1-spectral(pred_rgb.unsqueeze(0), target_rgb.unsqueeze(0))) if weights[2]!=0 else 0
    #l1_loss = l1loss(pred, target) if weights[3]!=0 else 0
    #loss = rmse_loss * weights[0] + rgb_loss * weights[1] + spectral_loss * weights[2] + l1_loss * weights[3]
    #print("rmse loss %.5f, rgb_loss %.5f , spectral loss %.5f , L1 %.9f " % (rmse_loss, weights[1]*rgb_loss, spectral_loss, l1_loss))
    #return loss
    
    ssim_loss=pytorch_ssim.SSIM(window_size=11)
    pred_rgb, target_rgb = tensor_to_rgb(pred, target, False,device)
    #add plot of both of them
    l1loss=nn.L1Loss()
    rmse_loss = rmse(pred_rgb, target_rgb) if weights[0]!=0 else 0
    rgb_loss = (1-ssim_loss(pred_rgb, target_rgb)) if weights[1]!=0 else 0
    spectral_loss = (spectral(pred, target)) if weights[2]!=0 else 0
    l1_loss = l1loss(pred, target) if weights[3]!=0 else 0
    loss = rmse_loss * weights[0] + rgb_loss * weights[1] + spectral_loss * weights[2] + l1_loss * weights[3]
    # print("rmse loss %.5f, rgb_loss %.5f , spectral loss %.5f , L1 %.9f " % (weights[0]*rmse_loss, weights[1]*rgb_loss, weights[2]*spectral_loss, weights[3]*l1_loss))
    return loss