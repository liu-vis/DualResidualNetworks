# python 2.7, pytorch 0.3.1

import os, sys
sys.path.insert(1, '../')
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from pietorch import data_convertors
from pietorch.DuRN_P import cleaner as cleaner
from pietorch.DuRN_P_no_norm import cleaner as cleaner_no_norm
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

#------ Options -------
tag = 'DuRN_P_no_norm' # 'DuRN_P' or 'DuRN_P_no_norm' for gaussion or real-world noise removal
data_name = 'RealNoiseHKPoly' # 'BSD_gray' or 'RealNoiseHKPoly'

# Gaussian noise level. Comment it if you set data_name = 'RealNoiseHKPoly'.
#noise_level = 70  # choose one from [30, 50, 70]
#----------------------

if data_name == 'BSD_gray':    
    testroot = "../data/"+data_name+"/test/"
    test_list_pth = '../lists/'+data_name+'/testlist.txt'
else:
    testroot = "../data/"+data_name+"/test1/"
    test_list_pth = '../lists/'+data_name+'/test1_list.txt'

Pretrained = '../trainedmodels/'+data_name+'/'+tag+'_model.pt' 
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Make the transformer and the network
if data_name == 'BSD_gray':
    transform = [transforms.ToTensor(), noise_level]    
    cleaner = cleaner().cuda()
else:
    transform = transforms.ToTensor()    
    cleaner = cleaner_no_norm().cuda()
    
cleaner.load_state_dict(torch.load(Pretrained))
cleaner.eval()

# Make the dataloader
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform)
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

ave_psnr = 0    
ave_ssim = 0    
ct_num = 0
for i, data in enumerate(dataloader):
    ct_num+= 1.0
    im_input, label, im_name = data    
    im_input = Variable(im_input, requires_grad=False).cuda()
    res = cleaner(im_input)
    res = res.data.cpu().numpy()
    res[res>1] = 1
    res[res<0] = 0
    res*= 255
    if data_name == 'BSD_gray':
        res = res.astype(np.uint8)[0,0]
        label = label.numpy()[0,0]
        label*= 255
        label = label.astype(np.uint8)        
        cv2.imwrite(show_dst+im_name[0].split('.')[0]+'_'+str(noise_level)+'.png', res)
        ave_psnr+= psnr(res, label, data_range=255)
        ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=False)
        
    elif data_name == 'RealNoiseHKPoly':
        res = res.astype(np.uint8)[0]
        res = res.transpose((1,2,0))
        label = label.numpy()[0].transpose((1,2,0))
        label*= 255
        label = label.astype(np.uint8)                
        Image.fromarray(res).save(show_dst+im_name[0].split('real')[0]+'.png')
        ave_psnr+= psnr(res, label, data_range=255)
        ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=True)
        
    else:
        print 'Unknown dataset name.'
        
print 'psnr: '+str(ave_psnr/ct_num)
print 'ssim: '+str(ave_ssim/ct_num)
print 'Test done.'
