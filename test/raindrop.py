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
from pietorch.DuRN_S_P import cleaner as cleaner_sp
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

#------- Options -------
tag = 'DuRN_S_P'
data_name = 'RainDrop'
test_set = 'test_a'
# ----------------------


testroot = "../../data/"+data_name+"/"+test_set+"/"+test_set+"/"
test_list_pth = '../lists/'+data_name+'/'+test_set+'_list.txt'

Pretrained = '../trainedmodels/'+data_name+'/'+tag+'_model.pt'
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'+test_set+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform) 
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

# Make the network
cleaner = cleaner_sp().cuda()
cleaner.load_state_dict(torch.load(Pretrained))
cleaner.eval()

ave_psnr = 0    
ave_ssim = 0    
ct_num = 0
for i, data in enumerate(dataloader):    
    ct_num+= 1.0    
    im_input, label, im_name = data    
    im_input = Variable(im_input, requires_grad=False).cuda()
    res = cleaner(im_input)
    res = res.data.cpu().numpy()[0]
    res[res>1] = 1
    res[res<0] = 0
    res*= 255
    res = res.astype(np.uint8)
    res = res.transpose((1,2,0))
    cv2.imwrite(show_dst+im_name[0]+'.png', res)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2YCR_CB)[:,:,0]

    label = label.numpy()[0]
    label*= 255
    label = label.astype(np.uint8)
    label = label.transpose((1,2,0))
    label = cv2.cvtColor(label, cv2.COLOR_BGR2YCR_CB)[:,:,0]
    ave_psnr+= psnr(res, label, data_range=255)
    ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=False)
    
print 'psnr: '+str(ave_psnr/ct_num)
print 'ssim: '+str(ave_ssim/ct_num)
print 'Test done.'
