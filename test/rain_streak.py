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
from pietorch.DuRN_S import cleaner as cleaner
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

# DDN_Data contains a few huge images. You need "module parallel" to run the code on it.
#------- Option -------
tag = 'DuRN_S'
data_name = 'DDN_Data'  # DIDMDN_Data or DDN_Data
#-----------------------

if data_name == 'DIDMDN_Data':
    testroot = "../data/"+data_name+"/test/"
    test_list_pth = '../lists/'+data_name+'/testlist.txt'
else:
    testroot = "../data/"+data_name+"/"
    test_list_pth = '../lists/'+data_name+'/testlist.txt'    

    
Pretrained = '../trainedmodels/'+data_name+'/'+tag+'_model.pt'    
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform) 
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

# Make the network
if data_name == 'DDN_Data':    
    cleaner = cleaner(test_with_multigpus=True).cuda()
else:
    cleaner = cleaner().cuda()
cleaner.load_state_dict(torch.load(Pretrained))

# You need multple GPUs to run the DDN_Data, it has a few !Huge! images in the dataset.
if data_name == 'DDN_Data':
    cleaner.conv1 = cleaner.conv1.cuda(0)
    cleaner.norm1 = cleaner.norm1.cuda(0)
    cleaner.conv2 = cleaner.conv2.cuda(0)
    cleaner.norm2 = cleaner.norm2.cuda(0)
    
    cleaner.block1 = cleaner.block1.cuda(1)
    cleaner.block2 = cleaner.block2.cuda(1)
    
    cleaner.block3 = cleaner.block3.cuda(2)
    cleaner.block4 = cleaner.block4.cuda(2)
    
    cleaner.block5 = cleaner.block5.cuda(3)
    cleaner.block6 = cleaner.block6.cuda(3)
    
    cleaner.conv3 = cleaner.conv3.cuda(3)
    cleaner.norm3 = cleaner.norm3.cuda(3)
    cleaner.conv4 = cleaner.conv4.cuda(3)
else:
    pass
    
cleaner.eval()

ave_psnr = 0    
ave_ssim = 0    
ct_num = 0
for i, data in enumerate(dataloader):
    im_input, label, im_name = data    
    if data_name == 'DIDMDN_Data':    
        ct_num+= 1.0    
        im_input = Variable(im_input, requires_grad=False).cuda()
        res = cleaner(im_input)
        res = res.data.cpu().numpy()[0]
        res[res>1] = 1
        res[res<0] = 0
        res*= 255
        res = res.astype(np.uint8)
        res = res.transpose((1,2,0))
        Image.fromarray(res).save(show_dst+im_name[0]+'.png')
        res = cv2.cvtColor(res, cv2.COLOR_RGB2YCR_CB)[:,:,0]
    
        label = label.numpy()[0]
        label*= 255
        label = label.astype(np.uint8)
        label = label.transpose((1,2,0))
        label = cv2.cvtColor(label, cv2.COLOR_RGB2YCR_CB)[:,:,0]
        ave_psnr+= psnr(res, label, data_range=255)
        ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=False)
        
    elif data_name == 'DDN_Data':
        b,v,c,h,w = im_input.size()

        for bi in range(b):
            label_v = label[bi]
            label_v = label_v.numpy()
            label_v*= 255
            label_v = label_v.astype(np.uint8)
            label_v = label_v.transpose((1,2,0))
            label_v = cv2.cvtColor(label_v, cv2.COLOR_RGB2YCR_CB)[:,:,0]
            for vi in range(v):
                ct_num+= 1.0
                im_input_vi = im_input[bi, vi]
                im_input_vi = im_input_vi.unsqueeze(dim=0)
                im_input_vi = Variable(im_input_vi, requires_grad=False).cuda()
                res = cleaner(im_input_vi)
                res = res.data.cpu().numpy()[0]
                res[res>1] = 1
                res[res<0] = 0
                res*= 255
                res = res.astype(np.uint8)
                res = res.transpose((1,2,0))
                Image.fromarray(res).save(show_dst+im_name[0].split('.')[0]+'_'+str(vi+1)+'.png')
                res = cv2.cvtColor(res, cv2.COLOR_RGB2YCR_CB)[:,:,0]
                
                ave_psnr+= psnr(res, label_v, data_range=255)
                ave_ssim+= ski_ssim(res, label_v, data_range=255, multichannel=False)    

    else:
        print 'Unknown dataset name.'

        
print 'psnr: '+str(ave_psnr/ct_num)
print 'ssim: '+str(ave_ssim/ct_num)
print 'Test done.'
