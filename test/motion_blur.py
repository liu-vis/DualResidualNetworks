# python 2.7, pytorch 0.3.1

import os, sys
import cv2
sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from pietorch import data_convertors
from pietorch.DuRN_U import cleaner
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

#------- Options -------
tag = 'DuRN_U'

# Choose a dataset.
data_name = 'GoPro' # 'GoPro' or 'CarDataset'

# To resize
resize_to = (640, 360) # For 'GoPro'
#resize_to = (360, 360) # For 'CarDataset'
#-----------------------

if data_name == 'GoPro':
    testroot = "../data/"+data_name+"/test/"
    test_list_pth = "../lists/"+data_name+'/test_list.txt'
elif data_name == 'CarDataset':
    testroot = "../data/"+data_name+"/"
    test_list_pth = "../lists/"+data_name+"/test_list.txt"
else:
    print 'Unknown dataset name.'

Pretrained = '../trainedmodels/GoPro/'+tag+'_model.pt'    
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform,
                                            resize_to=resize_to)
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

# Make the network
cleaner = cleaner().cuda()
cleaner.load_state_dict(torch.load(Pretrained))
cleaner.eval()
ave_psnr = 0.0
ave_ssim = 0.0
ct_num = 0

print 'Start testing '+tag+'...'
for i, data in enumerate(dataloader):
    blur, label, im_name = data
    ct_num+= 1
    label = label.numpy()[0]
    label*= 255
    label = label.astype(np.uint8)
    label = label.transpose((1,2,0))

    blur = Variable(blur, requires_grad=False).cuda()    
    res = cleaner(blur)
    res = res.data.cpu().numpy()[0]
    res[res>1] = 1
    res[res<0] = 0
    res*= 255
    res = res.astype(np.uint8)
    res = res.transpose((1,2,0))
    ave_psnr+= psnr(res, label, data_range=255)
    ave_ssim+= ski_ssim(res, label, data_range=255, multichannel=True)

    if data_name == 'CarDataset':
        res = cv2.resize(res, (700, 700))
        Image.fromarray(res).save(show_dst+im_name[0])
    else:
        Image.fromarray(res).save(show_dst+'_'.join(im_name[0].split('/')))

print 'psnr: '+str(ave_psnr/float(ct_num))+'.'
print 'ssim: '+str(ave_ssim/float(ct_num))+'.'
print 'Test done.'


