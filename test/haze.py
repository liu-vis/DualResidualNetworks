# python 2.7, pytorch 0.3.1

import os, sys
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
from pietorch.DuRN_US import cleaner
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim

#------- Option --------
tag = 'DuRN-US'
# Choose a dataset.
data_name = 'RESIDE' # 'DCPDNData' or 'RESIDE'
#-----------------------

if data_name == 'RESIDE':
    testroot = "../data/"+data_name+"/sots_indoor_test/"
    test_list_pth = "../lists/RESIDE_indoor/sots_test_list.txt"
elif data_name == 'DCPDNData':    
    testroot = "../data/"+data_name+"/TestA/"
    test_list_pth = '../lists/'+data_name+'/testA_list.txt'
else:
    print('Unknown dataset name.')

Pretrained = '../trainedmodels/'+data_name+'/'+tag+'_model.pt'    
show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform)
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=1)

# Make the network
cleaner = cleaner().cuda()
cleaner.load_state_dict(torch.load(Pretrained))
cleaner.eval()

ave_psnr = 0.0
ave_ssim = 0.0
ct_num = 0
print('Start testing '+tag+'...')
for i, data in enumerate(dataloader):
    hazy, label, im_name = data
    if data_name == 'RESIDE':
        b,v,c,h,w = hazy.size()
        
        for bi in range(b):
            label_v = label[bi]
            label_v = label_v.numpy()
            label_v*= 255
            label_v = label_v.astype(np.uint8)
            label_v = label_v.transpose((1,2,0))
            
            for vi in range(v):
                ct_num+= 1
                hazy_vi = hazy[bi, vi]
                hazy_vi = hazy_vi.unsqueeze(dim=0)
                hazy_vi = Variable(hazy_vi, requires_grad=False).cuda()
                res = cleaner(hazy_vi)
                res = res.data.cpu().numpy()[0]
                res[res>1] = 1
                res[res<0] = 0
                res*= 255
                res = res.astype(np.uint8)
                res = res.transpose((1,2,0))
                ave_psnr+= psnr(res, label_v, data_range=255)
                ave_ssim+= ski_ssim(res, label_v, data_range=255, multichannel=True)
                Image.fromarray(res).save(show_dst+im_name[0].split('.')[0]+'_'+str(vi+1)+'.png')

    elif data_name == 'DCPDNData':
        ct_num+= 1
        label = label.numpy()[0]
        label = label.transpose((1,2,0))        
        hazy = Variable(hazy, requires_grad=False).cuda()
        res = cleaner(hazy)
        res = res.data.cpu().numpy()[0]
        res[res>1] = 1
        res[res<0] = 0
        res = res.transpose((1,2,0))
        ave_psnr+= psnr(res, label, data_range=1)
        ave_ssim+= ski_ssim(res, label, data_range=1, multichannel=True)
        
        res*= 255
        res = res.astype(np.uint8)        
        Image.fromarray(res).save(show_dst+im_name[0].split('.')[0]+'.png')

    else:
        print("Unknown dataset name.")

print('psnr: '+str(ave_psnr/float(ct_num))+'.')
print('ssim: '+str(ave_ssim/float(ct_num))+'.')
print('Test done.')


