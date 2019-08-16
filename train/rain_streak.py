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

#------ Options -------
tag        = 'DuRN_S'
data_name  = 'DIDMDN_Data'
bch_size   = 40
base_lr    = 0.0001
epoch_size = 2000
gpus       = 1
crop_size  = 64

ssim_weight    = 1.1
l1_loss_weight = 0.75
with_data_aug  = True  # Image.FLIP_LEFT_RIGHT
#----------------------

# Set pathes
data_root  = '../data/' +data_name+'/train/'
imlist_pth = '../lists/'+data_name+'/trainlist.txt'

# dstroot for saving models. 
# logroot for writting some log(s), if is needed.
dstroot = './trainedmodels/'+data_name+'/'+tag+'/'
logroot = './logs/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', dstroot])
subprocess.check_output(['mkdir', '-p', logroot])

# Transform
transform = transforms.ToTensor()
# Dataloader
convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, data_name,
                                             transform=transform, is_train=True,
                                             with_aug=with_data_aug, crop_size=crop_size)
dataloader = DataLoader(convertor, batch_size=bch_size, shuffle=False, num_workers=5)

# Make network
cleaner = cleaner().cuda()
cleaner.train()

# Optimizer and Loss
optimizer = optim.Adam(cleaner.parameters(), lr=base_lr)
schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch_size)
L1_loss = nn.L1Loss()

# Start training
print('Start training...')
for epoch in range(epoch_size):        
    for iteration, data in enumerate(dataloader):
        img, label, _ = data
        img_var   = Variable(img,   requires_grad=False).cuda()
        label_var = Variable(label, requires_grad=False).cuda()

        # Cleaning noisy images
        cleaned = cleaner(img_var)

        # Compute ssim loss (not used)
        ssim_loss = -ssim(cleaned, label_var)
        ssim_loss = ssim_loss*ssim_weight

        # Compute L1 loss (not used)
        l1_loss   = L1_loss(cleaned, label_var)
        l1_loss   = l1_loss*l1_loss_weight

        loss = ssim_loss + l1_loss
        # Backward and update params        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check 
#        torchvision.utils.save_image(img[:16],     logroot+'input_images.png')
#        torchvision.utils.save_image(label[:16],   logroot+'label_images.png')
#        torchvision.utils.save_image(cleaned[:16], logroot+'temp_res.png'    )
        print('Epoch('+str(epoch+1)+'), iteration('+str(iteration+1)+'): '+str(loss.item()))

    if epoch%10 == 9:
        if gpus == 1:            
            torch.save(cleaner.state_dict(),        dstroot+'epoch_'+str(epoch+1)+'_model.pt')
        else:
            torch.save(cleaner.module.state_dict(), dstroot+'epoch_'+str(epoch+1)+'_model.pt') 
