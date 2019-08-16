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

#------ Options -------
tag       = 'DuRN_P'
data_name = 'BSD_gray' 
bch_size  = 100
base_lr   = 0.001
gpus = 1

epoch_size = 3000
crop_size  = 64
Vars = [30, 50, 70]

l2_loss_weight = 1
locally_training_num = 10
#----------------------
def AddNoiseToTensor(patchs, Vars): # Pixels must be in [0,1]
    bch, c, h, w = patchs.size()
    for b in range(bch):
        Var = random.choice(Vars)
        noise_pad = torch.FloatTensor(c, h, w).normal_(0, Var)
        noise_pad = torch.div(noise_pad, 255.0)
        patchs[b]+= noise_pad    
    return patchs

# Set pathes
data_root  = '../data/'+data_name+'/train_and_val/'
imlist_pth = '../lists/'+data_name+'/train_list.txt'

# dstroot for saving models. 
# logroot for writting some log(s), if is needed.
dstroot = './trainedmodels/'+data_name+'/'+tag+'/'
logroot = './logs/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', dstroot])
subprocess.check_output(['mkdir', '-p', logroot])

# Transform
transform  = transforms.Compose([transforms.RandomCrop((crop_size, crop_size)), 
                                transforms.ToTensor()])
# Dataloader
convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, data_name,
                                            transform=transform, is_train=True)
dataloader = DataLoader(convertor, batch_size=bch_size, shuffle=False, num_workers=2)

# Make network
cleaner = cleaner().cuda()
cleaner.train()

# Optimizer and Loss
optimizer = optim.Adam(cleaner.parameters(), lr=base_lr)
L2_loss = nn.MSELoss()

# Start training
print('Start training...')
for epoch in range(epoch_size):        
    for iteration, data in enumerate(dataloader):
        img, label, _ = data # "img" which is clean, will be added noise.
        label_var = Variable(label[:,0,:,:], requires_grad=False).cuda()
        label_var = label_var.unsqueeze(1)

        for loc_tr in range(locally_training_num):
            noisy_patchs = AddNoiseToTensor(img.clone(), Vars)
            noisy_patchs = Variable(noisy_patchs, requires_grad=False).cuda()
            noisy_patchs = noisy_patchs[:,0,:,:].unsqueeze(1)

            # Cleaning noisy images
            cleaned = cleaner(noisy_patchs)

            # Compute L2 loss
            l2_loss = L2_loss(cleaned, label_var)
            loss    = l2_loss*l2_loss_weight

            # Backward and update params        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Check 
#        torchvision.utils.save_image(noisy_patchs[:16], logroot+'input_images.png')
#        torchvision.utils.save_image(label[:16],        logroot+'label_images.png')
#        torchvision.utils.save_image(cleaned[:16],      logroot+'temp_res.png'    )
        print('Epoch('+str(epoch+1)+'), iteration('+str(iteration+1)+'): '+str(loss.item()))

    if epoch%10 == 9:
        if gpus == 1:            
            torch.save(cleaner.state_dict(),        dstroot+'epoch_'+str(epoch+1)+'_model.pt')
        else:
            torch.save(cleaner.module.state_dict(), dstroot+'epoch_'+str(epoch+1)+'_model.pt') 

    if epoch in [500, 1000, 1500, 2500]:
        for param_group in optimizer.param_groups:
            param_group['lr']*= 0.1    

