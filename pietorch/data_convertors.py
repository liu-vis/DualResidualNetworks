import numpy as np
import torch
import random
import torchvision
import math
import torch.nn as nn
import itertools
import skimage as ski
import torch.utils.data as data
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
from scipy import ndimage
from scipy.special import gamma
from skimage.transform import warp
from .N_modules import CropSample, DataAugmentation
import cv2, h5py

class ConvertImageSet(data.Dataset):
    # Init. it.
    def __init__(self, dataroot,
                 imlist_pth,
                 data_name,
                 transform=None,
                 resize_to=None,
                 crop_size=None,
                 is_train =False,
                 with_aug =False):
        self.is_train  = is_train
        self.with_aug  = with_aug
        self.dataroot  = dataroot
        self.transform = transform
        self.data_name = data_name
        self.resize_to = resize_to
        self.crop_size = crop_size
        self.imlist    = self.flist_reader(imlist_pth)        


    # Process data.
    def __getitem__(self, index):
        im_name         = self.imlist[index]
        im_input, label = self.sample_loader(im_name)


        # Resize a sample, or not.
        if not self.resize_to is None:
            im_input = cv2.resize(im_input, self.resize_to)
            label    = cv2.resize(label,    self.resize_to)

        
        # Transform: output torch.tensors of [0,1] and (C,H,W).
        # Note: for test on DDN_Data and RESIDE, the output is in [0,1] and (V,C,H,W).
        #       V means the distortation types of a dataset (e.g., V == 14 for DDN_Data)
        if not self.transform is None:
            im_input, label = self.Transformer(im_input, label)
            
        return im_input, label, im_name

            
    # Read a image name list.
    def flist_reader(self, flist):
        imlist = []
        for l in open(flist).read().splitlines():
            imlist.append(l)
        return imlist

        
    # Return a pair of images (input, label).
    def sample_loader(self, im_name):
        if self.data_name   == 'RESIDE':
            return RESIDE_loader(self.dataroot, im_name, self.is_train)
            
        elif self.data_name == 'DCPDNData':            
            return DCPDNData_loader(self.dataroot, im_name)
            
        elif self.data_name == 'BSD_gray':
            return BSDgray_loader(self.dataroot, im_name)
            
        elif self.data_name == 'RealNoiseHKPoly':
            return RealNoiseHKPoly_loader(self.dataroot, im_name, self.is_train)

        elif self.data_name == 'GoPro':
            return GoPro_loader(self.dataroot, im_name)
            
        elif self.data_name == 'CarDataset':
            return Car_loader(self.dataroot, im_name)
            
        elif self.data_name == 'RainDrop':
            return RainDrop_loader(self.dataroot, im_name, self.is_train, color_fmt='BGR')
            
        elif self.data_name == 'DDN_Data':
            return DDNdata_loader(self.dataroot, im_name, self.is_train)

        elif self.data_name == 'DIDMDN_Data':
            return DIDMDNdata_loader(self.dataroot, im_name, self.is_train)
        else:
            print("Unknown dataset.")
            quit()

            
    def Transformer(self, im_input, label):
        if self.data_name == 'RESIDE':
            if not self.is_train:
                label = self.transform(label)
                im_input = im_input.transpose((3, 2, 0, 1))
                im_input = torch.FloatTensor(im_input)
                im_input/= 255.0
            else:
                if not self.crop_size is None:
                    im_input, label = CropSample(im_input, label, self.crop_size)
                if self.with_aug:
                    im_input, label = DataAugmentation(im_input, label)

                im_input = self.transform(im_input)
                label    = self.transform(label)
               
        elif self.data_name == 'DCPDNData':
            im_input = im_input.transpose((2, 0, 1))
            im_input = torch.FloatTensor(im_input)
            label = label.transpose((2, 0, 1))
            label = torch.FloatTensor(label)
            
        elif self.data_name in ['RainDrop',
                                'GoPro',
                                'CarDataset',
                                'RealNoiseHKPoly',
                                'DIDMDN_Data']:
            if not self.crop_size is None:
                im_input, label = CropSample(im_input, label, self.crop_size)
            if self.with_aug:
                im_input, label = DataAugmentation(im_input, label)

            im_input = self.transform(im_input)                
            label    = self.transform(label)

        elif self.data_name == 'BSD_gray':
            if not self.is_train:
                transf, noise_level = self.transform
                im_input = transf(im_input)
                im_input = AddGaussianNoise(im_input, noise_level)
                im_input = im_input[0,:,:].unsqueeze(0)
                
                label = transf(label)
                label = label[0,:,:].unsqueeze(0)
            else:
                label = self.transform(label)
                im_input = label.clone()
    
        elif self.data_name == 'DDN_Data':
            label = self.transform(label)
            im_input = im_input.transpose((3, 2, 0, 1))
            im_input = torch.FloatTensor(im_input)
            im_input/= 255.0
        else:
            pass
            
        return im_input, label
        
    def __len__(self):
        return len(self.imlist)


def RESIDE_loader(dataroot, im_name, is_train):
    if not is_train:
        Vars = np.arange(1, 11, 1)
        label_pth = dataroot+'labels/'+im_name
        label = Image.open(label_pth).convert("RGB")
        for var in Vars:
            if var == 1:
                hazy = np.asarray(Image.open(
                    dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'))
                hazy = np.expand_dims(hazy, axis=3)
            else:
                current = np.asarray(Image.open(
                    dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'))
                current = np.expand_dims(current, axis=3) 
                hazy = np.concatenate((hazy, current), axis=3)
    else:
        var = random.choice(np.arange(1, 11, 1))
        label_pth = dataroot+'labels/'+im_name
        hazy_pth = dataroot+'images/'+im_name.split('.')[0]+'_'+str(var)+'.png'

        label = Image.open(label_pth).convert("RGB")
        hazy  = Image.open(hazy_pth).convert("RGB")

    return hazy, label

        
def DCPDNData_loader(dataroot, im_name):
    sample_pth = dataroot+im_name
    f = h5py.File(sample_pth, 'r')
    keys = f.keys()
    
    # h5 to numpy, ato and trm are not used.
#    ato = np.asarray(f[keys[0]])
    label = np.asarray(f[keys[1]])
    hazy = np.asarray(f[keys[2]])
#    trm = np.asarray(f[keys[3]])

    if label.max() > 1 or hazy.max() > 1:
        print("DCPDNData out of range [0, 1].")
        quit()
    return hazy, label


def RainDrop_loader(dataroot, im_name, is_train, color_fmt='BGR'):
    if not is_train:
        if dataroot.split('/')[-2] == 'test_a':
            houzhui = 'png'
        else:
            assert(dataroot.split('/')[-2] == 'test_b')
            houzhui = 'jpg'
    
        label = cv2.imread(dataroot+'gt/'+im_name+'_clean.'+houzhui)
        label = align_to_k(label, k=4)    
        rainy = cv2.imread(dataroot+'data/'+im_name+'_rain.'+houzhui)
        rainy = align_to_k(rainy, k=4)
    
        if color_fmt == 'RGB':
            rainy = cv2.cvtColor(rainy, cv2.COLOR_BGR2RGB)    
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)    
    else:
        label = cv2.imread(dataroot+'gt/'  +im_name+'_clean.png')
        rainy = cv2.imread(dataroot+'data/'+im_name+'_rain.png')
        
    return rainy, label

def GoPro_loader(dataroot, im_name):
    name1, name2 = im_name.split('/')
    blur_pth  = dataroot+name1+'/blur/'+name2   
    label_pth = dataroot+name1+'/sharp/'+name2   

    blur = cv2.imread(blur_pth)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    label = cv2.imread(label_pth)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return blur, label


def Car_loader(dataroot, im_name):
    blur_pth  = dataroot+'/blurred/'+im_name
    label_pth = dataroot+'/sharp/'+im_name

    blur = cv2.imread(blur_pth)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

    label = cv2.imread(label_pth)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
    return blur, label

    
def BSDgray_loader(dataroot, im_name):    
    im_pth = dataroot+im_name
    label  = Image.open(im_pth).convert('RGB')
    noisy  = Image.open(im_pth).convert('RGB') # to which noise will be added (in test).
    return noisy, label

    
def RealNoiseHKPoly_loader(dataroot, im_name, is_train):
    if is_train:
        noisy_pth = dataroot+im_name.split('mean')[0]+'Real.JPG'
    else:
        noisy_pth = dataroot+im_name.split('mean')[0]+'real.PNG'
    label_pth = dataroot+im_name
    noisy = Image.open(noisy_pth).convert('RGB')
    label = Image.open(label_pth).convert('RGB')
    return noisy, label

    
def DIDMDNdata_loader(dataroot, im_name, is_train):
    if is_train:
        var      = random.choice(['Rain_Heavy', 'Rain_Medium', 'Rain_Light'])
        pair_pth = dataroot+var+'/train2018new/'+im_name
    else:
        pair_pth = dataroot+im_name

    pair = Image.open(pair_pth)
    pair_w, pair_h = pair.size
    rainy = pair.crop((0, 0, pair_w/2, pair_h))
    label = pair.crop((pair_w/2, 0, pair_w, pair_h))
    return rainy, label    

    
def DDNdata_loader(dataroot, im_name, is_train):    
    label_pth = dataroot+'label/'+im_name
    label = Image.open(label_pth).convert("RGB")
    
    if is_train:
        var = random.choice(np.arange(1, 15, 1))
        rainy_pth = dataroot+'rain_image/'+im_name.split('.')[0]+str(var)+'.jpg'
        rainy = Image.open(rainy_pth).convert("RGB")
    else:
        for var in np.arange(1, 15, 1):
            if var == 1:
                rainy = np.asarray(Image.open(
                    dataroot+'rain_image/'+im_name.split('.')[0]+'_'+str(var)+'.jpg'))            
                rainy = np.expand_dims(rainy, axis=3)
                
            else:
                current = np.asarray(Image.open(
                    dataroot+'rain_image/'+im_name.split('.')[0]+'_'+str(var)+'.jpg'))            
                current = np.expand_dims(current, axis=3) 
                rainy   = np.concatenate((rainy, current), axis=3)
    
    return rainy, label
   
    
def align_to_k(img, k=4):
    a_row = int(img.shape[0]/k)*k
    a_col = int(img.shape[1]/k)*k
    img = img[0:a_row, 0:a_col]
    return img


def AddGaussianNoise(patchs, var):
    # A randomly generated seed. Use it for an easy performance comparison.
    m_seed_cpu = 8526081014239199321
    m_seed_gpu = 8223752412272754
    torch.cuda.manual_seed(m_seed_gpu)
    torch.manual_seed(m_seed_cpu)
    
    c, h, w = patchs.size()
    noise_pad = torch.FloatTensor(c, h, w).normal_(0, var)
    noise_pad = torch.div(noise_pad, 255.0)
    patchs+= noise_pad    
    return patchs    
    
