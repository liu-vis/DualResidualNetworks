import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
from PIL import Image
from torchvision import models
from torch.autograd import Variable
from .se_nets import SEBasicBlock

# This is the DuRN for Gaussian noise removal.
class cleaner(nn.Module):
    def __init__(self, test_with_multigpus=False):
        super(cleaner, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = ConvLayer(3, 64, kernel_size=3, stride=1)
        self.norm1 = FeatNorm("batch_norm", 64)
        self.conv2 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm2 = FeatNorm("batch_norm", 64)
        self.mult_gpu_test = test_with_multigpus

        # DuRBs
        self.block1 = DuRB_s(k1_size=5, k2_size=3, dilation=1)
        self.block2 = DuRB_s(k1_size=7, k2_size=5, dilation=1)
        self.block3 = DuRB_s(k1_size=7, k2_size=5, dilation=2)
        self.block4 = DuRB_s(k1_size=11, k2_size=7, dilation=2)
        self.block5 = DuRB_s(k1_size=11, k2_size=5, dilation=1)
        self.block6 = DuRB_s(k1_size=11, k2_size=7, dilation=3)

        self.conv3 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm3 = FeatNorm("batch_norm", 64)        
        self.conv4 = ConvLayer(64, 3, kernel_size=3, stride=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):        
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))  
        res = out

        if self.mult_gpu_test:
            out = out.cuda(1)
            res = res.cuda(1)            
        out, res = self.block1(out, res)
        out, res = self.block2(out, res)

        if self.mult_gpu_test:        
            out = out.cuda(2)
            res = res.cuda(2)        
        out, res = self.block3(out, res)
        out, res = self.block4(out, res)

        if self.mult_gpu_test:        
            out = out.cuda(3)
            res = res.cuda(3)
        out, res = self.block5(out, res)
        out, res = self.block6(out, res)

        out = self.relu(self.norm3(self.conv3(out)))                
        out = self.tanh(self.conv4(out))
        if self.mult_gpu_test:
            out = out.cuda(0)
            
        out = out + x

        return out        

        
        
class DuRB_s(nn.Module):     
    def __init__(self, in_dim=64, out_dim=64, res_dim=64, k1_size=3, k2_size=1, dilation=1, norm_type='batch_norm', with_relu=True):
        super(DuRB_s, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm1 = FeatNorm(norm_type, in_dim)        
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm2 = FeatNorm(norm_type, in_dim)

        # T^{l}_{1}: (conv. + bn)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=k1_size, stride=1, dilation=dilation)
        self.up_norm = FeatNorm(norm_type, res_dim)

        # T^{l}_{2}: (se + conv. + bn)
        self.se = SEBasicBlock(res_dim, res_dim, reduction=int(res_dim/2), with_norm=True)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=k2_size, stride=1)
        self.down_norm = FeatNorm(norm_type, out_dim)
        
        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(self.norm2(x))

        # T^{l}_{1}
        x = self.up_norm(self.up_conv(x))
        x+= res
        x = self.relu(x)
        res = x

        # T^{l}_{2}
        x = self.se(x)
        x = self.down_norm(self.down_conv(x))
        x+= x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass
            
        return x, res

        
#---------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        reflect_padding = int(dilation * (kernel_size - 1) / 2)
        self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)        
        return out
        
        
class FeatNorm(nn.Module):
    def __init__(self, norm_type, dim):
        super(FeatNorm, self).__init__()
        if norm_type == "instance":
            self.norm = InsNorm(dim)
        elif norm_type == "batch_norm":
            self.norm = nn.BatchNorm2d(dim)
        else:
            raise Exception("Normalization type incorrect.")

    def forward(self, x):
        out = self.norm(x)        
        return out






        
