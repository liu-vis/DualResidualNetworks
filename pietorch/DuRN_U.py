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
from .N_modules import InsNorm
        
class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
        self.norm1 = FeatNorm('instance', 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = FeatNorm('instance', 128)        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = FeatNorm('instance', 256)

        # DuRBs, a DualUpDownLayer is a DuRB_U.
        self.rud1 = DualUpDownLayer(256, 256, 128, f_size=3, dilation=3, norm_type='instance')
        self.rud2 = DualUpDownLayer(256, 256, 128, f_size=7, dilation=1, norm_type='instance')
        self.rud3 = DualUpDownLayer(256, 256, 128, f_size=3, dilation=3, norm_type='instance')
        self.rud4 = DualUpDownLayer(256, 256, 128, f_size=7, dilation=1, norm_type='instance')
        self.rud5 = DualUpDownLayer(256, 256, 128, f_size=3, dilation=2, norm_type='instance')
        self.rud6 = DualUpDownLayer(256, 256, 128, f_size=5, dilation=1, norm_type='instance')        

        #Last layers
        # -- Up1 --
        self.upconv1 = ConvLayer(256, 512, kernel_size=1, stride=1)
        self.upnorm1 = FeatNorm('instance', 512)
        self.upsamp1 = nn.PixelShuffle(2)
        # ---------
        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=1)
        self.norm4 = FeatNorm('instance', 128)

        # -- Up2 --
        self.upconv2 = ConvLayer(128, 256, kernel_size=1, stride=1)
        self.upnorm2 = FeatNorm('instance', 256) 
        self.upsamp2 = nn.PixelShuffle(2)        
        # ---------
        self.conv5 = ConvLayer(64, 64, kernel_size=3, stride=1)
        self.norm5 = FeatNorm('instance', 64)

        self.end_conv = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        res = x
        x = self.relu(self.norm3(self.conv3(x)))

        x, res = self.rud1(x, res)
        x, res = self.rud2(x, res)
        x, res = self.rud3(x, res)
        x, res = self.rud4(x, res)
        x, res = self.rud5(x, res)
        x, res = self.rud6(x, res)

        x = self.upnorm1(self.upconv1(x))
        x = self.upsamp1(x)
        x = self.relu(self.norm4(self.conv4(x)))

        x = self.upnorm2(self.upconv2(x))        
        x = self.upsamp2(x)        
        x = self.relu(self.norm5(self.conv5(x)))

        x = self.tanh(self.end_conv(x))
        x = x + residual
        
        return x


# DualUpDownLayer IS DuRB_U, defined here:
class DualUpDownLayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_size=3, dilation=1, norm_type="instance", with_relu=True):
        super(DualUpDownLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm1 = FeatNorm(norm_type, in_dim)       
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm2 = FeatNorm(norm_type, in_dim)

        # T^{l}_{1}: (Up+conv+insnorm)
        #-- Up --
        self.conv_pre = ConvLayer(in_dim, 2*in_dim, 1, 1)
        self.norm_pre = FeatNorm(norm_type, 2*in_dim)        
        self.upsamp = nn.PixelShuffle(2)
        #--------
        self.up_conv = ConvLayer(res_dim, res_dim, kernel_size=f_size, stride=1, dilation=dilation)
        self.up_norm = FeatNorm(norm_type, res_dim)

        # T^{l}_{2}: (conv+insnorm), stride=2 for down-scaling.        
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=3, stride=2)
        self.down_norm = FeatNorm(norm_type, out_dim)

        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(self.norm2(x))

        x = self.norm_pre(self.conv_pre(x))
        x = self.upsamp(x)
        x = self.up_conv(x)
        x+= res
        x = self.relu(self.up_norm(x))
        res = x

        x = self.down_conv(x)
        x+= x_r        
        x = self.down_norm(x)

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res        


#------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
        super(ConvLayer, self).__init__()
        self.dilation=dilation
        if dilation == 1:
            reflect_padding = int(np.floor(kernel_size/2))
            self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)
        else:
            self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation, padding=dilation)
            
    def forward(self, x):
        if self.dilation == 1:
            out = self.reflection_pad(x)
            out = self.conv2d(out)
        else:
            out = self.conv2d(x)            
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
