import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import itertools
import pietorch.N_modules as n_mods
from PIL import Image
from torchvision import models
from torch.autograd import Variable
from .N_modules import InsNorm
from . import se_nets

class cleaner(nn.Module):
    def __init__(self):
        super(cleaner, self).__init__()

        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1)
        self.norm1 = FeatNorm('batch_norm', 64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = FeatNorm('batch_norm', 128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm3 = FeatNorm('batch_norm', 128)
        
        # DuRB_S x 3
        self.rud1 = DualSELayer(128, 128, 128, f_size=3, dilation=12, norm_type='batch_norm')
        self.rud2 = DualSELayer(128, 128, 128, f_size=3, dilation=8, norm_type='batch_norm')
        self.rud3 = DualSELayer(128, 128, 128, f_size=3, dilation=6, norm_type='batch_norm')

        # DuRB_P x 6
        self.rud4 = DualResLayer(128, 128, 128, f_size=3, dilation=2, norm_type='batch_norm')        
        self.rud5 = DualResLayer(128, 128, 128, f_size=5, dilation=1, norm_type='batch_norm')
        self.rud6 = DualResLayer(128, 128, 128, f_size=3, dilation=3, norm_type='batch_norm')
        self.rud7 = DualResLayer(128, 128, 128, f_size=7, dilation=1, norm_type='batch_norm')
        self.rud8 = DualResLayer(128, 128, 128, f_size=3, dilation=4, norm_type='batch_norm')
        self.rud9 = DualResLayer(128, 128, 128, f_size=7, dilation=1, norm_type='batch_norm')
        
        # Last layers
        self.deconv1 = nn.Sequential(
                        nn.ConvTranspose2d(128, 128, 4, 2, 1),
                        nn.ReflectionPad2d((1, 0, 1, 0)),
                        nn.AvgPool2d(2, stride = 1),
                        nn.ReLU()
                        )
        
        self.conv4 = ConvLayer(128, 128, kernel_size=3, stride=1)
        self.norm4 = FeatNorm('batch_norm', 128)

        self.deconv2 = nn.Sequential(
                        nn.ConvTranspose2d(128, 64, 4, 2, 1),
                        nn.ReflectionPad2d((1, 0, 1, 0)),
                        nn.AvgPool2d(2, stride = 1),
                        nn.ReLU()
                        )
        self.end_conv = ConvLayer(64, 3, kernel_size=5, stride=1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # PReLU is not used.
        self.prelu = nn.PReLU()        

        
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))        
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.relu(self.norm3(self.conv3(x)))
        res = x        
        
        x, res = self.rud1(x, res)
        x, res = self.rud2(x, res)
        x, res = self.rud3(x, res)
        res = x
        
        x, res = self.rud4(x, res)       
        x, res = self.rud5(x, res)
        x, res = self.rud6(x, res)
        x, res = self.rud7(x, res)
        x, res = self.rud8(x, res)
        x, res = self.rud9(x, res)                

        x = self.deconv1(x)
        x = self.relu(self.norm4(self.conv4(x)))

        x = self.deconv2(x)        
        x = self.tanh(self.end_conv(x))

        x = residual - x
        
        return x                                                            
        

class DualResLayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_size, dilation=1, norm_type="batch_norm", with_relu=True):
        super(DualResLayer, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm1 = FeatNorm(norm_type, in_dim)        
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm2 = FeatNorm(norm_type, in_dim)

        # T^{l}_{1} (conv. + bn)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=f_size, dilation=dilation, stride=1)
        self.up_norm = FeatNorm(norm_type, res_dim)

        # T^{l}_{2} (conv. + bn)        
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=5, stride=1)
        self.down_norm = FeatNorm(norm_type, out_dim)
        
        self.with_relu = with_relu        
        self.relu = nn.ReLU()
        
    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x+= x_r            
        x = self.relu(self.norm2(x))

        # T^{l}_{1} (conv. + bn)        
        x = self.up_norm(self.up_conv(x))
        x+= res
        x = self.relu(x)
        res = x

        # T^{l}_{2} (conv. + bn)        
        x = self.down_norm(self.down_conv(x))
        x+= x_r

        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res


class DualSELayer(nn.Module):
    def __init__(self, in_dim, out_dim, res_dim, f_size=3, f2_size=3, dilation=1, norm_type="instance", with_relu=True):
        super(DualSELayer, self).__init__()
        
        self.conv1 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm1 = FeatNorm(norm_type, in_dim)        
        self.conv2 = ConvLayer(in_dim, in_dim, 3, 1)
        self.norm2 = FeatNorm(norm_type, in_dim)

        # T^{l}_{1} (conv. + bn)
        self.up_conv = ConvLayer(in_dim, res_dim, kernel_size=f_size, stride=1, dilation=dilation) 
        self.up_norm = FeatNorm(norm_type, res_dim)

        # T^{l}_{2} (se + conv. + bn)
        self.se = se_nets.SEBasicBlock(res_dim, res_dim, reduction=64, with_norm=True)
        self.down_conv = ConvLayer(res_dim, out_dim, kernel_size=f2_size, stride=1) 
        self.down_norm = FeatNorm(norm_type, out_dim)

        self.with_relu = with_relu            
        self.relu = nn.ReLU()

    def forward(self, x, res):
        x_r = x
        
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.conv2(x)
        x+= x_r
        x = self.relu(self.norm2(x))

        # T^{l}_{1} (conv. + bn)        
        x = self.up_norm(self.up_conv(x))
        x+= res
        x = self.relu(x)
        res = x

        # T^{l}_{2} (se + conv. + bn)        
        x = self.se(x)
        x = self.down_norm(self.down_conv(x))
        x+= x_r
        
        if self.with_relu:
            x = self.relu(x)
        else:
            pass

        return x, res         

# ------------------------------------                
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

