import torch
from inplace_abn.abn import ABN, InPlaceABN, InPlaceABNSync
import torch.distributed as dist

## need to use iABNsync layer with leakyRelu

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


# LSFE module
class LSFE(nn.Module):
    def __init__(self, ):
        super(LSFE, self).__init__()
        self.conv1 = SeparableConv2d(256, 128, 3)
        self.bn1 = ABN(128)
        self.conv2 = SeparableConv2d(128, 128, 3)
        self.bn2 = ABN(128)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return x

# Mismatch Correction Module (MC)
class CorrectionModule(nn.Module):
    def __init__(self):
        super(CorrectionModule, self).__init__()
        self.conv1 = SeparableConv2d(128, 128, 3)
        self.bn1 = ABN(128)
        self.conv2 = SeparableConv2d(128, 128, 3)
        self.bn2 = ABN(128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        ## upsampling 

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.up(x)
        return x

# Dense Prediction Cells (DPC)
class DPC(nn.Module):
    def __init__(self, height, width, channels = 256):
        super(DPC, self).__init__()

        self.height = height
        self.width = width

        self.bn1 = ABN(256)
        self.conv1 = SeparableConv2d(256, 256, 3, dilation=(1, 6))
        self.up1 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.bn2 = ABN(256)
        self.conv2 = SeparableConv2d(256, 256, 3, dilation=(1, 1))
        self.up2 = nn.Upsample((self.height, self.width), mode='bilinear')


        self.bn3 = ABN(256)
        self.conv3 = SeparableConv2d(256, 256, 3, dilation=(6, 21))
        self.up3 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.bn4 = ABN(256)
        self.up_tocalculate18x3 = nn.Upsample((36, 64), mode='bilinear')
        self.conv4 = SeparableConv2d(256, 256, 3, dilation=(18, 15))
        self.up4 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.bn5 = ABN(256)
        self.conv5 = SeparableConv2d(256, 256, 3, dilation=(6,3))
        self.up5 = nn.Upsample((self.height, self.width), mode='bilinear')

        self.lastconv = nn.Conv2d(1280, 128, 1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x1 = self.up1(x)

        x2 = self.bn2(x1)
        x2 = self.conv2(x2)
        x2 = self.up2(x2)  

        x3 = self.bn3(x1)
        x3 = self.conv3(x3)
        x3 = self.up3(x3)

        x4 = self.bn4(x1)
        if( self.height < 33 ):
          x4 = self.up_tocalculate18x3(x4)
        x4 = self.conv4(x4)
        x4 = self.up4(x4)    

        x5 = self.bn5(x4)
        x5 = self.conv5(x5)
        x5 = self.up5(x5)

        cat = torch.cat(( x1, x2, x3, x4, x5), dim = 1)

        cat = self.lastconv(cat)

        return cat

class SemanticHead(nn.Module):
    def __init__(self):
        super(SemanticHead, self).__init__()
        self.dpcp32 = DPC(32, 64)
        self.dpcp16 = DPC(64, 128)
        self.lsfep8 = LSFE()
        self.lsfep4 = LSFE()

        self.up_p32 = nn.Upsample((64, 128), mode='bilinear')

        self.mc1 = CorrectionModule()
        self.mc2 = CorrectionModule()

        self.up1 = nn.Upsample((256, 512), mode = 'bilinear')
        self.up2 = nn.Upsample((256, 512), mode = 'bilinear')
        self.up3 = nn.Upsample((256, 512), mode = 'bilinear')
        
        self.lastconv = nn.Conv2d(512, 20, 1) ####### NEED TO CHANGE OUTPUT CHANNELS
        self.uplast = nn.Upsample((1024, 2048), mode = 'bilinear')
    
    
    def forward(self, p32, p16, p8, p4):

        d32 = self.dpcp32(p32)
        d16 = self.dpcp16(p16)

        lp8 = self.lsfep8(p8)
        lp4 = self.lsfep4(p4)

        up32 = self.up_p32(d32)
        
        add1 = torch.add(up32, d16)
        
        up16 = self.mc1(add1)
        
        add2 = torch.add(up16, lp8)
        up8 = self.mc2(add2)
        add3 = torch.add(up8, lp4) 
        
        cat1 = self.up1(d32)
        cat2 = self.up2(d16) 
        cat3 = self.up3(add2) 

        cat = torch.cat(( cat1, cat2, cat3, add3), dim = 1)

        cat = self.lastconv(cat)

        cat = self.uplast(cat)
        
        return cat