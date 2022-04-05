import torch.nn as nn
import torch
import numpy as np
from torch.nn import functional as F
import pdb
import scipy.io as scio
import functions

class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, stride):
        super(ConvBlock, self).__init__()
        self.add_module('pad',nn.ReflectionPad2d(1)),  #镜像padding
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=0)),#padd)),
        self.add_module('LeakyRelu', nn.LeakyReLU(0.2, inplace=True))

class FUSION(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FUSION, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_ = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, ms, pan):
        image = torch.cat([ms, pan], 1)#堆叠
        image_ = self.conv(image)
        image_ = self.conv_(image_)#卷积
        mask = F.softmax(image_,dim=1)#缩小到0—1
        out_1 = torch.mul(mask,ms)#点乘
        out_2 = torch.mul((1 - mask), pan)
        return (out_1 + out_2)# out_

class FUSION_end(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(FUSION_end, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv_ = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1)

    def forward(self, ms, pan):
        image = torch.cat([ms, pan], 1)#堆叠
        image_ = self.conv(image)
        image_ = self.conv_(image_)#卷积
        mask = F.softmax(image_, dim=1)
        out_1 = torch.mul(mask,ms)#点乘
        out_2 = torch.mul((1 - mask), pan)
        return (out_1 + out_2) # out_

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self,opt):
        super(Net, self).__init__()

        self.ms1=ConvBlock(in_channel=4,out_channel=opt.nfc,ker_size=3,stride=1)
        self.pan1=ConvBlock(in_channel=1,out_channel=opt.nfc,ker_size=3,stride=1)
        self.ms1_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc,ker_size=3,stride=1)
        self.pan1_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc,ker_size=3,stride=1)
        self.fuBlock1=FUSION(in_channels=opt.nfc, out_channels=opt.nfc)
        self.con1=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc, out_channels=opt.nfc, kernel_size=3, stride=1, padding=0)
        )

        self.ms2 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.pan2 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.ms2_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc,ker_size=3,stride=1)
        self.pan2_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc,ker_size=3,stride=1)
        self.fuBlock2 = FUSION(in_channels=opt.nfc, out_channels=opt.nfc)
        self.con2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc*2, out_channels=opt.nfc, kernel_size=3, stride=1, padding=0)
        )
        self.ms3 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.pan3 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.ms3_1 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.pan3_1 = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.fuBlock3 = FUSION(in_channels=opt.nfc, out_channels=opt.nfc)
        self.con3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc * 2, out_channels=opt.nfc, kernel_size=3, stride=1, padding=0)
        )

        self.ms_end = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.pan_end = ConvBlock(in_channel=opt.nfc, out_channel=opt.nfc, ker_size=3, stride=1)
        self.ms_end_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc64,ker_size=3,stride=1)
        self.pan_end_1=ConvBlock(in_channel=opt.nfc,out_channel=opt.nfc64,ker_size=3,stride=1)
        self.fuBlock_end = FUSION_end(in_channels=opt.nfc64, out_channels=opt.nfc)
        self.last_con = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc*3, out_channels=opt.nfc64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc64, out_channels=opt.nfc, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc, out_channels=4, kernel_size=3, stride=1, padding=0)
        )

        self.ms_finally=nn.Conv2d(in_channels=opt.nfc*2,out_channels=1,kernel_size=3,padding=1,stride=1)
        self.pan_finally=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=opt.nfc*2,out_channels=4,kernel_size=3,stride=1,padding=0)
        )

    def forward(self, ms,pan):

        ms1=self.ms1(ms)
        pan1=self.pan1(pan)
        ms1_1=self.ms1_1(ms1)
        pan1_1=self.pan1_1(pan1)
        mask1,mask1_1,mask1_2,mask1_3=self.fuBlock1(ms1_1,pan1_1)
        last1=self.con1(mask1)

        ms2 = self.ms2(ms1_1)
        pan2 = self.pan2(pan1_1)
        ms2_1=self.ms2_1(ms2)
        pan2_1=self.pan2_1(pan2)
        mask2= self.fuBlock2(ms2_1, pan2_1)
        end2 = torch.cat([last1, mask2], 1)
        last2=self.con2(end2)

        ms3 = self.ms3(ms2_1)
        pan3 = self.pan3(pan2_1)
        ms3_1 = self.ms3_1(ms3)
        pan3_1 = self.pan3_1(pan3)
        mask3= self.fuBlock3(ms3_1, pan3_1)
        end3 = torch.cat([last2, mask3], 1)
        last3 = self.con3(end3)

        ms_end = self.ms_end(ms3_1)
        pan_end = self.pan_end(pan3_1)
        ms_end_1=self.ms_end_1(ms_end)
        pan_end_1=self.pan_end_1(pan_end)
        mask_end = self.fuBlock_end(ms_end_1, pan_end_1)

        last_end = torch.cat([last3,mask_end], 1)
        out3 = self.last_con(last_end)


        pan_=self.ms_finally(ms_end_1)
        ms_=self.pan_finally(pan_end_1)

        return ms_,pan_,out3
