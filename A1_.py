import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F
from torch.nn import init
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, datasets, models
from mytransforms import *
from mytransforms import mytransforms
from skimage.filters import threshold_otsu
from skimage import feature
from skimage.color import rgb2gray
from numpy import matlib
import cv2
import os,sys
import numpy as np
from numpy import *
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import random

class MD(Dataset):
    def __init__(self,  path='train', H=600,W=450,pow_n=3, aug=True):

        init_trans = transforms.Compose([transforms.Resize((H, W)),
                                         #transforms.Grayscale(1),
                                         transforms.ToTensor(),
                                         ])
        init_trans2 = transforms.Compose([transforms.Resize((H, W)),
                                          transforms.Grayscale(1),
                                          transforms.ToTensor(),
                                          ])
        self.datainfo = torchvision.datasets.ImageFolder(root=path, transform=init_trans)
        self.datainfo2 = torchvision.datasets.ImageFolder(root=path, transform=init_trans2)

        self.mask_num=len(self.datainfo.classes)-1
        self.data_num = int(len(self.datainfo)/len(self.datainfo.classes))
        self.aug=aug
        self.pow_n = pow_n
        self.task = path
        self.H = H
        self.W = W

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):

        if self.aug == True: self.rv = random.random()
        else: self.rv=-1
        if self.rv>=.1:
            # augmenation of img and masks
            angle = random.randrange(-25, 25)
            trans_rand = [random.uniform(0, 0.05) , random.uniform(0, 0.05)]
            scale_rand = random.uniform(0.9, 1.1)
            # trans img with masks
            self.input_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                     mytransforms.Affine(angle,
                                                                         translate=trans_rand,
                                                                         scale=scale_rand,
                                                                         fillcolor=0),
                                                     mytransforms.ToTensor(),
                                                     ])
            self.mask_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                    mytransforms.Affine(angle,
                                                                        translate=trans_rand,
                                                                        scale=scale_rand,
                                                                        fillcolor=0),
                                                    mytransforms.ToTensor(),
                                                    ])

            self.col_trans = mytransforms.Compose([mytransforms.ToPILImage(),
                                                   mytransforms.ColorJitter(brightness=random.random(),
                                                                            contrast=random.random(),
                                                                            saturation=random.random(),
                                                                            hue=random.random() / 2
                                                                            ),
                                                   mytransforms.ToTensor(),
                                                   ])

            #print("angle:", angle, "vfilp:", vfilp)
            image, _ = self.datainfo.__getitem__(idx)

            #plt.imshow(image[0], cmap='gray');plt.show()
            image = self.col_trans(image)
            image = self.input_trans(image)

            #plt.imshow(image[0], cmap= 'gray' ); plt.show()
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)

            for k in range(0, self.mask_num):
                X, _ = self.datainfo2.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = self.mask_trans(X)
####################################################
        else:
            image, _ = self.datainfo.__getitem__(idx)
            mask = torch.empty(self.mask_num, image.shape[1], image.shape[2], dtype=torch.float)
            for k in range(0, self.mask_num):
                X, _ = self.datainfo2.__getitem__(idx + (self.data_num * (1 + k)))
                mask[k] = X

        mask = torch.pow(mask, self.pow_n)
        mask = mask / mask.max()
        #print("idx :", idx, "path ", self.task)

        #plt.imshow(image[0], cmap='gray');   plt.show()
        return [image, mask ]

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x
class single_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, dila=1):
        super(single_conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, dilation=dila, stride=1, padding=dila, bias=True),
            nn.LeakyReLU(inplace=True),
        )
    def forward(self, x):
        x = self.conv(x)
        return x
class SELayer(nn.Module):
    def __init__(self, channel=512, r_channel=256, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce=single_conv_block(ch_in=channel,ch_out=r_channel )
        self.fc = nn.Sequential(
            nn.Linear(r_channel, r_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(r_channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.conv_reduce(x)
        b, c, _, _ = x.size()
        print(b, c)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)

        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)

        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)

        x4 = self.Conv4(x4)
        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1

