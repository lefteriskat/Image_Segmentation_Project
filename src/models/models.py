import torch
import torch.nn as nn
import torch.nn.functional as F


class EncDecModel(nn.Module):
    '''
    Basic encoder decoder NN
    '''
    def __init__(self, in_channels, out_channels, conv_channels = 64):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(in_channels, conv_channels, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(16)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.upsample1 = nn.Upsample(32)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.upsample2 = nn.Upsample(64)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(conv_channels, conv_channels, 3, padding=1)
        self.upsample3 = nn.Upsample(128)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(conv_channels, out_channels, 3, padding=1)


    def forward(self, x):
         # encoder
        e0 = self.pool0(F.relu(self.enc_conv0(x)))
        e1 = self.pool1(F.relu(self.enc_conv1(e0)))
        e2 = self.pool2(F.relu(self.enc_conv2(e1)))
        e3 = self.pool3(F.relu(self.enc_conv3(e2)))

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        d0 = F.relu(self.dec_conv0(self.upsample0(b)))
        d1 = F.relu(self.dec_conv1(self.upsample1(d0)))
        d2 = F.relu(self.dec_conv2(self.upsample2(d1)))
        d3 = self.dec_conv3(self.upsample3(d2))  # no activation
        return d3
    
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )
        self.pool0 = nn.MaxPool2d(2, stride=2)

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.pool1 = nn.MaxPool2d(2, stride=2)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )

        # decoder (upsampling)
        self.upconv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()
        )

        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )

        self.upconv0 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()
        )

        self.final_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        e1 = self.enc_conv1(self.pool0(e0))
        e2 = self.enc_conv2(self.pool1(e1))
        e3 = self.enc_conv3(self.pool2(e2))

        # bottleneck
        b = self.bottleneck_conv(self.pool3(e3))

        # decoder
        d3 = self.dec_conv3(torch.cat([self.upconv3(b), e3], 1))
        d2 = self.dec_conv2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec_conv1(torch.cat([self.upconv1(d2), e1], 1))
        d0 = self.dec_conv0(torch.cat([self.upconv0(d1), e0], 1))

        return torch.sigmoid(self.final_conv(d0))



class UnetBlock(nn.Module):
    '''
    UNet block
    It can be used to sequrntially build a larger UNet from the bottom up.
    '''
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x
    
    def cnn_layer(self, in_channels, out_channels, kernel_size=3, bn=True):
        padding = kernel_size//2 # To preserve img dimensions. Equal to int((k-1)/2)
        cnn_bias = False if bn else True # Fewer parameters to save
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding,bias=cnn_bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.LeakyReLU()
        )
    
