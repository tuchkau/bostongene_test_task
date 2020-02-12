import torch
import torch.nn as nn


def d_conv(in_c, out_c):
    return torch.nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(inplace=True))


class UNet(nn.Module):
    
    def __init__(self, n_class):
        super().__init__()
        self.conv_d1 = d_conv(3, 64)
        self.conv_d2 = d_conv(64, 128)
        self.conv_d3 = d_conv(128, 256)
        self.conv_d4 = d_conv(256, 512)
        #self.conv_d5 = d_conv(512, 1024)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
        
        self.conv_u1 = d_conv(512 + 256, 256)
        self.conv_u2 = d_conv(256 + 128, 128)
        self.conv_u3 = d_conv(128 + 64, 64)
        self.conv_out = nn.Conv2d(64, n_class, 1)
    
    def forward(self, x):
        conv1 = self.conv_d1(x)
        x = self.pool(conv1)
        
        conv2 = self.conv_d2(x)
        x = self.pool(conv2)
        
        conv3 = self.conv_d3(x)
        x = self.pool(conv3)
        
        x = self.conv_d4(x)
        #print(x.shape)
        x = self.up(x)
        #print(x.shape)
        x = torch.cat([x, conv3], dim=1)
        #print(x.shape)
        x = self.conv_u1(x)
        
        x = self.up(x)
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_u2(x)
        
        x = self.up(x)
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_u3(x)
        
        out = self.conv_out(x)
        
        return out