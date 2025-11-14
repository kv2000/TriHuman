"""
@File: Unet_rollout_large.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The self-implemented unet with rollout convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from icecream import ic

# borrow from POP, :D
class Conv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, use_bias=False, use_bn=True, use_relu=True):
        super(Conv2DBlock, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(
            input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias
        )
        #nn.init.normal_(
        #    self.conv, 0., 0.02
        #)
        self.conv.weight.data.normal_(0, 0.02)
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        
        self.relu = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x

# the covolution from rodin
class Conv2DBlock_Rodin(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1, use_bias=False, use_bn=True, use_relu=True):
        super(Conv2DBlock_Rodin, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu
        self.conv = nn.Conv2d(
            input_nc * 3, output_nc, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias
        )

        self.conv.weight.data.normal_(0, 0.02)
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        
        self.relu = nn.LeakyReLU(0.01, inplace=False)

    def forward(self, x):
        if self.use_relu:
            x = self.relu(x)

        ww = x.shape[-1] // 3

        uv = x[:,:,:,:ww]
        uw = x[:,:,:,ww:(2 * ww)]
        wv = x[:,:,:,(-ww):]
        
        # average pooling, u_v -> u_1
        uv_u = torch.mean(uv, dim = -1, keepdim=True).expand([-1, -1, -1, ww])
        uv_v = torch.mean(uv, dim = -2, keepdim=True).expand([-1, -1, ww, -1])

        uw_u = torch.mean(uw, dim = -1, keepdim=True).expand([-1, -1, -1, ww])
        uw_w = torch.mean(uw, dim = -2, keepdim=True).expand([-1, -1, ww, -1])
        
        wv_w = torch.mean(wv, dim = -1, keepdim=True).expand([-1, -1, -1, ww])  
        wv_v = torch.mean(wv, dim = -2, keepdim=True).expand([-1, -1, ww, -1])

        f_uv_uv = torch.cat([uw_u, wv_v], dim = 1)
        f_uw_uw = torch.cat([uv_u, wv_w], dim = 1)
        f_wv_wv = torch.cat([uw_w, uv_v], dim = 1)

        f_layers = torch.cat([
            f_uv_uv, f_uw_uw, f_wv_wv
        ], dim = -1)

        x_cat = torch.cat(
            [x, f_layers], dim = 1
        )
        
        x_cat_down = self.conv(x_cat)
        
        if self.use_bn:
            x_cat_down = self.bn(x_cat_down)

        return x_cat_down

# also borrow from POP, :D, #Conv2DBlock_Rodin
class UpConv2DBlock_Rodin(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=False, use_bn=True, up_mode='upconv', use_dropout=False, use_relu = True):
        super(UpConv2DBlock_Rodin, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        self.use_relu = use_relu

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(input_nc * 3, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
            self.up.weight.data.normal_(0, 0.02)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(input_nc * 3, output_nc, kernel_size=3, padding=1, stride=1),
            )
        
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        
        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):

        if self.use_relu:
            x = self.relu(x)
        
        ww = x.shape[-1] // 3

        uv = x[:,:,:,:ww]
        uw = x[:,:,:,ww:(2 * ww)]
        wv = x[:,:,:,(-ww):]

        # average pooling, u_v -> u_1
        uv_u = torch.mean(uv, dim = -1, keepdim=True).expand([-1, -1, -1, ww])
        uv_v = torch.mean(uv, dim = -2, keepdim=True).expand([-1, -1, ww, -1])

        uw_u = torch.mean(uw, dim = -1, keepdim=True).expand([-1, -1, -1, ww])
        uw_w = torch.mean(uw, dim = -2, keepdim=True).expand([-1, -1, ww, -1])
        
        wv_w = torch.mean(wv, dim = -1, keepdim=True).expand([-1, -1, -1, ww])  
        wv_v = torch.mean(wv, dim = -2, keepdim=True).expand([-1, -1, ww, -1])

        f_uv_uv = torch.cat([uw_u, wv_v], dim = 1)
        f_uw_uw = torch.cat([uv_u, wv_w], dim = 1)
        f_wv_wv = torch.cat([uw_w, uv_v], dim = 1)

        f_layers = torch.cat([
            f_uv_uv, f_uw_uw, f_wv_wv
        ], dim = -1)

        x_cat = torch.cat(
            [x, f_layers], dim = 1
        )

        x_cat_up = self.up(x_cat)
        
        if self.use_bn:
            x_cat_up = self.bn(x_cat_up)
       
        if self.use_dropout:
            x_cat_up = self.drop(x_cat_up)
       
        if skip_input is not None:
            x_cat_up = torch.cat([x_cat_up, skip_input], 1)
        
        return x_cat_up

# also borrow from POP, :D
class UpConv2DBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=4, stride=2, padding=1,
                 use_bias=False, use_bn=True, up_mode='upconv', use_dropout=False):
        super(UpConv2DBlock, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.use_bn = use_bn
        self.use_dropout = use_dropout
        self.relu = nn.ReLU()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=use_bias)
            self.up.weight.data.normal_(0, 0.02)
        else:
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=1, stride=1),
            )
        
        if use_bn:
            self.bn = nn.BatchNorm2d(output_nc, affine=False)
        
        if use_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x, skip_input=None):
        x = self.relu(x)
        
        x = self.up(x)
        
        if self.use_bn:
            x = self.bn(x)
        
        if self.use_dropout:
            x = self.drop(x)
        
        if skip_input is not None:
            x = torch.cat([x, skip_input], 1)
        
        return x
    
class UnetD5Layers_rollout_256(nn.Module):
    def __init__(self, 
        input_nc_org = 9, input_nc = 16,
        output_nc = 12, last_activation = 'tanh',
        bottlenet_feature_dim = 16
    ):
        super().__init__()
        self.input_nc_org = input_nc_org
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.last_activation = last_activation
        self.bottlenet_feature_dim = bottlenet_feature_dim

        print("+++++ start initalizing ***UnetD5Layers****")
        print(
            ' number of real input  channel: \t \t', self.input_nc_org, '\n', 
            'number of 1st feature channel: \t \t', self.input_nc, '\n', 
            'number of real output channel: \t \t', self.output_nc, '\n', 
            'number of real output bottlenet_feature: \t', self.bottlenet_feature_dim
        )

        self.conv_uv = Conv2DBlock(input_nc_org,  input_nc,       3, 1, 1, use_bias=False, use_bn=False, use_relu = False)
        self.conv_uw = Conv2DBlock(input_nc_org,  input_nc,       3, 1, 1, use_bias=False, use_bn=False, use_relu = False)
        self.conv_wv = Conv2DBlock(input_nc_org,  input_nc,       3, 1, 1, use_bias=False, use_bn=False, use_relu = False)

        # 256 * 256 * 9 -> 128 * 128 * 16
        #self.conv0 = Conv2DBlock_Rodin(input_nc,      input_nc,       4, 2, 1, use_bias=False, use_bn=True)
        
        # 128 * 128 * 16 -> 64 * 64 * 32
        self.conv1 = Conv2DBlock_Rodin(input_nc * 1,  input_nc * 2,   4, 2, 1, use_bias=False, use_bn=True) 
        # 64 * 64 * 32 -> 32 * 32 * 64
        self.conv2 = Conv2DBlock_Rodin(input_nc * 2,  input_nc * 4,   4, 2, 1, use_bias=False, use_bn=True)
        # 32 * 32 * 64 -> 16 * 16 * 128
        self.conv3 = Conv2DBlock_Rodin(input_nc * 4,  input_nc * 8,   4, 2, 1, use_bias=False, use_bn=True)
        # 16 * 16 * 128 -> 8 * 8 * 256
        self.conv4 = Conv2DBlock_Rodin(input_nc * 8,  input_nc * 16,  4, 2, 1, use_bias=False, use_bn=True)
        # 16 * 16 * 128 -> 8 * 8 * 256
        self.conv5 = Conv2DBlock_Rodin(input_nc * 16,  input_nc * 32,  4, 2, 1, use_bias=False, use_bn=True)

        # -> here we do a concate

        self.upconv1 = UpConv2DBlock_Rodin(32  * input_nc  + bottlenet_feature_dim, 16 *  input_nc,  4, 2, 1, up_mode='upconv', use_dropout=True)
        self.upconv2 = UpConv2DBlock_Rodin(16  * input_nc  * 2, 8  *  input_nc,  4, 2, 1, up_mode='upconv', use_dropout=False) # 
        self.upconv3 = UpConv2DBlock_Rodin(8   * input_nc  * 2, 4  *  input_nc,  4, 2, 1, up_mode='upconv', use_dropout=False) # 
        self.upconv4 = UpConv2DBlock_Rodin(4   * input_nc  * 2, 2  *  input_nc,  4, 2, 1, up_mode='upconv', use_dropout=False) # 
        # 64  -> 16 
        self.upconv5 = UpConv2DBlock_Rodin(2   * input_nc  * 2,       input_nc,  4, 2, 1, up_mode='upconv', use_dropout=False) # 
        # 32 -> 16
        self.conv6 = UpConv2DBlock_Rodin(2 * input_nc, output_nc // 3,  3, 1, 1, up_mode='upconv', use_bn=False, use_dropout=False) # 
        
        self.relu_activation = nn.ReLU()
        self.tanh_activation = nn.Tanh()

        print("+++++ end initalizing ****UnetD5Layers****")

        
    def forward(self, x, bottle_net_feature = None):
        
        # first layer
        f_uv = self.conv_uv(x)
        f_uw = self.conv_uw(x)
        f_wv = self.conv_wv(x)

        d0 = torch.cat(
            [f_uv, f_uw, f_wv], dim = 3
        )
                
        d1 = self.conv1(d0)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        #if bottle_net_feature is not None:
        if bottle_net_feature is not None:
            bottle_net_feature = bottle_net_feature.unsqueeze(-1).unsqueeze(-1)
            d5 = torch.cat(
                [d5, bottle_net_feature.expand([-1, -1, d5.shape[-2], d5.shape[-1]])], dim = 1
            )

        u1 = self.upconv1(d5, d4)
        u2 = self.upconv2(u1, d3)
        u3 = self.upconv3(u2, d2)
        u4 = self.upconv4(u3, d1)
        u5 = self.upconv5(u4, d0)
        
        u6 = self.conv6(u5)
        
        ww = u6.shape[-1] // 3
        
        uv = u6[:,:,:,:ww]
        uw = u6[:,:,:,ww:(2 * ww)]
        wv = u6[:,:,:,(-ww):]

        fin_feats = torch.cat(
            [uv, torch.swapaxes(uw, 2, 3), torch.swapaxes(wv, 2, 3)], dim = 1
        )

        if self.last_activation == 'tanh':
            out = self.tanh_activation(fin_feats)
            out = (out + 1.0) / 2.0
        elif self.last_activation == 'relu':
            out = self.relu_activation(fin_feats)
        else:
            out = fin_feats

        return out
