"""
@File: triplane.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The triplane sampler 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def bilinear_sampler_pure_plane_bad_func(coords, _img, scale_0 = 1.0, scale_1 = 1.0):
    img = rearrange(_img, 'c h w -> h w c')
    D0, D1 = img.shape[0], img.shape[1]
    
    x, y = coords[:,0], coords[:,1]
    x = ((x / scale_0) + 1.) * (D0 - 1) * 0.5
    y = ((y / scale_1) + 1.) * (D1 - 1) * 0.5

    with torch.no_grad():
        x0_id = torch.floor(x)
        x1_id = x0_id + 1
        y0_id = torch.floor(y)
        y1_id = y0_id + 1 
    
    wa = (x1_id - x) * (y1_id - y)
    wb = (x - x0_id) * (y1_id - y)
    wc = (x1_id - x) * (y - y0_id)
    wd = (x - x0_id) * (y - y0_id)

    with torch.no_grad():
        x0_id = x0_id.clamp(0, D0 - 2)
        x1_id = x1_id.clamp(1, D0 - 1)
        y0_id = y0_id.clamp(0, D1 - 2)
        y1_id = y1_id.clamp(1, D1 - 1)

        x0_id = x0_id.long()
        x1_id = x1_id.long()
        y0_id = y0_id.long()
        y1_id = y1_id.long()
    
    Ia = img[x0_id, y0_id, :]
    Ib = img[x1_id, y0_id, :]
    Ic = img[x0_id, y1_id, :]
    Id = img[x1_id, y1_id, :]

    out = wa[...,None] * Ia + wb[...,None] * Ib + wc[...,None] * Ic + wd[...,None] * Id

    return out

class Triplane(nn.Module):
    def __init__(self, 
        resolution=256, channel=4, scale_d = 0.05
    ):
        super(Triplane, self).__init__()
        print("########################################################")
        
        self.resolution = resolution
        # channel for the surface
        self.channel = channel
        self.tot_channel = self.channel * 3
        self.scale_d = scale_d
        self.scale_z = 1.0 / self.scale_d

        # print the settings
        print(
            ' resolution: \t\t', self.resolution,'\n',
            'channel: \t\t', self.channel,'\n',
            'total channel: \t', self.tot_channel,'\n',
            'scale z: \t\t', self.scale_z,'\n'
        )
        
        print('+++++ triplane initialized ++++')
        print("########################################################")
  
    def forward(self, xyz, feature_maps = None):
        
        hh, ww = feature_maps.shape[-2], feature_maps.shape[-1]
        to_be_queried = feature_maps.reshape([3, self.channel, hh, ww])

        xy_coords = xyz[:,[0,1]]
        yz_coords = xyz[:,[1,2]]
        zx_coords = xyz[:,[2,0]]

        sampled_features_xy = bilinear_sampler_pure_plane_bad_func(
            xy_coords, to_be_queried[0,:, :, :], scale_0 = 1.0, scale_1 = 1.0
        )
        sampled_features_yz = bilinear_sampler_pure_plane_bad_func(
            yz_coords, to_be_queried[2,:, :, :], scale_0 = 1.0, scale_1 = self.scale_d
        )
        sampled_features_zx = bilinear_sampler_pure_plane_bad_func(
            zx_coords, to_be_queried[1,:, :, :], scale_0 = self.scale_d, scale_1 = 1.0
        )

        ret_sampled_feats = torch.cat(
            [sampled_features_xy, sampled_features_yz, sampled_features_zx], axis = -1
        )
        
        return ret_sampled_feats
