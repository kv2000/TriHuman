import os
import sys
sys.path.append("../")
import trimesh

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
#import mcubes
import time

class TrihumanRender(nn.Module):
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 use_triplane = False,
                 use_uvd = False,
                 padding_height = 0.049,
                 global_to_local = None,
                 batch_size = 4096,
                 device = 'cuda:0',
                 use_translation = False,
                 triplane = None,  
        ):
        super(TrihumanRender, self).__init__()

        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.global_to_local=global_to_local
        self.padding_height = padding_height
        self.use_triplane = use_triplane
        self.use_uvd = use_uvd
        self.batch_size = batch_size
        self.device = device
        self.use_translation = use_translation
        self.triplane = triplane

    def set_train(self):
        
        self.sdf_network.train()
        self.deviation_network.train()
        self.color_network.train()
        
        return 

    def set_eval(self):
        
        self.sdf_network.eval()
        self.deviation_network.eval()
        self.color_network.eval()
        
        return 
    
    def mapping_wrapper(self, xyz, mesh_vert_pos=None, _padding_height=None, real_cloest_fid = None):
        real_padding_height = self.padding_height
        
        if _padding_height is not None:
            real_padding_height = _padding_height
        
        ret_coords = None
        ret_mask = None
        
        if self.use_uvd:
            fin_uv, uv_distance, _, _, _, _, fin_m_type, fin_proj_pts, fin_fn = self.global_to_local.forward_with_real_fid(
                xyz, mesh_vert_pos, real_cloest_fid
            )
            ret_coords = torch.cat([fin_uv, uv_distance[...,None]], dim = -1)
            ret_mask = torch.abs(uv_distance) < real_padding_height
        else:
            ret_coords = xyz
            ret_mask = torch.ones(xyz.shape[0]).bool().to(self.device)

        return ret_coords, ret_mask

    def feature_wrapper(self, x, feat=None):
        
        ret_feature = None

        if self.use_triplane and feat is not None:
            ret_feature = torch.cat([x, self.triplane(x, feature_maps=feat)], dim = -1)
        else:
            ret_feature = x
        
        return ret_feature


    def forward(self, rays_o_, rays_d_, near_, far_, normalized_vert_pos_, cos_anneal_ratio = 0., 
        pose_feature = None, global_translation = None, global_scale = None, is_train = False
    ):       
        z_vals = torch.linspace(0.0, 1.0, self.n_samples + 1).float().to(self.device)
                    
        near, far = near_.reshape([-1, 1]), far_.reshape([-1, 1])   
        z_vals = near + (far - near) * z_vals[None, :]

        batch_size, n_samples = z_vals.shape[0], z_vals.shape[1] - 1
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        mid_z_vals = z_vals[..., :-1] + dists * 0.5
        
        rays_o, rays_d = rays_o_, rays_d_
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]

        global_dir = rays_d[:, None, :].expand(pts.shape)
        
        flattend_sample_pts = pts.reshape(shape=[-1, 3])
        flattend_sample_pts = flattend_sample_pts.contiguous()

        normalized_mesh_vert_pos = normalized_vert_pos_
        normalized_mesh_vert_pos = normalized_mesh_vert_pos.contiguous()

        flattend_sample_pts.requires_grad_(True)

        real_cloest_fid, merged_distance = self.global_to_local.get_real_fid(
            flattend_sample_pts.detach(), normalized_mesh_vert_pos.detach()
        )

        fin_mask = torch.abs(merged_distance) < self.padding_height
        flattend_sample_pts_grad = flattend_sample_pts[fin_mask]
        
        real_cloest_fid = real_cloest_fid.detach()
        flattend_sample_pts_grad.requires_grad_(True)
        
        fin_uvd, _ = self.mapping_wrapper(
            flattend_sample_pts_grad, normalized_mesh_vert_pos,
            real_cloest_fid = real_cloest_fid[fin_mask]
        )

        input_feats = self.feature_wrapper(
            fin_uvd, pose_feature
        )
                
        sdf_nn_output = self.sdf_network(input_feats)
        sdf_mock = sdf_nn_output[:,:1]
        
        feature_vector = sdf_nn_output[:,1:]

        d_output = torch.ones_like(sdf_mock, requires_grad=False, device=sdf_mock.device)
        global_gradient_mock = torch.autograd.grad(
            outputs=sdf_mock,
            inputs=flattend_sample_pts_grad,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        global_gradient_mock = global_gradient_mock[:,:3]   
        global_dir = global_dir.reshape([-1, 3])
        global_dir = global_dir.detach()
        
        differed_sample_pts = flattend_sample_pts * global_scale + global_translation
        mock_sampled_color = self.color_network(
            differed_sample_pts[fin_mask], global_gradient_mock, global_dir[fin_mask], feature_vector
        )

        # roll back to the grid
        sampled_color = torch.zeros_like(flattend_sample_pts)
        sampled_color[fin_mask] = mock_sampled_color
        sampled_color = sampled_color.reshape([batch_size, n_samples, 3])

        sdf = torch.zeros([flattend_sample_pts.shape[0], 1], device=self.device)
        sdf[fin_mask] = sdf_mock 
        
        global_gradient = torch.zeros_like(flattend_sample_pts)
        global_gradient[fin_mask] = global_gradient_mock
        
        # the original computation
        inv_s = self.deviation_network(torch.zeros([1, 3]).to(self.device))[:, :1].clip(1e-6, 1e6) # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)
        
        true_cos = torch.sum(
            (global_dir * global_gradient), dim = -1, keepdims=True
        )

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive
        
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        gradient_mask = fin_mask.detach()

        masked_global_gradient = torch.nan_to_num(
            global_gradient * gradient_mask.reshape([-1, 1]).float()
        )

        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        masked_alpha = alpha * gradient_mask.reshape([batch_size, n_samples])

        weights = masked_alpha * torch.cumprod(
            torch.cat(
                [torch.ones([batch_size, 1]).to(self.device), 1. - masked_alpha + 1e-7], -1
            ), 
        -1)[:, :-1]

        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (
            torch.nan_to_num(sampled_color * weights[:, :, None])
        ).sum(dim=1)

        return {
            'color_fine': color,
            'weights_sum': weights_sum,
        }
