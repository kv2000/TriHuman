
"""
@File: exp_runner.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The main entry
"""

import os
import sys
sys.path.append("./")
sys.path.append("../")

import time
from icecream import ic
import argparse
from pyhocon import ConfigFactory
import numpy as np
import cv2 as cv
import trimesh
from tqdm import tqdm
import math
from icecream import ic
ic.configureOutput(includeContext=False)

import pickle as pkl
from PIL import Image
from einops import rearrange
import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.dataset import SurfaceFinetuneDataset


from models.renderer import TrihumanRender
from models.triplane import Triplane
from models.Unet_rollout_large import UnetD5Layers_rollout_256 as UnetD5Layers
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, MotionNetwork
from models.wootGlobalToLocal import wootGlobalToLocal
from models.WootCharacter import compute_laplacian


class Runner:
    def __init__(self, conf):
        self.conf = conf
        # put on spearate device for data loading and model, makes it fasster
        self.data_loader_device = torch.device(conf['general']['data_loader_device'])
        self.device = torch.device(conf['general']['device'])
        
        print("++++++ Runner initialized with device:", self.device)
        print('++++++ Data loader device:', self.data_loader_device)
        
        ###########################################################################################
        
        self.base_exp_dir = conf['general']['base_exp_dir']
        self.output_base_exp_dir = conf['general']['output_base_exp_dir']
        print("++++++ Base exp dir:", self.base_exp_dir)
        print("++++++ Output base exp dir:", self.output_base_exp_dir)
        
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        ###########################################################################################

        self.dataset = SurfaceFinetuneDataset(
            self.conf, device = self.data_loader_device
        )

        ###################################################################################################################
        #                                        rendering related                                                        #
        ###################################################################################################################

        self.no_split = self.conf.get_bool('dataset.no_split', default = False)
        self.no_split = True  # always true for inference
        self.split_per_ray = self.conf.get_int('dataset.split_per_ray', default=65536)
                
        ###################################################################################################################
        #                                        Model Initialization                                                     #
        ###################################################################################################################

        self.global_to_local_cand_num = self.conf.get_int('mapping.global_to_local_cand_num')
        self.jelly_offset = self.conf.get_float('mapping.jelly_offset')
        self.unet_input_size = self.conf.get_int('dataset.barycentric_tex_size')
        self.unet_output_size = self.conf.get_int('model.triplane.resolution', default=256)

        self.use_triplane = self.conf.get_bool('model.neus_renderer.use_triplane', default=True)
        self.use_translation = self.conf.get_bool('model.neus_renderer.use_translation', default=True)
        self.use_global_motion_code = self.conf.get_bool('model.extra_settings.use_global_motion_code', default=False)
                
        # feture extractor
        self.feature_extractor = UnetD5Layers(
            **self.conf['model.feature_extractor']
        ).to(self.device)

        if self.use_global_motion_code:
            self.global_motion_mlp = MotionNetwork(
                **self.conf['model.global_motion_network']
            ).to(self.device)
        else:
            self.global_motion_mlp = None

        # init the networks
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        if self.use_triplane:
            self.triplane_network = Triplane(**self.conf['model.triplane'])
        else:
            self.triplane_network = None

        # load the networks
        self.is_load_implicit_checkpoints = self.conf.get_bool('train.load_implicit_checkpoints', default=False)
        self.implicit_checkpoint_dir = self.conf['train.implicit_checkpoint_dir']

        if self.is_load_implicit_checkpoints:
            self.load_implicit_init_checkpoint()
            
        ###########################################################################################
        
        self.init_global_to_local()

        self.render = TrihumanRender(
            self.sdf_network, self.deviation_network, self.color_network, 
            **self.conf['model.neus_renderer'],
            global_to_local = self.global_to_local,
            triplane = self.triplane_network,
            padding_height = self.jelly_offset,
            device=self.device
        )
        
        ############################################################################################
        self.run_inference()

    # loading implicit checkpoints, including the sdf and the rendering module
    def load_implicit_init_checkpoint(self):
        print('+++++ init with implicit checkpoint', self.implicit_checkpoint_dir)
        
        if os.path.isfile(self.implicit_checkpoint_dir):
            cur_state_dict = torch.load(self.implicit_checkpoint_dir, map_location=self.device)      
                    
            if (self.sdf_network is not None) and ('sdf_network' in cur_state_dict.keys()):
                print('+++++ loading checkpoints sdf_network')
                self.sdf_network.load_state_dict(cur_state_dict['sdf_network'])     

            if (self.deviation_network is not None) and ('deviation_network' in cur_state_dict.keys()):
                print('+++++ loading checkpoints deviation_network')
                self.deviation_network.load_state_dict(cur_state_dict['deviation_network'])

            if (self.color_network is not None) and ('color_network' in cur_state_dict.keys()):
                print('+++++ loading checkpoints color_network')
                self.color_network.load_state_dict(cur_state_dict['color_network'])

            if (self.feature_extractor is not None) and ('feature_extractor' in cur_state_dict.keys()) :
                print('+++++ loading checkpoints feature_extractor')
                self.feature_extractor.load_state_dict(cur_state_dict['feature_extractor'])

            if (self.global_motion_mlp is not None) and ('global_motion_mlp' in cur_state_dict.keys()) :
                print('+++++ loading checkpoints global_motion_mlp')
                self.global_motion_mlp.load_state_dict(cur_state_dict['global_motion_mlp'])
          
        else:
            print(self.implicit_checkpoint_dir, 'check point not found')

        return 
    
    def run_inference(self):

        num_frames_iter = len(self.dataset.chosen_frame_id_list)
        print(
            'frame num:', len(self.dataset.chosen_frame_id_list),
            '\n total sample num:', num_frames_iter
        )

        self.render.set_eval()
        self.feature_extractor.eval()
        self.global_motion_mlp.eval() 

        for param in self.deviation_network.parameters():
            param.grad = None

        for param in self.color_network.parameters():
            param.grad = None
        
        for each_cam in self.dataset.val_camera:
            os.makedirs(os.path.join(self.output_base_exp_dir, 'validations_val', str(each_cam)), exist_ok=True)
 

        self.train_dataloader = DataLoader(
            self.dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=1,
        )

        it = iter(self.train_dataloader)

        woot_bar = tqdm(range(num_frames_iter))

        for i in woot_bar:
            #torch.cuda.synchronize()
            st = time.time()
            with torch.no_grad():
                ret_dict = next(it)
                
                dofs = ret_dict['trans_normalized_dofs']
                
                dofs = dofs.float().to(self.device)
                
                global_motion_code = self.global_motion_mlp(dofs)

                # rendered motion feature
                pose_feature = ret_dict['pose_feature']
                pose_feature = rearrange(pose_feature, 'b h w c -> b c h w')

                normalized_mesh_vert_pos = ret_dict['normalized_ret_posed_delta_with_grad'][0][0]

                tri_feature = self.feature_extractor(
                    pose_feature, global_motion_code
                )

                # foreground rays
                rays_o = ret_dict['rays_o'][0]
                rays_d = ret_dict['rays_v'][0]
                near = ret_dict['near'][0]
                far = ret_dict['far'][0]
                pixel_idh_np = ret_dict['pixel_idh_np'][0]
                pixel_idw_np = ret_dict['pixel_idw_np'][0]
                    
                global_translation_input = ret_dict['global_translation'][0].float().to(self.device)
                global_scale_input = ret_dict['global_scale'][0]

            H, W = self.dataset.img_height // self.dataset.img_scale_factor, self.dataset.img_width // self.dataset.img_scale_factor
            non_empty_pixel_num = pixel_idh_np.shape[0]
            out_rgb_fine = np.zeros(shape=(H,W,3),dtype=np.float32)       

            # process in split way, the it might be slower
            if (not self.no_split) and (non_empty_pixel_num > self.split_per_ray):
                
                rays_o = rays_o.cpu().numpy()
                rays_d = rays_d.cpu().numpy()
                near = near.cpu().numpy()
                far = far.cpu().numpy()
                
                rays_o = np.array_split(
                    rays_o.reshape(-1, 3), ((non_empty_pixel_num // self.split_per_ray) + 1)
                )
                rays_d = np.array_split(
                    rays_d.reshape(-1, 3), ((non_empty_pixel_num // self.split_per_ray) + 1)
                )
                
                pixel_idh_np = np.array_split(
                    pixel_idh_np.reshape(-1), ((non_empty_pixel_num // self.split_per_ray) + 1)
                )
                pixel_idw_np = np.array_split(
                    pixel_idw_np.reshape(-1), ((non_empty_pixel_num // self.split_per_ray) + 1)
                )

                far = np.array_split(far.reshape(-1, 1), ((non_empty_pixel_num // self.split_per_ray) + 1))
                near = np.array_split(near.reshape(-1, 1), ((non_empty_pixel_num // self.split_per_ray) + 1))
                
                for rays_o_batch, rays_d_batch, pixel_idh_np_batch, pixel_idw_np_batch, near_batch, far_batch in zip(
                    rays_o, rays_d, pixel_idh_np, pixel_idw_np, near, far
                ):
                    rays_o_batch = torch.FloatTensor(rays_o_batch).to(self.device)
                    rays_d_batch = torch.FloatTensor(rays_d_batch).to(self.device)
                    
                    near_batch = torch.FloatTensor(near_batch).to(self.device)
                    far_batch = torch.FloatTensor(far_batch).to(self.device)
                    
                    render_out = self.render.forward(
                        rays_o_batch, rays_d_batch, near_batch, far_batch, normalized_mesh_vert_pos,
                        cos_anneal_ratio = 1.0,
                        pose_feature = tri_feature,
                        global_translation=global_translation_input,
                        global_scale=global_scale_input,
                        is_train=False
                    )
                    
                    temp_out_color = render_out['color_fine'][:,[2,1,0]].detach().cpu().numpy()      
                    out_rgb_fine[pixel_idh_np_batch, pixel_idw_np_batch, :3] = temp_out_color                 
            # need a proper gpu to process all rays together,     
            else:          
                rays_o = rays_o.to(self.device)
                rays_d = rays_d.to(self.device)
                    
                near = near.to(self.device)
                far = far.to(self.device)

                render_out = self.render.forward(
                    rays_o, rays_d, near, far, normalized_mesh_vert_pos,
                    cos_anneal_ratio = 1.0,
                    pose_feature = tri_feature,
                    global_translation=global_translation_input,
                    global_scale=global_scale_input,
                    is_train=False
                )
                
                temp_out_color = render_out['color_fine'][:,[2,1,0]].detach().cpu().numpy()
                out_rgb_fine[pixel_idh_np, pixel_idw_np, :3] = temp_out_color
                

            img_fine = (out_rgb_fine * 256).clip(0, 255)
            cv.imwrite(os.path.join(
                self.output_base_exp_dir, 'validations_val', str(ret_dict['camera_id'][0].item()), str(ret_dict['current_frame_id'][0].item()) + '.png'
            ), img_fine.astype(np.uint8))   
                        

    def init_global_to_local(self):
        print('+++++ start to initialize the global to local layer')
        
        temp_faces = np.array(self.dataset.charactor.obj_reader.facesVertexId).reshape([-1, 3])

        temp_vert_num = np.max(temp_faces) + 1

        self.global_to_local = wootGlobalToLocal(
            vert_num = temp_vert_num,
            _faces = temp_faces,
            _uv_coords = np.array(self.dataset.charactor.obj_reader.textureCoordinates).reshape([-1,3,2]),
            _cand_num = self.global_to_local_cand_num,
            _jelly_offset = self.jelly_offset,
            device=self.device
        )

        print('+++++ end to initialize the global to local layer')


        


if __name__ == '__main__':
    print('wootwootwo')

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/wmask_1024_multi_continue_wo_boundary_low_normal.conf')
    args = parser.parse_args()

    f = open(args.conf)
    conf_text = f.read()
    f.close()
    preload_conf = ConfigFactory.parse_string(conf_text)

    runner = Runner(
        preload_conf
    )
