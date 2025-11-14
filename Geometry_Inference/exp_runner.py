
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
        self.data_loader_device = torch.device(conf['general']['data_loader_device'])
        self.device = torch.device(conf['general']['device'])
        
        print("++++++ Runner initialized with device:", self.device)
        print('++++++ Data loader device:', self.data_loader_device)
        
        ###########################################################################################
        
        self.base_exp_dir = conf['general']['base_exp_dir']
        self.output_base_exp_dir = conf['general']['output_base_exp_dir']
        print("++++++ Base exp dir:", self.base_exp_dir)
        
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        ###########################################################################################

        self.dataset = SurfaceFinetuneDataset(
            self.conf, device = self.data_loader_device
        )
        
        ###################################################################################################################
        #                                        template  related                                                        #
        ###################################################################################################################

        self.test_xyz_barycentric_info_dir = self.conf['dataset.test_xyz_barycentric_info_dir']
        self.subdivision_st = []
        self.subdivision_ed = []
        self.subdivision_faces = []

        self.vert_adj_faces = []
        self.vert_adj_weights = []

        # for computing normals
        self.face_idx = []
        self.vert_adj_faces_cu = []
        self.vert_adj_weights_cu = []
        self.face_idx_cu = []
        
        self.sparse_laplacians = []
        self.load_template_subdivison_info()

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
        
        self.smooth_iter_num = self.conf.get_int('train.surface_finetune.smooth_iter_num', default=2)
        
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
        
        os.makedirs(os.path.join(self.output_base_exp_dir, 'meshes'), exist_ok=True)           

        self.train_dataloader = DataLoader(
            self.dataset, 
            batch_size=1, 
            shuffle=False, 
            num_workers=1,
        )

        it = iter(self.train_dataloader)

        woot_bar = tqdm(range(num_frames_iter))

        for i in woot_bar:
            torch.cuda.synchronize()
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
                
                normalized_posed_mesh = normalized_mesh_vert_pos.clone().detach() + 2e-6
                surface_sample_pts_ = normalized_mesh_vert_pos.clone().detach()

                surface_sample_pts = (
                    surface_sample_pts_[self.subdivision_st[0],:] + surface_sample_pts_[self.subdivision_ed[0],:]
                ) * 0.5

                ones_vec = torch.ones(surface_sample_pts.shape[0], device=self.device)            
    
            for ttt in range(self.smooth_iter_num):

                lap = compute_laplacian(
                    surface_sample_pts.unsqueeze(0), self.sparse_laplacians[1]
                )[0]

                if len(self.hand_mask) > 0:
                    surface_sample_pts = surface_sample_pts - lap * self.hand_mask[1][...,None]
                else:
                    surface_sample_pts = surface_sample_pts - lap
                    
                surface_sample_pts = surface_sample_pts.detach().requires_grad_(True)
                
                surface_sample_pts.requires_grad_(True) 
                
                surface_samples_uvd, _ = self.render.mapping_wrapper(
                    surface_sample_pts, normalized_posed_mesh
                )
                
                surface_sample_features = self.render.feature_wrapper(
                    surface_samples_uvd, tri_feature
                )
                
                sdf_surface = self.sdf_network(surface_sample_features)[:,0]
                sdf_surface = torch.nan_to_num(sdf_surface,0)
                
                #d_output = torch.ones_like(sdf_surface, requires_grad=False, device=sdf_surface.device)
                
                global_gradient = torch.autograd.grad(
                    sdf_surface,
                    surface_sample_pts,
                    grad_outputs=ones_vec,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]

                global_normal = torch.nan_to_num(
                    F.normalize(global_gradient, dim=-1),
                    1e-6
                )
                
                if len(self.hand_mask) > 0:
                    fin_pos = surface_sample_pts - sdf_surface[:,None] * global_normal * self.hand_mask[1][...,None]
                else:
                    fin_pos = surface_sample_pts - sdf_surface[:,None] * global_normal

                surface_sample_pts = fin_pos.detach()              
            
            torch.cuda.synchronize()
            ed = time.time()
            
            # print fps
            print('fps:', 1./(ed - st))
            
            ret_posed_template = surface_sample_pts.cpu().numpy()
            raw_template = ret_posed_template * ret_dict['normalized_scale_with_grad'][0].cpu().numpy() + ret_dict['normalized_trans_with_grad'][0][0].cpu().numpy()

            output_file_name = os.path.join(
                self.output_base_exp_dir, 'meshes', str(ret_dict['current_frame_id'][0].numpy()) + '.ply'
            )
            
            trimesh_obj = trimesh.Trimesh(
                vertices=raw_template,
                faces=self.subdivision_faces[1],
                process=False
            )
            trimesh_obj.export(output_file_name)
            
        return 
        

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
    
    
    def load_template_subdivison_info(self):
        subdivision_dict = pkl.load(
            open(self.test_xyz_barycentric_info_dir,'rb')
        )
                
        self.subdivision_faces = [
            subdivision_dict['stage_0_dict']['faces'],
            subdivision_dict['stage_1_dict']['faces'],
            subdivision_dict['stage_2_dict']['faces']
        ]

        vert_adj_faces_1, vert_adj_weights_1 = self.compute_subd_adj(self.subdivision_faces[1])

        self.vert_adj_faces = [
            [], vert_adj_faces_1, []
        ]
        self.vert_adj_weights = [
            [], vert_adj_weights_1, []   
        ]

        # for computing normals
        self.face_idx = [
            [], subdivision_dict['stage_1_dict']['faces'], []
        ]

        self.vert_adj_faces_cu = [
            [], torch.LongTensor(vert_adj_faces_1).to(self.device), []
        ]
        self.vert_adj_weights_cu = [
            [], torch.FloatTensor(vert_adj_weights_1).to(self.device), []
        ]
        self.face_idx_cu = [
            [], torch.LongTensor(subdivision_dict['stage_1_dict']['faces']).to(self.device), []
        ]

        # laplacian level 0
        self.sparse_laplacians = [
            [],
            self.compute_laplacian(self.subdivision_faces[1]),
            []
        ]

        # also the hand mask 
        
        self.subdivision_st = [
            torch.LongTensor(subdivision_dict['stage_1_dict']['prev_to_cur_st']).to(self.device),
            torch.LongTensor(subdivision_dict['stage_2_dict']['prev_to_cur_st']).to(self.device)
        ]
    
        self.subdivision_ed = [
            torch.LongTensor(subdivision_dict['stage_1_dict']['prev_to_cur_ed']).to(self.device),
            torch.LongTensor(subdivision_dict['stage_2_dict']['prev_to_cur_ed']).to(self.device)
        ]

        self.hand_mask = []

        if self.dataset.charactor.hand_mask is not None:
            #print(self.dataset.charactor.hand_mask.shape, type(self.dataset.charactor.hand_mask))
            hand_mask_1 = torch.min(
                self.dataset.charactor.hand_mask[self.subdivision_st[0]], self.dataset.charactor.hand_mask[self.subdivision_ed[0]], 
            ).clone().detach().to(self.device)

            self.hand_mask = [
                [], hand_mask_1, []
            ]

        return    

    def compute_subd_adj(self, face_idx):
        
        print(face_idx.shape)
        vert_adj_faces = []
        vert_adj_weights = []

        max_vert_ind = -114514
        min_vert_ind =  114514

        vert_num = np.max(face_idx) + 1

        temp_adj_list = [[] for i in range(vert_num)]

        for i in range(face_idx.shape[0]):
            t0, t1, t2 = face_idx[i][0], face_idx[i][1], face_idx[i][2]
            temp_adj_list[t0].append(i)
            temp_adj_list[t1].append(i)
            temp_adj_list[t2].append(i)

        for i in range(len(temp_adj_list)):
            max_vert_ind = max(len(temp_adj_list[i]), max_vert_ind)
            min_vert_ind = min(len(temp_adj_list[i]), min_vert_ind)
  
        assert min_vert_ind >= 1

        for i in range(len(temp_adj_list)):
            
            cur_adj_num = len(temp_adj_list[i])
            tmp_faces_idx = []
            tmp_weights = []

            for j in range(max_vert_ind + 1):
            
                if j < cur_adj_num:
                    tmp_faces_idx.append(temp_adj_list[i][j])
                    tmp_weights.append(1.0/cur_adj_num)
                else:
                    tmp_faces_idx.append(temp_adj_list[i][-1])
                    tmp_weights.append(0.0)
            
            vert_adj_faces.append(tmp_faces_idx)
            vert_adj_weights.append(tmp_weights)

        
        vert_adj_faces = np.array(vert_adj_faces, dtype=np.int32)
        vert_adj_weights = np.array(vert_adj_weights, dtype=np.float32)
        
        return vert_adj_faces, vert_adj_weights

    def compute_laplacian(self, np_faces):
        print('computing sparse laplaican', np_faces.shape)
        num_verts = np.max(np_faces) + 1
        max_ind = -114514
        min_ind = 114514

        print('number of vertices', num_verts)
        
        verticesNeighborID = []
        temp_verticesNeighborID = [[] for i in range(num_verts)]
        
        for i in range(len(np_faces)):
            v0, v1, v2 = np_faces[i]
            
            temp_verticesNeighborID[v0].append(v1)
            temp_verticesNeighborID[v0].append(v2)

            temp_verticesNeighborID[v1].append(v0)
            temp_verticesNeighborID[v1].append(v2)

            temp_verticesNeighborID[v2].append(v0)
            temp_verticesNeighborID[v2].append(v1)

        for i in range(num_verts):
            cur_neighbor = temp_verticesNeighborID[i]
            cur_neighbor = list(set(cur_neighbor))
            cur_neighbor.sort()
            
            max_ind = max(len(cur_neighbor), max_ind)
            min_ind = min(len(cur_neighbor), min_ind)

            verticesNeighborID.append(cur_neighbor)

        print('max and min ind', max_ind,' ', min_ind)

        laplacian_temp_st = []
        laplacian_temp_ed = []
        laplacian_temp_weight = []
        
        for i in range(len(verticesNeighborID)):
            cur_st = []
            cur_ed = []
            cur_weight = []
            cur_arr = verticesNeighborID[i]
            
            cur_st.append(i)
            cur_ed.append(i)
            cur_weight.append(1.0)

            for j in range(len(cur_arr)):
                cur_st.append(i)
                cur_ed.append(cur_arr[j])
                cur_weight.append(-1.0 / (1.0 * len(cur_arr)))

            laplacian_temp_ed.append(cur_ed)
            laplacian_temp_st.append(cur_st)
            laplacian_temp_weight.append(cur_weight)

        laplacian_temp_st = torch.LongTensor(np.concatenate(laplacian_temp_st, axis = 0)).to(self.device)
        laplacian_temp_ed = torch.LongTensor(np.concatenate(laplacian_temp_ed, axis = 0)).to(self.device)
        laplacian_temp_weight = torch.FloatTensor(np.concatenate(laplacian_temp_weight, axis = 0)).to(self.device)        

        sparse_laplacian = torch.sparse_coo_tensor(
            indices= torch.stack([laplacian_temp_st, laplacian_temp_ed], dim = 0),
            values=laplacian_temp_weight, size=(num_verts, num_verts), device=self.device
        )
        sparse_laplacian = sparse_laplacian.coalesce()
        return sparse_laplacian       


        


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
