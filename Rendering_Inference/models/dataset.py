import sys
import os
sys.path.append("../")
sys.path.append("../../")

import numpy as np
from models.utils import gen_uv_barycentric, dilate_barycentric_maps, load_calibrations_v10, load_K_Rt_from_P
import AdditionalUtils.CSVHelper as CSVHelper
import AdditionalUtils.OBJReader as OBJReader
import AdditionalUtils.CameraReader as CameraReader

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

import torch.multiprocessing as mp
mp.set_start_method(method='forkserver', force=True)

from models.WootCharacter import WootCharacter
from models.WootGCN import WootSpatialGCN
from CudaRenderer.CudaRendererGPU import CudaRendererGpu
from models.depth_renderer import renderDepth


from tqdm import tqdm
import pickle as pkl
import random
import math
from PIL import Image
import cv2 as cv

class SurfaceFinetuneDataset(Dataset):
    def __init__(self, conf,device = 'cuda'):
        print("+++++ start creating dataset  on device", device)
        self.conf = conf
        # first create the directory
        self.never_stop_size = int(1e7)
        self.device = device
        self.render_device = conf.get('general.device', default='cuda:0')
        
        if self.device != self.render_device:
            print("+++++ using separate device for data loading and rendering:", self.device, self.render_device)

        #################################################################################################
    
        self.dof_dir = self.conf['dataset.skeleton_angles']
        self.dof_angle_normalized_dir = self.conf['dataset.skeleton_angles_rotation_normalized']
        self.base_trans_dir = self.conf['model.extra_settings.base_trans_dir']
        self.precomputed_human_dir = None

        # WARNing -> precomputed human 
        self.use_precomputed_human = self.conf.get_bool('dataset.use_precomputed_human', default=False)
        #################################################################################################

        self.start_frame = self.conf.get_int('dataset.start_frame') 
        self.end_frame = self.conf.get_int('dataset.end_frame')
        self.sample_interval = self.conf.get_int('dataset.sample_interval')
        
        print("+++++ start frame:", self.start_frame)
        print("+++++ end frame:", self.end_frame)
        print("+++++ sample interval:", self.sample_interval)
        
        if self.use_precomputed_human:
            self.precomputed_human_dir = os.path.join(
                self.conf['general.base_exp_dir'], 'precomputed_human',  str(self.start_frame) + '_' + str(self.end_frame) + '_' + str(self.sample_interval)
            )
            print("+++++ use precomputed human from :", self.precomputed_human_dir)
            os.makedirs(self.precomputed_human_dir, exist_ok=True)
        else:
            print("+++++ do not use precomputed human ")
            
    
        self.chosen_frame_id_list = [
            i for i in range(self.start_frame, self.end_frame + 1, self.sample_interval)
        ]
        
        self.chosen_frame_id_list.sort()
        self.chosen_frame_id_list_0 = [(x - 1) for x in self.chosen_frame_id_list]
        self.chosen_frame_id_list_1 = [(x - 2) for x in self.chosen_frame_id_list]
            
        print("+++++ total frame num:", len(self.chosen_frame_id_list))
        
        #################################################################################################
        # prepare cameara related
        #################################################################################################

        self.img_width = self.conf.get_int('dataset.img_width')
        self.img_height = self.conf.get_int('dataset.img_height')
        # the depth offset for sampling
        self.depth_offset = self.conf.get_int('mapping.depth_offset')
        
        self.camera_reader = None
        self.camera_dir = self.conf['dataset.camera_dir']
        self.val_camera = self.conf.get_list('dataset.val_camera')
        
        # the number of pixels to scale down the image for faster rendering, 2 means half size
        self.img_scale_factor = self.conf.get_int('dataset.img_scale_factor', default=1)

        self.init_camera_info()        
 
        #################################################################################################
        # load base trans for the subject
        self.base_trans = pkl.load(open(self.base_trans_dir,'rb'))['base_trans']
        self.global_scale = self.conf.get_float('dataset.global_scale')
        self.real_global_scale = 2.5

        #################################################################################################
        # explicit charactor, which is the spatial and delta gcn
        #################################################################################################   
        
        self.charactor = None
        self.initialize_charactor()

        self.spatial_gcn = None
        self.eg_checkpoint_dir = self.conf['train.eg_checkpoint_dir']
        self.initialize_spatial_gcn()
        self.is_load_eg_checkpoints = self.conf.get_bool('train.load_eg_checkpoints', default=False)
        if self.is_load_eg_checkpoints:
            self.load_eg_init_checkpoint()

        self.delta_gcn = None
        self.delta_checkpoint_dir = self.conf['train.delta_checkpoint_dir']
        self.initialize_delta_gcn()
        self.is_load_delta_checkpoints = self.conf.get_bool('train.load_delta_checkpoints', default=False)
        if self.is_load_delta_checkpoints:
            self.load_delta_init_checkpoint()

        self.spatial_gcn.eval()
        self.delta_gcn.eval()
        
        for each_param in self.spatial_gcn.parameters():
            each_param.grad = None

        for each_param in self.delta_gcn.parameters():
            each_param.grad = None
            
        #################################################################################################

        self.dof_arr = None
        self.dof_angle_normalized_arr = None
        
        self.num_dof = -1
        self.tot_frame_num = -1
        self.normalization_arr = None
        
        #################################################################################################

        self.load_dof()
        self.init_normalization_info()

        # for the charactors with hands
        self.hand_dof_mask_dir = None
        self.hand_dof_mask = None 
        self.with_hand_dof_mask = self.conf.get_bool('model.extra_settings.with_hand_dof_mask',default=False)
        
        if self.with_hand_dof_mask:
            self.hand_dof_mask_dir = self.conf['model.extra_settings.hand_dof_mask_dir']
            self.hand_dof_mask = np.array(pkl.load(open(self.hand_dof_mask_dir,'rb'))['body_dof_in_full_dof'])

        ##################################################################################################
        # the uv related 
        ###################################################################################################

        self.uv_face_id = None
        self.uv_bary_weights = None
        self.uv_non_empty = None
        self.uv_non_empty_mask = None
        self.uv_coord_img = None
        
        self.uv_idx_np = None
        self.uv_idx_cu = None
        self.uv_idy_np = None
        self.uv_idy_cu = None

        self.uv_vert_idx_np = None
        self.uv_vert_idx_cu = None
        self.uv_bary_weights_cu = None

        #################################################################################################      
        self.obj_reader = None
        
        self.face_idx = None
        self.face_texture_coords = None
        self.vert_adj_faces = None
        self.vert_adj_weights = None

        self.face_idx_cu = None
        self.vert_adj_faces_cu = None
        self.vert_adj_weights_cu = None

        self.vert_num = None

        self.max_vert_ind = 0
        self.min_vert_ind = 114514
        self.barycentric_tex_size = self.conf.get_int('dataset.barycentric_tex_size')

        self.init_template_mesh_info()
        
        #################################################################################################
        self.worker_num = self.conf.get_int('general.data_loader_worker_num', default=1)
        if self.worker_num >= 1:
            self.set_tensor_dense()
            
        #################################################################################################
        # init the depth rendering
        #################################################################################################        
        self.render_func = None
        self.init_cuda_renderer()

        print("+++++ end creating dataset ")

    def init_cuda_renderer(self):
        print('+++++ start to initialize the cuda renderer')
        
        self.render_func = CudaRendererGpu(
            faces_attr = torch.LongTensor(self.obj_reader.facesVertexId).to(device=self.device).reshape([-1,3]).contiguous(),
            texCoords_attr = torch.FloatTensor(self.obj_reader.textureCoordinates).to(device=self.device).reshape([-1,3,2]).contiguous(),
            textureResolutionV = self.obj_reader.texHeight,
            textureResolutionU = self.obj_reader.texWidth
        )

        print('+++++ end to initialize the cuda renderer')
        return 

    def init_camera_info(self):
        print('+++++ start init world matrix from file:', self.camera_dir)
        calibration_arr = load_calibrations_v10(self.camera_dir)
        
        self.world_camera_arr = []   
        self.camera_reader = CameraReader.CameraReader(
            self.camera_dir
        )

        for i in range(len(calibration_arr)):
            P = np.dot(
                calibration_arr[i]['intrinsics'],
                calibration_arr[i]['extrinsics']
            )
            final_world_matrix = np.concatenate(
                [P,np.array([[0,0,0,1]])], axis=0
            )
            self.world_camera_arr.append(final_world_matrix)
        
        print('+++++ end init world matrix from file')
        
        return


    def set_tensor_dense(self):
        print("+++++ START Setting sparse tensor dense:")
        
        self.charactor.set_dense()
        
        if self.delta_gcn is not None:
            self.delta_gcn.set_dense()
        
        if self.spatial_gcn is not None:
            self.spatial_gcn.set_dense()

        print("+++++ END Setting sparse tensor dense:")
        return 
    
    def init_template_mesh_info(self):
        print('+++++ create template mesh related from:', self.charactor.template_mesh_dir)
        self.obj_reader = self.charactor.obj_reader
        
        self.face_idx = np.array(self.obj_reader.facesVertexId).reshape([-1, 3])
        self.face_texture_coords = np.array(self.obj_reader.textureCoordinates).reshape([-1,3,2])
        self.vert_num = self.obj_reader.numberOfVertices
        
        self.gen_adjacent_list()           
        self.gen_barycentric_coords()

        print('+++++ end create template mesh related')
        return 

    def load_dof(self):
        print("+++++ Loading All Sorts of dofs")
        # end frame for train and val 
        max_end_frame = self.end_frame + 1

        # here there is hack just for early end of the csv loader
        self.dof_arr = CSVHelper.load_csv_sequence_2D(
            self.dof_dir, type='float', skipRows=1, skipColumns=1, end_frame=(self.end_frame + 1)
        )
        self.num_dof = self.dof_arr.shape[-1]
        self.tot_frame_num = self.dof_arr.shape[0]

        self.dof_angle_normalized_arr = (CSVHelper.load_csv_compact_4D(
            self.dof_angle_normalized_dir, 3, self.num_dof, 1, 1, 1, 'float', end_frame=(3 * (max_end_frame + 1))
        )).reshape((-1, 3, self.num_dof))

        print(
            ' dof shape: ', self.dof_arr.shape, '\n',
            'dof rotation normalized shape: ',self.dof_angle_normalized_arr.shape, '\n'
        ) 
        print("+++++ Finished Loading All Sorts of dofs")
        return

    def init_normalization_info(self):
        print('+++++ start initialization with motion file:',self.dof_dir)
     
        self.normalization_arr = np.zeros(shape=[self.tot_frame_num,4,4])
        
        self.normalization_arr[:,0,0] = self.global_scale
        self.normalization_arr[:,1,1] = self.global_scale
        self.normalization_arr[:,2,2] = self.global_scale
        self.normalization_arr[:,3,3] = 1.

        # hard code for skeletontool
        self.normalization_arr[:,0,3] = self.dof_arr[:,0] * 1000.
        self.normalization_arr[:,1,3] = self.dof_arr[:,1] * 1000.
        self.normalization_arr[:,2,3] = self.dof_arr[:,2] * 1000.
        
        print('+++++ finish initialization with motion file')
        return 

    def gen_adjacent_list(self):   
        
        self.vert_adj_faces = []
        self.vert_adj_weights = []
        
        temp_adj_list = [[] for i in range(self.vert_num)]

        for i in range(self.face_idx.shape[0]):
            t0, t1, t2 = self.face_idx[i][0], self.face_idx[i][1], self.face_idx[i][2]
            temp_adj_list[t0].append(i)
            temp_adj_list[t1].append(i)
            temp_adj_list[t2].append(i)
        
        for i in range(len(temp_adj_list)):
            self.max_vert_ind = max(len(temp_adj_list[i]), self.max_vert_ind)
            self.min_vert_ind = min(len(temp_adj_list[i]), self.min_vert_ind)
        
        assert self.min_vert_ind >= 1
        
        for i in range(len(temp_adj_list)):
            cur_adj_num = len(temp_adj_list[i])
            tmp_faces_idx = []
            tmp_weights = []
            for j in range(self.max_vert_ind + 1):
                if j < cur_adj_num:
                    tmp_faces_idx.append(temp_adj_list[i][j])
                    tmp_weights.append(1.0/cur_adj_num)
                else:
                    tmp_faces_idx.append(temp_adj_list[i][-1])
                    tmp_weights.append(0.0)
            
            self.vert_adj_faces.append(tmp_faces_idx)
            self.vert_adj_weights.append(tmp_weights)
        
        self.vert_adj_faces = np.array(self.vert_adj_faces, dtype=np.int32)
        self.vert_adj_weights = np.array(self.vert_adj_weights, dtype=np.float32)

        self.face_idx_cu = torch.LongTensor(self.face_idx).to(self.device)
        self.vert_adj_faces_cu = torch.LongTensor(self.vert_adj_faces).to(self.device)
        self.vert_adj_weights_cu = torch.FloatTensor(self.vert_adj_weights).to(self.device)
        
        return     
    
    def gen_barycentric_coords(self):
        print('+++++ create mesh barycentric related')

        self.uv_face_id, self.uv_bary_weights = gen_uv_barycentric(
            self.face_idx, self.face_texture_coords, resolution=self.barycentric_tex_size, 
        )
        
        self.uv_face_id, self.uv_bary_weights = dilate_barycentric_maps(
            self.uv_face_id, self.uv_bary_weights
        )
        
        self.uv_idx_np, self.uv_idy_np = np.where(self.uv_face_id >= 0)
        
        self.uv_idx_cu = torch.LongTensor(self.uv_idx_np).to(self.device)
        self.uv_idy_cu = torch.LongTensor(self.uv_idy_np).to(self.device)

        self.uv_vert_idx_np = self.face_idx[self.uv_face_id[self.uv_idx_np, self.uv_idy_np],:]
        self.uv_vert_idx_cu = torch.LongTensor(self.uv_vert_idx_np).to(self.device)

        self.uv_bary_weights_np = self.uv_bary_weights[self.uv_idx_np, self.uv_idy_np,:]
        self.uv_bary_weights_cu = torch.FloatTensor(self.uv_bary_weights_np).to(self.device)

        self.uv_non_empty = np.where(self.uv_face_id >= 0)
        self.uv_non_empty_mask = self.uv_face_id >= 0 

        self.gen_uv_map()

        print('+++++ end create mesh barycentric related')
        return 

    def gen_uv_map(self):
        print('+++++ create mesh uv maps')
        self.uv_coord_img = np.zeros([
            self.uv_non_empty_mask.shape[0], 
            self.uv_non_empty_mask.shape[1],
            2
        ])

        # picked texture vaue
        normalized_face_tex = (self.face_texture_coords - 0.5) * 2.0

        w = self.uv_bary_weights[self.uv_non_empty[0], self.uv_non_empty[1],:]

        f_id = self.uv_face_id[self.uv_non_empty[0], self.uv_non_empty[1]]
        
        p = normalized_face_tex[f_id]

        w_p = p[:,0,:] * w[:,0:1] + p[:,1,:] * w[:,1:2] + p[:,2,:] * w[:,2:3]

        self.uv_coord_img[self.uv_non_empty[0], self.uv_non_empty[1],:2] = w_p
        
        print('+++++ end create mesh uv map')
        return 

    def initialize_charactor(self):
        print('+++++ start initializing the character ')
        
        self.charactor = WootCharacter(
            **self.conf['character'],
            device=self.device
        )
                
        print('+++++ end initializing the character ')
        return

    def initialize_spatial_gcn(self):
        print('+++++ start initializing the spatial gcn ')

        self.spatial_gcn = WootSpatialGCN(
            **self.conf['spatial_gcn'],
            obj_reader=self.charactor.graph_obj_reader, device=self.device
        )
        self.spatial_gcn = self.spatial_gcn.to(self.device)

        print('+++++ end initializing the spatial gcn ')
        return 
    
    def initialize_delta_gcn(self):
        print('+++++ start initializing the delta gcn ')
        
        self.delta_gcn = WootSpatialGCN(
            **self.conf['delta_gcn'],
            obj_reader=self.charactor.obj_reader, device=self.device
        )
        self.delta_gcn = self.delta_gcn.to(self.device)
        print('+++++ end initializing the delta gcn ')
        return 
    
    def load_eg_init_checkpoint(self):
        print('+++++ init with egnet checkpoint', self.eg_checkpoint_dir)
        
        if os.path.isfile(self.eg_checkpoint_dir):
            cur_state_dict = torch.load(self.eg_checkpoint_dir, map_location=self.device) 
        
        if (self.spatial_gcn is not None) and ('spatial_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints spatial_gcn from ', self.eg_checkpoint_dir )
                self.spatial_gcn.load_state_dict(cur_state_dict['spatial_gcn'])     
                print('+++++ loading checkpoints spatial_gcn successful')
        else:
            print(self.eg_checkpoint_dir, 'check point not found')

        return 

    def load_delta_init_checkpoint(self):
        print('+++++ init with delta checkpoint', self.delta_checkpoint_dir)
        
        if os.path.isfile(self.delta_checkpoint_dir):
            cur_state_dict = torch.load(self.delta_checkpoint_dir, map_location=self.device)      
            # sdf_network <--- due to the typo
            if (self.spatial_gcn is not None) and ('spatial_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints spatial_gcn')
                self.spatial_gcn.load_state_dict(cur_state_dict['spatial_gcn'])     
            
            if (self.delta_gcn is not None) and ('delta_gcn' in cur_state_dict.keys()):
                print('+++++ loading checkpoints delta_gcn')
                self.delta_gcn.load_state_dict(cur_state_dict['delta_gcn'])             
        else:
            print(self.delta_checkpoint_dir, 'check point not found')
        
        return 
    
    def get_delta_base_template(self, cur_frame_id):

        history_frame_id = np.array([cur_frame_id, max(cur_frame_id - 1, 0), max(cur_frame_id - 2, 0)]) 
        
        anglesNormalized0 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,0,:]).to(self.device)
        anglesNormalized1 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,1,:]).to(self.device)
        anglesNormalized2 = torch.FloatTensor(self.dof_angle_normalized_arr[history_frame_id][:,2,:]).to(self.device)
        
        concated_angles_normalized = torch.cat(
            [anglesNormalized0, anglesNormalized1, anglesNormalized2],  dim = 0
        )

        pose_only_template, picked_r, picked_t = self.charactor.compute_posed_template_embedded_graph(
            dof = concated_angles_normalized
        )

        v0 = pose_only_template[0:3, ...]
        v1 = pose_only_template[3:6, ...]
        v2 = pose_only_template[6:9, ...]

        inputTemporalPoseDeltaNet = torch.concat(
            [v0, v1, v2], dim = -1
        )

        r0 = picked_r[0:3, :, :]
        r1 = picked_r[3:6, :, :]
        r2 = picked_r[6:9, :, :]

        t0 = picked_t[0:3, :, :]
        t1 = picked_t[3:6 :, :]
        t2 = picked_t[6:9, :, :]

        inputTemporalPoseEGNet = torch.concat([
            t0 / 1000., r0, 
            t1 / 1000., r1, 
            t2 / 1000., r2
        ], dim = 2)

        eg_node_RT = self.spatial_gcn(inputTemporalPoseEGNet)
        delta_T = eg_node_RT[:, :, :3] * 1000.
        delta_R = eg_node_RT[:, :, 3:6]

        per_vertex_deformation = self.delta_gcn(inputTemporalPoseDeltaNet)
        per_vertex_deformation = per_vertex_deformation * 1000.0    

        dofs = self.dof_arr[history_frame_id,...]
        dofs = torch.FloatTensor(dofs).to(self.device)
        
        ret_posed_delta, ret_joint_pos = self.charactor.forward_test(
            dof = dofs, delta_R = delta_R, delta_T = delta_T, per_vertex_T = per_vertex_deformation
        )

        return ret_posed_delta, ret_joint_pos[:,:, :3, -1]

    def compute_vert_and_face_normal(self, verts):

        v0 = verts[:,self.face_idx_cu[:,0]]
        v1 = verts[:,self.face_idx_cu[:,1]]
        v2 = verts[:,self.face_idx_cu[:,2]]
        
        v2v1 = v2 - v1
        v0v1 = v0 - v1 

        temp_face_normal = torch.cross(v2v1, v0v1, dim = -1)
        temp_face_normal = F.normalize(temp_face_normal, dim = -1)

        temp_vert_normal = temp_face_normal[:,self.vert_adj_faces_cu]
        temp_vert_weights = self.vert_adj_weights_cu[None,...,None]
        
        fin_vert_normal = torch.sum(temp_vert_normal * temp_vert_weights, dim=2)     
        
        fin_vert_normal = F.normalize(fin_vert_normal, dim = -1)
            
        return fin_vert_normal, temp_face_normal 

    def render_feature_tex(self, feats, resolution):
        feat_dim = feats.shape[-1]
        device = self.device

        # gather and compute weighted features directly
        p = feats[self.uv_vert_idx_cu]  # (N, 3, C)
        w = self.uv_bary_weights_cu     # (N, 3)
        weighted_feats = torch.sum(p * w.unsqueeze(-1), dim=1)  # (N, C)

        # directly scatter into target tensor
        ret_feat = torch.zeros((resolution, resolution, feat_dim), device=device)
        ret_feat.index_put_(
            (self.uv_idx_cu, self.uv_idy_cu),
            weighted_feats,
            accumulate=False
        )

        return ret_feat

    def get_val_mesh_dict(self, frame_id):
        """
           prepare the explicit mesh inputs for the specific frame
        """
        
        current_frame_id = frame_id

        base_frame_id = np.array([
            current_frame_id, current_frame_id, current_frame_id
        ])
        
        history_frame_id = np.array([
            current_frame_id, max(current_frame_id - 1, 0), max(current_frame_id - 2, 0)
        ])

        dofs = self.dof_arr[history_frame_id,...]
    
        with torch.no_grad():
            # 3 * vn * 3, 3 * vj * 3
            ret_posed_delta, skeleton_joints = self.get_delta_base_template(
                current_frame_id
            )

            normalized_trans_with_grad = torch.FloatTensor(
                np.reshape(self.normalization_arr[base_frame_id][:,:3,-1:],[3, 1, 3])
            ).to(self.device)     

            normalized_scale_with_grad = torch.FloatTensor(
                self.normalization_arr[base_frame_id][:,0,0]
            ).to(self.device)

            normalized_ret_posed_delta_with_grad = (ret_posed_delta - normalized_trans_with_grad) / normalized_scale_with_grad

            temp_vert_normal, _ = self.compute_vert_and_face_normal(
                normalized_ret_posed_delta_with_grad
            )

            # normal, pose, speed 0, speed 1
            concat_vert_features = torch.cat(
                [
                    normalized_ret_posed_delta_with_grad[0], normalized_ret_posed_delta_with_grad[1], normalized_ret_posed_delta_with_grad[2],
                    temp_vert_normal[0], temp_vert_normal[1], temp_vert_normal[2]
                ], dim = -1
            )

            pose_feature = self.render_feature_tex(
                feats = concat_vert_features,
                resolution = self.barycentric_tex_size
            )

            normalized_ret_posed_delta_with_grad = normalized_ret_posed_delta_with_grad.detach()
            pose_feature = pose_feature.detach()
            ret_posed_delta = ret_posed_delta.detach()
            skeleton_joints = skeleton_joints.detach().cpu()

        # stay in cuda if further will be used for compuation
        ret_dict = {
            'current_frame_id':current_frame_id,
            'ret_posed_delta': ret_posed_delta[0:1],
            'dofs': dofs,
            'normalized_ret_posed_delta_with_grad': normalized_ret_posed_delta_with_grad,
            'pose_feature': pose_feature,
            'normalized_scale_with_grad': normalized_scale_with_grad,
            'normalized_trans_with_grad': normalized_trans_with_grad
        }

        cur_trans = (self.dof_arr[current_frame_id,:3] - self.base_trans) / self.real_global_scale
        
        ret_dict['global_translation'] = cur_trans
        ret_dict['global_scale'] = 1.2 / self.real_global_scale

        selected_dofs = self.dof_arr[
            [history_frame_id[0], history_frame_id[1], history_frame_id[2]]
        ]  

        if self.hand_dof_mask is not None:
            selected_dofs = selected_dofs[:,self.hand_dof_mask]
        
        selected_dofs[1:3,:3]-= selected_dofs[0:1,:3]
        selected_dofs = selected_dofs.reshape([-1])[3:]
        ret_dict['trans_normalized_dofs'] = selected_dofs

        return ret_dict
    
    def get_val_image_dict(self, ret_dict, camera_id=0):
        """
           prepare the rendering inputs for the specific frame
        """
        
        camera_ids = torch.LongTensor(np.array([[camera_id]]))  

        with torch.no_grad():
            # [b, h, w, 2]=> near and far, [b, h, w]
            depth_image, near_far_mask = renderDepth(
                render_func=self.render_func,
                cameraId=camera_ids,
                objreader=self.obj_reader,
                cameraReader=self.camera_reader,
                meshInstance=ret_dict['ret_posed_delta'],
                jellyOffset=self.depth_offset,
                device=self.device
            )
            
            # foreground pixels for rendering
            non_empty = torch.where(near_far_mask[0])
            pixels_x = non_empty[1]
            pixels_y = non_empty[0]

            depth_image = depth_image / self.global_scale
            near = depth_image[0, non_empty[0], non_empty[1],:1].detach()
            far = depth_image[0, non_empty[0], non_empty[1],-1:].detach()

            p = torch.cat(
                (pixels_x[...,None], pixels_y[...,None], torch.ones_like(pixels_x[...,None])),
                dim=-1
            ).float() 

            world_mat = self.world_camera_arr[camera_id]
            scale_mat = np.eye(4)
            scale_mat[:3,:3] *= self.global_scale

            scale_mat[0,3] = ret_dict['dofs'][0][0] * 1000.
            scale_mat[1,3] = ret_dict['dofs'][0][1] * 1000.
            scale_mat[2,3] = ret_dict['dofs'][0][2] * 1000.

            P = world_mat @ scale_mat
            P = P[:3, :4]

            intrinsics, pose = load_K_Rt_from_P(P)
            intrinsics_inv = np.linalg.inv(intrinsics)

            intrinsics_inv = torch.FloatTensor(intrinsics_inv).to(self.device)
            pose = torch.FloatTensor(pose).to(self.device)
    
            p = (intrinsics_inv[None,:3,:3] @ p[:,:,None])[...,0]
            
            rays_v = p / torch.linalg.norm(p, ord =2, dim = -1, keepdim=True)
            rays_v = (pose[None, :3, :3] @ rays_v[:, :, None])[...,0]
            
            rays_o = pose[None, :3, 3].expand(
                (p.shape[0], -1)
            )
            
            pixel_idh = non_empty[0].detach().cpu()
            pixel_idw = non_empty[1].detach().cpu()

            rays_o = rays_o.detach()
            rays_v = rays_v.detach()
            
            near = near.detach()
            far = far.detach()
            
            if (self.img_scale_factor > 1):
            
                chosen_a = (pixel_idh % self.img_scale_factor == 0)
                chosen_b = (pixel_idw % self.img_scale_factor == 0)

                chosen_xxx = torch.where(torch.logical_and(chosen_a, chosen_b))[0]
                
                rays_o =  rays_o[chosen_xxx]
                rays_v =  rays_v[chosen_xxx]

                near = near[chosen_xxx]
                far = far[chosen_xxx]
                
                pixel_idh = (pixel_idh[chosen_xxx] * (1.0 / self.img_scale_factor)).long()
                pixel_idw = (pixel_idw[chosen_xxx] * (1.0 / self.img_scale_factor)).long()
        
        
        ret_dict['rays_o'] = rays_o
        ret_dict['rays_v'] = rays_v
        ret_dict['near'] = near
        ret_dict['far'] = far
        ret_dict['pixel_idh_np'] = pixel_idh
        ret_dict['pixel_idw_np'] = pixel_idw
        ret_dict['camera_id'] = camera_id
        ret_dict['pose'] = pose[:3, :3]
        
        # if spearated device for data prepare and rendering, we need to move the data to render device
        if self.device != self.render_device:            
            ret_dict['rays_o'] = ret_dict['rays_o'].to(self.render_device,non_blocking=True)
            ret_dict['rays_v'] = ret_dict['rays_v'].to(self.render_device,non_blocking=True)
            ret_dict['near'] = ret_dict['near'].to(self.render_device,non_blocking=True)
            ret_dict['far'] = ret_dict['far'].to(self.render_device,non_blocking=True)
            ret_dict['pose'] = ret_dict['pose'].to(self.render_device,non_blocking=True)
            ret_dict['ret_posed_delta'] = ret_dict['ret_posed_delta'].to(self.render_device,non_blocking=True)
            ret_dict['normalized_ret_posed_delta_with_grad'] = ret_dict['normalized_ret_posed_delta_with_grad'].to(self.render_device,non_blocking=True)
            ret_dict['pose_feature'] = ret_dict['pose_feature'].to(self.render_device,non_blocking=True)
            ret_dict['normalized_scale_with_grad'] = ret_dict['normalized_scale_with_grad'].to(self.render_device,non_blocking=True)
            ret_dict['normalized_trans_with_grad'] = ret_dict['normalized_trans_with_grad'].to(self.render_device,non_blocking=True)  
                    
        return ret_dict

    # get item
    def __getitem__(self, index):
        #current_frame_id = self.chosen_frame_id_list[
        #    (index % len(self.chosen_frame_id_list *))
        #]
        
        index = index % (len(self.chosen_frame_id_list) * len(self.val_camera))
        current_frame_id = self.chosen_frame_id_list[
            index // len(self.val_camera)
        ]
        current_camera_id = self.val_camera[
            index % len(self.val_camera)
        ]
        
        ret_dict = self.get_val_mesh_dict(current_frame_id)
        # update the rendering related
        ret_dict = self.get_val_image_dict(ret_dict, camera_id=current_camera_id)
        
        return ret_dict

    def __len__(self):
        return self.never_stop_size