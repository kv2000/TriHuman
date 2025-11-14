"""
@File: wootGlobalToLocal.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The mapping the oberservation space to the uvd space, all differentialable
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from icecream import ic
from pytorch3d.ops import knn_points
import time

def compute_dot(p1, p2):
    return (p1[..., 0] * p2[..., 0] + p1[..., 1] * p2[..., 1] + p1[..., 2] * p2[..., 2])

def get_edge_point(vertex, edge, proj):
    return vertex + edge * proj[...,None]

def calculate_barycentric(v0, v1, v2, P):

    v2v1 = v2 - v1
    v0v1 = v0 - v1
    Pv1 = P - v1
    v0v2 = v0 - v2
    Pv2 = P - v2

    S = torch.clamp(
        torch.linalg.norm(torch.linalg.cross(v2v1, v0v1), dim=-1), min=1e-9
    )

    S_CBP = torch.linalg.norm(torch.linalg.cross(v2v1, Pv1), dim=-1)
    S_ACP = torch.linalg.norm(torch.linalg.cross(v0v2, Pv2), dim=-1)

    a = S_CBP / S
    b = S_ACP / S
    c = 1.0 - a - b

    return a, b, c

def project_edge(verts, edges, points):
    point_vec = points - verts
    length = compute_dot(edges, edges)
    return compute_dot(point_vec, edges) / torch.clamp_min(length, 1e-9)

def is_not_above(vertex, edge, norm, point):
    edge_norm = torch.linalg.cross(norm, edge)
    ret = (compute_dot(edge_norm, point - vertex)<=0)
    return ret

def compute_dist(vv0, vv1):
    return torch.sum((vv0 - vv1)**2, dim = -1)

def project_plane(vertex, normal, point):
    
    point_vec = point - vertex
    
    unit_normal = F.normalize(
        normal, eps=1e-9, dim=-1
    )
    
    directed_dist = compute_dot(
        point_vec, unit_normal
    )

    return point - unit_normal * directed_dist[...,None], directed_dist

class wootGlobalToLocal(nn.Module):
    def __init__(
            self, vert_num, _faces, _uv_coords, _cand_num = 10, 
            _jelly_offset = 0.020, device = 'cuda:0', max_sample_num = 5760001 * 2
        ):
        super().__init__()
        print('++++++ Init Woot GlobalToLocal: START')
        ic(
            vert_num, _faces.shape, _uv_coords.shape, _cand_num, _jelly_offset
        )
        
        self.device = device
        self.faces = torch.LongTensor(_faces).to(self.device)
        self.faces.requires_grad_ = False
        self.uv_coords = torch.FloatTensor(_uv_coords).to(self.device)
        self.uv_coords.requires_grad_ = False
        
        self.face_normals = None
        self.vert_normals = None
        self.vert_adj_faces = None
        self.vert_adj_weights = None

        self.max_ind = 0
        self.min_ind = 114514
        self.edge_length_eps = 1e-9

        self.max_sample_num = max_sample_num

        self.vert_num = vert_num
        self.face_num = self.faces.shape[0]

        self.cand_num = _cand_num
        self.jelly_offset = _jelly_offset

        self.arange_buffer = torch.arange(0, self.max_sample_num).long().to(self.device)
        self.arange_buffer.requires_grad_ = False
        self.zero_buffer = torch.zeros(self.max_sample_num).float().to(self.device)
        self.zero_buffer.requires_grad_ = False
        self.one_buffer = torch.ones(self.max_sample_num).float().to(self.device)
        self.one_buffer.requires_grad_ = False

        print('vert num :', self.vert_num)
        print('face num :', self.face_num)
        print('jelly offset :', self.jelly_offset)
        print('candidate num :', self.cand_num)

        self.prepare_adjacent_list()

        print('++++++ END Woot GlobalToLocal: START')

    @torch.no_grad()
    def prepare_adjacent_list(self):
        
        self.vert_adj_faces = []
        self.vert_adj_weights = []
        
        temp_adj_list = [[] for i in range(self.vert_num)]

        for i in range(self.faces.shape[0]):
            t0, t1, t2 = self.faces[i][0], self.faces[i][1], self.faces[i][2]
            temp_adj_list[t0].append(i)
            temp_adj_list[t1].append(i)
            temp_adj_list[t2].append(i)

        for i in range(len(temp_adj_list)):
            self.max_ind = max(len(temp_adj_list[i]), self.max_ind)
            self.min_ind = min(len(temp_adj_list[i]), self.min_ind)
          
        assert self.min_ind >= 1

        for i in range(len(temp_adj_list)):
            cur_adj_num = len(temp_adj_list[i])
            tmp_faces_idx = []
            tmp_weights = []
            for j in range(self.max_ind + 1):
                if j < cur_adj_num:
                    tmp_faces_idx.append(temp_adj_list[i][j])
                    tmp_weights.append(1.0/cur_adj_num)
                else:
                    tmp_faces_idx.append(temp_adj_list[i][-1])
                    tmp_weights.append(0.0)
            
            self.vert_adj_faces.append(tmp_faces_idx)
            self.vert_adj_weights.append(tmp_weights)
            
        self.vert_adj_faces = torch.LongTensor(self.vert_adj_faces).to(self.device)
        self.vert_adj_faces.requires_grad_ = False
        self.vert_adj_weights = torch.FloatTensor(self.vert_adj_weights).to(self.device)
        self.vert_adj_weights.requires_grad_ = False
        
    @torch.no_grad()
    def compute_closet_faces(self, t_query_xyz, temp_verts, cand_ids):

        face_vert_idx = self.faces[cand_ids]
        face_vertices = temp_verts[face_vert_idx]

        query_xyz_expanded = t_query_xyz.unsqueeze(1).expand(
            [-1, self.cand_num, -1]
        )

        # n * cand_num * 3
        v0 = face_vertices[:,:,0,:]
        v1 = face_vertices[:,:,1,:]
        v2 = face_vertices[:,:,2,:]

        v0 = v0.reshape([-1,3])
        v1 = v1.reshape([-1,3])
        v2 = v2.reshape([-1,3])
        query_xyz_expanded = query_xyz_expanded.reshape(-1, 3)

        # n , cand_num , 3
        e10 = v1 - v0
        e21 = v2 - v1
        e02 = v0 - v2

        fN = -1.0 * torch.linalg.cross(
            e10, e02
        )
        
        # sample_num * 5 -> flattend
        uab = project_edge(v0, e10, query_xyz_expanded)
        ubc = project_edge(v1, e21, query_xyz_expanded)
        uca = project_edge(v2, e02, query_xyz_expanded)

        # project onto the vertices
        is_type1 = (uca > 1.) & (uab < 0.)
        is_type2 = (uab > 1.) & (ubc < 0.)
        is_type3 = (ubc > 1.) & (uca < 0.)

        # project onto the edge
        is_type4 = (uab >= 0.) & (uab <= 1.) & is_not_above(v0, e10, fN, query_xyz_expanded)
        is_type5 = (ubc >= 0.) & (ubc <= 1.) & is_not_above(v1, e21, fN, query_xyz_expanded) & torch.logical_not(is_type4)
        is_type6 = (uca >= 0.) & (uca <= 1.) & is_not_above(v2, e02, fN, query_xyz_expanded) & torch.logical_not(is_type4) & torch.logical_not(is_type5)
        
        # project onto the faces
        is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

        # calculate six types of distances,  which is distance **2
        # type 0 map to faces
        type_0_pts, type_0_dist_pre = project_plane(v0, fN, query_xyz_expanded)   
        global_ray_dir = torch.sign(type_0_dist_pre)
        type_0_dist = type_0_dist_pre


        # type 1, 2, 3 map to vertices, no direction
        type_1_dist = torch.sqrt(compute_dist(query_xyz_expanded, v0))
        type_2_dist = torch.sqrt(compute_dist(query_xyz_expanded, v1))
        type_3_dist = torch.sqrt(compute_dist(query_xyz_expanded, v2))

        # type 4, 5, 6 map to edges
        type_4_pts = get_edge_point(v0, e10, uab)
        type_5_pts = get_edge_point(v1, e21, ubc)
        type_6_pts = get_edge_point(v2, e02, uca)

        # type 4, 5, 6 dist
        type_4_dist = torch.sqrt(compute_dist(query_xyz_expanded, type_4_pts))
        type_5_dist = torch.sqrt(compute_dist(query_xyz_expanded, type_5_pts))
        type_6_dist = torch.sqrt(compute_dist(query_xyz_expanded, type_6_pts))
        
        #tt4 = time.time()
        merged_distance = type_0_dist * is_type0.float() + type_1_dist * is_type1.float()\
             + type_2_dist * is_type2.float() + type_3_dist * is_type3.float() +\
             + type_4_dist * is_type4.float() + type_5_dist * is_type5.float() +\
             + type_6_dist * is_type6.float()
        
        merged_distance = merged_distance.reshape([-1, self.cand_num])

        min_idx = torch.argmin(torch.abs(merged_distance), dim=-1)

        iden_idx = self.arange_buffer[:min_idx.shape[0]] 
        
        ret = cand_ids[
            iden_idx, min_idx
        ]

        merged_distance = merged_distance[
            iden_idx, min_idx
        ]
            
        return ret, merged_distance

    @torch.no_grad()
    def compute_k_cloest_faces(self, temp_verts, xyz):
        
        temp_v_0 = temp_verts[self.faces[:,0],:]
        temp_v_1 = temp_verts[self.faces[:,1],:]
        temp_v_2 = temp_verts[self.faces[:,2],:]

        mid_pt = (temp_v_0 + temp_v_1 + temp_v_2) / 3.0

        _, cloest_fid , _ = knn_points(
            p1 = xyz.unsqueeze(0),
            p2 = mid_pt.unsqueeze(0),
            K  = self.cand_num
        )
        return cloest_fid[0]

    def compute_mapping_with_gradient(self, query_xyz, temp_verts, cand_ids):
        # the face vertices, for the candidates of each sample
        num_samples = query_xyz.shape[0]

        face_vert_idx = self.faces[cand_ids]
        face_vertices = temp_verts[face_vert_idx]

        # n * 3
        v0 = face_vertices[:,0,:]
        v1 = face_vertices[:,1,:]
        v2 = face_vertices[:,2,:]
        
        # n , 3
        e10 = v1 - v0
        e21 = v2 - v1
        e02 = v0 - v2
        
        fN = -1.0 * torch.linalg.cross(
            e10, e02
        )

        # sample_num * 5 -> flattend
        uab = project_edge(v0, e10, query_xyz)
        ubc = project_edge(v1, e21, query_xyz)
        uca = project_edge(v2, e02, query_xyz)

        # project onto the vertices
        is_type1 = (uca > 1.) & (uab < 0.)
        is_type2 = (uab > 1.) & (ubc < 0.)
        is_type3 = (ubc > 1.) & (uca < 0.)

        # project onto the edge
        is_type4 = (uab >= 0.) & (uab <= 1.) & is_not_above(v0, e10, fN, query_xyz)
        is_type5 = (ubc >= 0.) & (ubc <= 1.) & is_not_above(v1, e21, fN, query_xyz) & torch.logical_not(is_type4)
        is_type6 = (uca >= 0.) & (uca <= 1.) & is_not_above(v2, e02, fN, query_xyz) & torch.logical_not(is_type4) & torch.logical_not(is_type5)

        # project onto the faces
        is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

        # calculate six types of distances,  which is distance **2
        # type 0 map to faces
        type_0_pts, type_0_dist_pre = project_plane(v0, fN, query_xyz)
        global_ray_dir = torch.sign(type_0_dist_pre)
        real_type_0_dist = type_0_dist_pre
        
        # type_1,2,3 dist to the pts
        type_1_dist = compute_dist(query_xyz, v0)
        type_2_dist = compute_dist(query_xyz, v1)
        type_3_dist = compute_dist(query_xyz, v2)

        real_type_1_dist = torch.sqrt(type_1_dist) * global_ray_dir
        real_type_2_dist = torch.sqrt(type_2_dist) * global_ray_dir
        real_type_3_dist = torch.sqrt(type_3_dist) * global_ray_dir

        # type 4, 5, 6 map to edges
        type_4_pts = get_edge_point(v0, e10, uab)
        type_5_pts = get_edge_point(v1, e21, ubc)
        type_6_pts = get_edge_point(v2, e02, uca)

        # type 4, 5, 6 dist to edges
        type_4_dist = compute_dist(query_xyz, type_4_pts)
        type_5_dist = compute_dist(query_xyz, type_5_pts)
        type_6_dist = compute_dist(query_xyz, type_6_pts)

        real_type_4_dist = torch.sqrt(type_4_dist) * global_ray_dir
        real_type_5_dist = torch.sqrt(type_5_dist) * global_ray_dir
        real_type_6_dist = torch.sqrt(type_6_dist) * global_ray_dir
        
        #########################
        #merged distance is here
        #########################
        
        merged_distance = real_type_0_dist * is_type0.float() + real_type_1_dist * is_type1.float()\
             + real_type_2_dist * is_type2.float() + real_type_3_dist * is_type3.float() +\
             + real_type_4_dist * is_type4.float() + real_type_5_dist * is_type5.float() +\
             + real_type_6_dist * is_type6.float()
        
        ##########################
        #merged barycentic is here
        ##########################        
        
        # faces
        a_type_0, b_type_0, c_type_0 = calculate_barycentric(
            v0, v1, v2, type_0_pts
        )

        type_0_barycentric = torch.stack([a_type_0, b_type_0, c_type_0], dim=-1)
        
        # pts
        type_1_barycentric = torch.stack([self.one_buffer[:num_samples], self.zero_buffer[:num_samples], self.zero_buffer[:num_samples]], dim=-1)
        type_2_barycentric = torch.stack([self.zero_buffer[:num_samples], self.one_buffer[:num_samples], self.zero_buffer[:num_samples]], dim=-1)
        type_3_barycentric = torch.stack([self.zero_buffer[:num_samples], self.zero_buffer[:num_samples], self.one_buffer[:num_samples]], dim=-1)

        # edges
        type_4_barycentric = torch.stack([1. - uab,                         uab,                            self.zero_buffer[:num_samples]], dim=-1)
        type_5_barycentric = torch.stack([self.zero_buffer[:num_samples],   1. - ubc,                       ubc             ],               dim=-1)
        type_6_barycentric = torch.stack([uca,                              self.zero_buffer[:num_samples], 1. - uca        ],               dim=-1)        
        
        merged_barycentric = type_0_barycentric * is_type0[...,None].float() + type_1_barycentric * is_type1[...,None].float()\
             + type_2_barycentric * is_type2[...,None].float() + type_3_barycentric * is_type3[...,None].float() +\
             + type_4_barycentric * is_type4[...,None].float() + type_5_barycentric * is_type5[...,None].float() +\
             + type_6_barycentric * is_type6[...,None].float()
        
        ##########################
        #merged type is here
        ##########################

        merged_type = is_type0.long() * int(0) + is_type1.long() * int(1) + is_type2.long() * int(2)\
            + is_type3.long() * int(3) + is_type4.long() * int(4) + is_type5.long() * int(5) + is_type6.long() * int(6)

        ##########################
        #merged projection pos
        ##########################

        merged_proj_pos = type_0_pts * is_type0[...,None].float()\
             + v0 * is_type1[...,None].float()         + v1 * is_type2[...,None].float()         + v2 * is_type3[...,None].float()\
             + type_4_pts * is_type4[...,None].float() + type_5_pts * is_type5[...,None].float() + type_6_pts* is_type6[...,None].float()

        return merged_distance,\
            merged_barycentric[:,0], merged_barycentric[:,1], merged_barycentric[:,2],\
            merged_type, merged_proj_pos, fN
    
    def get_real_fid(self, xyz, temp_verts):

        with torch.no_grad():
            cloest_fid = self.compute_k_cloest_faces(
                temp_verts,
                xyz
            )
            closet_fid = cloest_fid.long()

        with torch.no_grad():
            real_cloest_fid, merged_distance = self.compute_closet_faces(
                t_query_xyz = xyz,
                temp_verts = temp_verts,
                cand_ids = closet_fid
            )
                
        return real_cloest_fid, merged_distance

    def forward_with_real_fid(self, xyz, temp_verts, real_cloest_fid):

        md, ma, mb, mc, m_type, m_proj_pos, m_fN  = self.compute_mapping_with_gradient(
            query_xyz=xyz,
            temp_verts=temp_verts,
            cand_ids=real_cloest_fid
        )

        chosen_face_uv = self.uv_coords[real_cloest_fid]
        fin_uv = ma[..., None] * chosen_face_uv[:,0,:] \
            + mb[..., None] * chosen_face_uv[:,1,:] \
            + mc[..., None] * chosen_face_uv[:,2,:]
        
        fin_uv = (fin_uv - 0.5) * 2.0

        return fin_uv, md, real_cloest_fid, ma, mb, mc, m_type, m_proj_pos, F.normalize(m_fN, dim=-1)

    def forward(self, xyz, temp_verts):

        with torch.no_grad():
            cloest_fid = self.compute_k_cloest_faces(
                temp_verts,
                xyz
            )
            closet_fid = cloest_fid.long()

        with torch.no_grad():
            real_cloest_fid = self.compute_closet_faces(
                t_query_xyz = xyz,
                temp_verts = temp_verts,
                cand_ids = closet_fid
            )
                
        md, ma, mb, mc, m_type, m_proj_pos, m_fN  = self.compute_mapping_with_gradient(
            query_xyz=xyz,
            temp_verts=temp_verts,
            cand_ids=real_cloest_fid
        )

        chosen_face_uv = self.uv_coords[real_cloest_fid]
        fin_uv = ma[..., None] * chosen_face_uv[:,0,:] \
            + mb[..., None] * chosen_face_uv[:,1,:] \
            + mc[..., None] * chosen_face_uv[:,2,:]
        
        fin_uv = (fin_uv - 0.5) * 2.0

        return fin_uv, md, real_cloest_fid, ma, mb, mc, m_type, m_proj_pos, F.normalize(m_fN, dim=-1)