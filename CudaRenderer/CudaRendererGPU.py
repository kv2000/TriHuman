
"""
@File: CudaRendererGPU.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The cuda renderer wrapper in pytorch, compared with the original TF version in the ddc, all the data structures and most operations have migrated native pytorch, major change. 
"""
########################################################################################################################
# Imports
########################################################################################################################
import sys
sys.path.append("../")
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import cv2 as cv
import numpy as np
from icecream import ic
import trimesh
from PIL import Image
import woot_cuda_renderer
import math

########################################################################################################################
# Render Wrapper, which contains/preprocessed all the data needed
########################################################################################################################

# a larger wrapper, no parameters here
class CudaRendererGpu(nn.Module):
    def __init__(self, faces_attr, texCoords_attr, textureResolutionV, textureResolutionU):
        super(CudaRendererGpu, self).__init__()
        ic('++++++ init cuda renderer gpu')
        self.face_attr = faces_attr
        self.texCoords_attr = texCoords_attr
        self.textureResolutionV = textureResolutionV
        self.textureResolutionU = textureResolutionU
        
        self.uv_face_id = None
        self.uv_barycentircs = None
        
        self.vert_adj_faces = None
        self.vert_adj_weights = None
        self.vert_num = -1

        self.max_ind = 0
        self.min_ind = 114514

        assert textureResolutionV == textureResolutionU, "Cuda Renderer: Texutre should be a square!"

        self.gen_adjacent_list(faces_attr)

        self.gen_uv_barycentric(
            self.face_attr, self.texCoords_attr, self.textureResolutionV
        )

        ic('++++++ end init cuda renderer gpu')
        
    def gen_uv_barycentric(self, _face_idx, _face_texture_coords, resolution=512):
        
        face_idx = _face_idx.cpu().numpy()
        face_texture_coords = _face_texture_coords.cpu().numpy()

        self.uv_face_id = np.ones(shape=(resolution,resolution)) * (-1.0)
        self.uv_barycentircs = np.zeros(shape=(resolution,resolution,3))

        for i in range(face_idx.shape[0]):
            cur_face_uv_coords = face_texture_coords[i]
            uu_min = np.clip(np.min(cur_face_uv_coords[:,0]) * resolution - 2, 0, resolution - 1)
            uu_max = np.clip(np.max(cur_face_uv_coords[:,0]) * resolution + 2, 0, resolution - 1)
            vv_min = np.clip(np.min(cur_face_uv_coords[:,1]) * resolution - 2, 0, resolution - 1)
            vv_max = np.clip(np.max(cur_face_uv_coords[:,1]) * resolution + 2, 0, resolution - 1)
            uu_min, uu_max, vv_min, vv_max = int(uu_min), int(uu_max), int(vv_min), int(vv_max)

            # uu height, vv weigth
            for xx in range(uu_min, uu_max + 1):
                for yy in range(vv_min, vv_max + 1):
                    fin_x, fin_y = xx, yy
                    if self.uv_face_id[fin_x,fin_y] == -1:            
                        px, py = (xx)/resolution, (yy)/resolution     
                        p0x, p0y = cur_face_uv_coords[0, 0], cur_face_uv_coords[0, 1]
                        p1x, p1y = cur_face_uv_coords[1, 0], cur_face_uv_coords[1, 1]
                        p2x, p2y = cur_face_uv_coords[2, 0], cur_face_uv_coords[2, 1]
                        
                        signed_area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y)
                        
                        w_1 = 1 / (2 * signed_area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py)
                        w_2 = 1 / (2 * signed_area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py)
                        w_0 = 1 - w_1 - w_2
                        
                        if (w_0 >= 0) and (w_1 >= 0) and (w_2 >= 0):
                            self.uv_face_id[fin_x, fin_y] = i
                            self.uv_barycentircs[fin_x, fin_y, 0] = w_0
                            self.uv_barycentircs[fin_x, fin_y, 1] = w_1
                            self.uv_barycentircs[fin_x, fin_y, 2] = w_2
               
        self.uv_face_id = torch.LongTensor(self.uv_face_id).to(_face_idx.device).contiguous()
        self.uv_barycentircs = torch.FloatTensor(self.uv_barycentircs).to(_face_idx.device).contiguous()
        
        return 

    def gen_adjacent_list(self, _face_idx):
        face_idx = _face_idx.cpu().numpy()
        num_pts = np.max(face_idx) + 1
        
        self.vert_num = num_pts
        self.vert_adj_faces = []
        self.vert_adj_weights = []
        self.max_ind = 0
        self.min_ind = 114514

        temp_adj_list = [[] for i in range(self.vert_num)]

        for i in range(face_idx.shape[0]):
            t0, t1, t2 = face_idx[i][0], face_idx[i][1], face_idx[i][2]
            temp_adj_list[t0].append(i)
            temp_adj_list[t1].append(i)
            temp_adj_list[t2].append(i)
        
        for i in range(len(temp_adj_list)):
            self.max_ind = max(len(temp_adj_list[i]), self.max_ind)
            self.min_ind = min(len(temp_adj_list[i]), self.min_ind)

        assert self.min_ind >= 1, "ioslated surface on the surface"
        
        for i in range(len(temp_adj_list)):
            cur_adj_num = len(temp_adj_list[i])
            tmp_faces_idx = []
            tmp_weights = []
            for j in range(self.max_ind):
                if j < cur_adj_num:
                    tmp_faces_idx.append(temp_adj_list[i][j])
                    tmp_weights.append(1.0/cur_adj_num)
                else:
                    tmp_faces_idx.append(temp_adj_list[i][-1])
                    tmp_weights.append(0.0)
            
            self.vert_adj_faces.append(tmp_faces_idx)
            self.vert_adj_weights.append(tmp_weights)
            
        self.vert_adj_faces = torch.LongTensor(self.vert_adj_faces).contiguous().to(_face_idx.device)
        self.vert_adj_weights = torch.FloatTensor(self.vert_adj_weights).contiguous().to(_face_idx.device)
    
        return  

    def forward(
            self, 
            faces_attr                 = [],
            texCoords_attr             = [],
            numberOfVertices_attr      = -1,
            numberOfCameras_attr       = -1,
            renderResolutionU_attr     = -1,
            renderResolutionV_attr     = -1,
            albedoMode_attr            = 'textured',
            shadingMode_attr           = 'shaded',
            image_filter_size_attr     = 1,
            texture_filter_size_attr   = 1,
            compute_normal_map_attr    = False,
            vertexPos_input            = None,
            vertexColor_input          = None,
            texture_input              = None,
            shCoeff_input              = None,
            extrinsics_input           = [],
            intrinsics_input           = []
        ):

        if (self.uv_barycentircs is None) or (self.uv_face_id is None):
            ic('------ CudaRendererGpu: please pre-compute the barycentric first')
            sys.exit(0)

        return CudaRendererGpuCppWrapper.apply(        
            faces_attr,
            texCoords_attr,
            numberOfVertices_attr,
            numberOfCameras_attr,
            renderResolutionU_attr,
            renderResolutionV_attr,
            albedoMode_attr,
            shadingMode_attr,
            image_filter_size_attr,
            texture_filter_size_attr,
            compute_normal_map_attr,
            vertexPos_input,
            vertexColor_input,
            texture_input,
            shCoeff_input,
            extrinsics_input,
            intrinsics_input,
            self.uv_face_id,
            self.uv_barycentircs,
            self.vert_adj_faces,
            self.vert_adj_weights
        )

########################################################################################################################
# Render Wrapper, which contains/preprocessed all the data needed
########################################################################################################################

# to compute the projection to the plane
def get_projected_vertices(vertex_pos, intrinsics, extrinsics):
    
    batch_size, num_camera = extrinsics.shape[0], extrinsics.shape[1]
    
    to_pad = torch.ones(
        [vertex_pos.shape[0],vertex_pos.shape[1],1]
    ).contiguous().float().to(vertex_pos.device)

    homo_verts = torch.concat(
        [vertex_pos, to_pad], dim=-1
    ).unsqueeze(1).expand(-1, num_camera, -1, -1)

    c_v0 = (homo_verts.unsqueeze(-2) * extrinsics.unsqueeze(-3)).sum(-1)
    i_v0 = (c_v0.unsqueeze(-2) * intrinsics.unsqueeze(-3)).sum(-1)

    i_v0[:,:,:,:2] /= i_v0[:,:,:,2:]
    
    bb_min, _ = torch.min(i_v0[...,:2], dim = -2)
    bb_max, _ = torch.max(i_v0[...,:2], dim = -2)

    return i_v0, bb_min[...,[1,0]], bb_max[...,[1,0]]

# to compute bounding boxes
def get_projected_face_bbox(proj_vertex_pos, face_idx):
    
    face_num = face_idx.shape[0]
    batch_size, camera_num, vert_num = proj_vertex_pos.shape[0], proj_vertex_pos.shape[1], proj_vertex_pos.shape[2]

    face_vertices = proj_vertex_pos[:,:,face_idx.view(-1),:]
    face_vertices = face_vertices.reshape([batch_size, camera_num, face_num, 3, 3])

    face_vertices_min = torch.min(face_vertices, dim=-1)[0][...,:2]
    face_vertices_max = torch.max(face_vertices, dim=-1)[0][...,:2]

    face_vertices_min = face_vertices_min.float().contiguous()
    face_vertices_max = face_vertices_max.float().contiguous()

    return face_vertices_min, face_vertices_max

def get_face_normal(vertex_pos, face_idx):
    
    batch_size= vertex_pos.shape[0]
    face_num = face_idx.shape[0]
    
    face_vertices = vertex_pos[:,face_idx.view(-1),:]
    face_vertices = face_vertices.reshape([batch_size, face_num, 3, 3])

    v0 = face_vertices[:, :, 0, :]
    v1 = face_vertices[:, :, 1, :]
    v2 = face_vertices[:, :, 2, :]

    face_normal = torch.cross(v1 - v0, v2 - v0, dim=-1)

    return face_normal

def get_vertex_normal(face_normal, vert_adj_faces, vert_adj_weights):
    
    batch_size = face_normal.shape[0]
    vert_num = vert_adj_faces.shape[0]
    adj_num = vert_adj_faces.shape[1]

    vert_adj_normal = face_normal[:,vert_adj_faces.view(-1),:].reshape([batch_size,vert_num,adj_num,3])
    reshaped_vert_adj_weights = vert_adj_weights.unsqueeze(0).expand([batch_size, -1, -1]).unsqueeze(-1)
    
    weighted_sum_normal = torch.sum(vert_adj_normal * reshaped_vert_adj_weights, dim=2)

    return weighted_sum_normal

def get_normal_map_texture(face_idx, vertex_normal, uv_face_id, uv_barycentrics):

    batch_size = vertex_normal.shape[0]
    renderResolutionV_attr, renderResolutionU_attr = uv_face_id.shape[0], uv_face_id.shape[1]

    output_img = torch.zeros(
        [batch_size, renderResolutionV_attr, renderResolutionU_attr, 3]
    ).float().cuda().contiguous()

    # the idx on the image space
    non_zero_idx = torch.where(uv_face_id >= 0)
    # the non_zero faces
    non_zero_fid = uv_face_id[non_zero_idx[0], non_zero_idx[1]]
    non_zero_barycentrics = uv_barycentrics[non_zero_idx[0], non_zero_idx[1]].unsqueeze(-1).unsqueeze(0).expand(
        [batch_size, -1, -1, -1]
    )
    
    non_zero_face_num = non_zero_fid.shape[0]
    non_zero_vid = face_idx[non_zero_fid]
    non_zero_normal = vertex_normal[:,non_zero_vid.view(-1),:].reshape([batch_size, non_zero_face_num, 3, 3])

    weighted_normal = torch.sum(non_zero_normal * non_zero_barycentrics, dim =-2)

    # restore
    output_img[:,non_zero_idx[0], non_zero_idx[1],:] = weighted_normal
    return output_img

########################################################################################################################
# Rendering functions , stuff after rasterization
########################################################################################################################

def render_uv_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, texture_coords):

    chosen_texture_coords = texture_coords[non_zero_fidx]
    
    # barycentric interpolated
    blended_texture_coords = torch.sum(
        chosen_texture_coords * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )

    blended_texture_coords = torch.clamp(
        blended_texture_coords, 0, 1
    )
    
    ret_render_buffer = torch.zeros(render_shape).float().to(blended_texture_coords.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :2
    ] = blended_texture_coords

    return ret_render_buffer

def render_normal_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, faces, rendered_vertex_normal):
    
    chosen_face_vert_idx = faces[non_zero_fidx]
    chosen_face_vert_num = chosen_face_vert_idx.shape[0]
    
    # need a repeat on the face normals
    non_zero_batch_idx = non_zero_idx[0].unsqueeze(-1).expand([-1,3])
    
    chosen_normal = rendered_vertex_normal[
        non_zero_batch_idx.reshape([-1]), chosen_face_vert_idx.view([-1]), :
    ].reshape([chosen_face_vert_num, 3, 3])
    
    blended_normal = torch.sum(
        chosen_normal * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )

    blended_normal = F.normalize(blended_normal, dim=-1)
    ret_render_buffer = torch.zeros(render_shape).float().to(blended_normal.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = blended_normal
    
    return ret_render_buffer

def render_foreground_images(render_shape, non_zero_idx, non_zero_fidx):
        
    ret_render_buffer = torch.zeros(render_shape).float().to(non_zero_fidx.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = 1.0

    return ret_render_buffer

def render_light_images(render_shape, non_zero_idx, non_zero_fidx):
        
    ret_render_buffer = torch.zeros(render_shape).float().to(non_zero_fidx.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = 1.0

    return ret_render_buffer

def render_position_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, faces, vertex_color):
    
    chosen_face_vert_idx = faces[non_zero_fidx]
    chosen_face_vert_num = chosen_face_vert_idx.shape[0]
    
    # need a repeat on the face normals
    non_zero_batch_idx = non_zero_idx[0].unsqueeze(-1).expand([-1,3])
    
    chosen_vertex_color = vertex_color[
        non_zero_batch_idx.reshape([-1]), chosen_face_vert_idx.view([-1]), :
    ].reshape([chosen_face_vert_num, 3, 3])
    
    blended_vertex_color = torch.sum(
        chosen_vertex_color * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )
    
    ret_render_buffer = torch.zeros(render_shape).float().to(blended_vertex_color.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = blended_vertex_color
    
    return ret_render_buffer

def render_depth_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, faces, vertex_color, inverse_extrinsics):
    
    chosen_face_vert_idx = faces[non_zero_fidx]
    chosen_face_vert_num = chosen_face_vert_idx.shape[0]
    
    # need a repeat on the face normals
    non_zero_batch_idx = non_zero_idx[0].unsqueeze(-1).expand([-1,3])
    
    chosen_vertex_color = vertex_color[
        non_zero_batch_idx.reshape([-1]), chosen_face_vert_idx.view([-1]), :
    ].reshape([chosen_face_vert_num, 3, 3])
    
    # this is the interpolated position
    blended_vertex_color = torch.sum(
        chosen_vertex_color * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )

    cam_o = inverse_extrinsics[:,:,:,-1].clone()
    cam_o = cam_o / cam_o[:,:,-1:]
    cam_origin = cam_o[:,:,:3]

    chosen_cam_origin = cam_origin[non_zero_idx[0], non_zero_idx[1],:]
    d = torch.linalg.norm(chosen_cam_origin - blended_vertex_color, dim=-1, keepdim=True)

    ret_render_buffer = torch.zeros(render_shape).float().to(blended_vertex_color.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = d.expand([-1,3])
    
    return ret_render_buffer

def render_vertex_color_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, faces, vertex_color):
    chosen_face_vert_idx = faces[non_zero_fidx]
    chosen_face_vert_num = chosen_face_vert_idx.shape[0]
    
    # need a repeat on the face normals
    non_zero_batch_idx = non_zero_idx[0].unsqueeze(-1).expand([-1,3])
    
    chosen_vertex_color = vertex_color[
        non_zero_batch_idx.reshape([-1]), chosen_face_vert_idx.view([-1]), :
    ].reshape([chosen_face_vert_num, 3, 3])
    
    blended_vertex_color = torch.sum(
        chosen_vertex_color * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )
    
    ret_render_buffer = torch.zeros(render_shape).float().to(blended_vertex_color.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = blended_vertex_color
    
    return ret_render_buffer

def render_textured_images(render_shape, non_zero_idx, non_zero_fidx, non_zero_barycentrics, texture_coords, texture_images):    
    # b h w 3
    tex_height, tex_width = texture_images.shape[-2], texture_images.shape[-3]

    chosen_texture_coords = texture_coords[non_zero_fidx]
    
    # barycentric interpolated
    blended_texture_coords = torch.sum(
        chosen_texture_coords * non_zero_barycentrics.unsqueeze(-1), dim = -2
    )
    blended_texture_coords = torch.clamp(
        blended_texture_coords, 0, 1
    )

    blended_texture_coords_id = torch.zeros_like(blended_texture_coords).contiguous().to(blended_texture_coords.device)
    
    # in the texture space, x y -> h w
    blended_texture_coords_id[:,0] = (1 - blended_texture_coords[:,1]) * tex_height
    blended_texture_coords_id[:,1] = blended_texture_coords[:,0] * tex_width

    blended_texture_coords_id[:,0] = torch.clamp(blended_texture_coords_id[:,0], 0, tex_height - 1)
    blended_texture_coords_id[:,1] = torch.clamp(blended_texture_coords_id[:,1], 0, tex_width - 1)

    blended_texture_coords_id = blended_texture_coords_id.long()

    color_lulv = texture_images[
        non_zero_idx[0], blended_texture_coords_id[:,0], blended_texture_coords_id[:,1], :
    ]

    ret_render_buffer = torch.zeros(render_shape).float().to(blended_texture_coords.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = color_lulv

    return ret_render_buffer

def render_shaded_images(render_shape, non_zero_idx, rendered_normal, sh_coeffs):
    n = rendered_normal[non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :]
    chosen_coeff = sh_coeffs[non_zero_idx[0], non_zero_idx[1],:]
    # ;D like this code from face3D wwww
    nver = n.shape[0]
    iden = torch.ones([nver]).float().contiguous().to(rendered_normal.device)
    sh = torch.stack([ 
        iden, 
        n[:,0], n[:,1], n[:,2], 
        n[:,0] *n[:,1], n[:,0] *n[:,2], n[:,1] *n[:,2],
        n[:,0]**2 - n[:,1]**2,
        3*(n[:,2]**2) - 1 
    ], dim=-1)

    chosen_coeff = chosen_coeff.view([chosen_coeff.shape[0], 3, 9])
    
    # n * 9 - > n * 3 * 9
    blended_color = torch.sum(sh.unsqueeze(-2) * chosen_coeff, dim = -1)
    ret_render_buffer = torch.zeros(render_shape).float().to(blended_color.device)
    
    ret_render_buffer[
        non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
    ] = blended_color

    return ret_render_buffer

########################################################################################################################
# Cuda Function Wrapper, which produce the results and the gradient with cuda code
########################################################################################################################

# wrapper for the cpp
class CudaRendererGpuCppWrapper(Function):
    @staticmethod
    def forward(
        ctx,             
        faces_attr,
        texCoords_attr,
        numberOfVertices_attr,
        numberOfCameras_attr,
        renderResolutionU_attr,
        renderResolutionV_attr,
        albedoMode_attr,
        shadingMode_attr,
        image_filter_size_attr,
        texture_filter_size_attr,
        compute_normal_map_attr,
        vertexPos_input,
        vertexColor_input,
        texture_input,
        shCoeff_input,
        #targetImage_input,
        extrinsics_input,
        intrinsics_input,
        uv_face_id, # the texture space
        uv_barycentircs, # the texture space,
        vert_adj_faces,
        vert_adj_weights
    ):
        # the output is render function forward it will return the rgb
        # these are attributes without the gradients
        batch_size = vertexPos_input.shape[0]
        texture_h, texture_w = uv_face_id.shape[0], uv_face_id.shape[1]
        
        projected_vertices, bb_min, bb_max = get_projected_vertices(
            vertex_pos=vertexPos_input,
            intrinsics=intrinsics_input,
            extrinsics=extrinsics_input
        )

        extrinsics_inv = torch.zeros(
            [batch_size, numberOfCameras_attr, 4, 4]
        ).float().to(projected_vertices.device)

        extrinsics_inv[:,:,:3,:4] = extrinsics_input
        extrinsics_inv[:,:,3,3] = 1.

        extrinsics_inv = torch.linalg.inv(
            extrinsics_inv.view(-1, 4, 4)
        ).reshape([batch_size, numberOfCameras_attr, 4, 4])

        project_face_bbox_min, project_face_bbox_max = get_projected_face_bbox(
            proj_vertex_pos = projected_vertices, face_idx=faces_attr
        )

        rendered_face_normal = get_face_normal(
            vertex_pos=vertexPos_input, face_idx=faces_attr
        )

        rendered_vertex_normal = get_vertex_normal(
            face_normal=rendered_face_normal, vert_adj_faces=vert_adj_faces, vert_adj_weights=vert_adj_weights
        )

        # do the normalization
        rendered_face_normal = F.normalize(rendered_face_normal, dim=-1)
        rendered_vertex_normal = F.normalize(rendered_vertex_normal, dim=-1)

        if compute_normal_map_attr == True:           
            
            normal_map = get_normal_map_texture(
                faces_attr, rendered_vertex_normal, uv_face_id=uv_face_id, uv_barycentrics=uv_barycentircs
            )
            # here are the placeholder 
            barycentric_buffer = torch.zeros(
                [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3]
            ).float().cuda().contiguous()
            
            face_buffer = torch.zeros(
                [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr]
            ).long().cuda().contiguous()
            
            render_buffer = torch.zeros(
                [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3]
            ).float().cuda().contiguous()                
        
        else:
            # empty normal map
            normal_map = torch.zeros(
                [batch_size, texture_h, texture_w, 3]
            ).float().to(projected_vertices.device)

            #print('projectedvertices', faces_attr.device, projected_vertices.device, project_face_bbox_min.device,project_face_bbox_max.device)
            face_buffer, barycentric_buffer = woot_cuda_renderer.render_fw(
                faces_attr,
                numberOfVertices_attr, numberOfCameras_attr, renderResolutionU_attr, renderResolutionV_attr,
                project_face_bbox_min,
                project_face_bbox_max,
                projected_vertices
            )

            # pre-compute the wanted regions
            non_zero_idx = torch.where(face_buffer >= 0)
            # barycentric buffer: b * c * h * w * 3 = > nn * 3
            non_zero_barycentrics = barycentric_buffer[
                non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3], :
            ]
            # face: b * c * h * w = > nn
            non_zero_fidx = face_buffer[
                non_zero_idx[0], non_zero_idx[1], non_zero_idx[2], non_zero_idx[3]
            ].long()

            if albedoMode_attr == 'uv':
                render_buffer = render_uv_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    texture_coords = texCoords_attr
                )
            elif albedoMode_attr == 'foreground_mask':
                render_buffer = render_foreground_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx
                )
            elif albedoMode_attr == 'light':
                render_buffer = render_light_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx
                )
            elif albedoMode_attr == 'position':
                render_buffer = render_position_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    faces = faces_attr, vertex_color=vertexPos_input
                )
            elif albedoMode_attr == 'depth':
                render_buffer = render_depth_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    faces = faces_attr, vertex_color=vertexPos_input, inverse_extrinsics = extrinsics_inv
                )
            elif albedoMode_attr == 'normal':
                render_buffer = render_normal_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    faces = faces_attr, rendered_vertex_normal=rendered_vertex_normal
                )
            elif albedoMode_attr == 'vertex_color':
                render_buffer = render_vertex_color_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    faces = faces_attr, vertex_color=vertexColor_input
                )
            elif albedoMode_attr == 'textured':
                render_buffer = render_textured_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    texture_coords = texCoords_attr, texture_images= texture_input
                )
            elif albedoMode_attr == 'rasterization' :
                render_buffer = None
            else:
                print('stop entering bullxxxx')
                sys.exit(0)

            if((shadingMode_attr == 'shaded') and (not albedoMode_attr == 'normal')):
                rendered_normal_image = render_normal_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, non_zero_fidx = non_zero_fidx, non_zero_barycentrics = non_zero_barycentrics,
                    faces = faces_attr, rendered_vertex_normal=rendered_vertex_normal
                )
                render_buffer *= render_shaded_images(
                    render_shape = [batch_size, numberOfCameras_attr, renderResolutionV_attr, renderResolutionU_attr, 3],
                    non_zero_idx = non_zero_idx, 
                    rendered_normal = rendered_normal_image,
                    sh_coeffs = shCoeff_input
                )

        return barycentric_buffer, face_buffer, render_buffer, rendered_vertex_normal, normal_map, bb_min, bb_max
    
    # backward function not implemented, only forward is needed for this project :D
    @staticmethod
    def backward(ctx, barycentric_buffer_grad, face_buffer_grad, render_buffer_grad, vertex_normal_grad, normal_map_grad):     
        return None, None, None, None, None, None, None, None, None, None, None, \
            None, None, \
            None, None, \
            None, None, None, None, \
            None, None       
