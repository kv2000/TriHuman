"""
@File: depth_renderer.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: the depth renderer like HD:humans https://people.mpi-inf.mpg.de/~mhaberma/projects/2023-hdhumans/
"""

import sys
import os
sys.path.append("../")

import numpy as np
import torch
import kornia as kornia
from einops import rearrange

def getSHCoeff(numBatches, numCams):
    # from tensorf
    shCoeff = np.array([
        2.5033429417967046, -1.7701307697799304, 0.9461746957575601,-0.6690465435572892, 0.10578554691520431, 
        -0.6690465435572892, 0.47308734787878004, -1.7701307697799304,0.6258357354491761, 2.5033429417967046,
        -1.7701307697799304, 0.9461746957575601, -0.6690465435572892,0.10578554691520431,-0.6690465435572892,
        0.47308734787878004,-1.7701307697799304,0.6258357354491761,2.5033429417967046,-1.7701307697799304,
        0.9461746957575601,-0.6690465435572892,0.10578554691520431,-0.6690465435572892,0.47308734787878004,
        -1.7701307697799304,0.6258357354491761
    ]) * 0.3
    shCoeff = shCoeff.reshape([1, 1, 27])
    shCoeff = np.tile(shCoeff,(numBatches , numCams, 1))
    return shCoeff

# multi_batched version :D     
def renderDepth(render_func, cameraId, objreader, cameraReader, meshInstance, jellyOffset, device='cuda:0'):
    
    vC = np.expand_dims(
        np.array(
            objreader.vertexColors
        ), axis=0
    )
    
    if type(meshInstance) is np.ndarray:
        vP = torch.FloatTensor(meshInstance).to(device=device).contiguous()
    else:
        # if not on device, move it
        if not meshInstance.device == device:
            vP = meshInstance.to(device=device).contiguous()
        else:
            vP = meshInstance.contiguous()
            
    vC = torch.FloatTensor(vC).to(device=device).contiguous()

    facesVertexId = torch.LongTensor(objreader.facesVertexId).to(device=device).reshape([-1,3]).contiguous()
    textureCoordinates = torch.FloatTensor(objreader.textureCoordinates).to(device=device).reshape([-1,3,2]).contiguous()
    
    T = torch.FloatTensor(
        np.asarray(objreader.textureMap)
    ).to(device=device).reshape([1, objreader.texHeight, objreader.texWidth, 3]).contiguous()
    
    textureResolutionV, textureResolutionU = T.shape[1], T.shape[2]

    c = cameraId.long()

    camExtrinsics = torch.reshape(
        torch.FloatTensor(cameraReader.extrinsics), [cameraReader.numberOfCameras, 3, 4]
    )[c].contiguous().to(device=device)
    
    camIntrinsics = torch.reshape(
        torch.FloatTensor(cameraReader.intrinsics), [cameraReader.numberOfCameras, 3, 3]
    )[c].contiguous().to(device=device)

    c = c.contiguous().to(device=device)
    SH = torch.FloatTensor(
        getSHCoeff(1, c.shape[1])
    ).to(device=device).contiguous()
    
    albedoMode = 'depth'
    shadingMode = 'shadeless'
    compute_normal_map_attr = False
   
    barycentric_buffer, face_buffer, render_buffer, vertex_normal, normal_map, bb_min, bb_max = render_func(
        faces_attr                  = facesVertexId,
        texCoords_attr              = textureCoordinates,
        numberOfVertices_attr       = vP.shape[1],
        numberOfCameras_attr        = c.shape[1],
        renderResolutionU_attr      = cameraReader.width,
        renderResolutionV_attr      = cameraReader.height,
        albedoMode_attr             = albedoMode,
        shadingMode_attr            = shadingMode,
        image_filter_size_attr      = 1,
        texture_filter_size_attr    = 1,
        compute_normal_map_attr     = compute_normal_map_attr,
        vertexPos_input             = vP,
        vertexColor_input           = vC,
        texture_input               = T,
        shCoeff_input               = SH,
        extrinsics_input            = camExtrinsics,
        intrinsics_input            = camIntrinsics
    )
    # face buffer -> b, c_num, h, w, 1
    # depth       -> b, c_num, h, w, 1
    renderImg_eroded = render_buffer[0,:,:,:,0:1]
    face_buffer = face_buffer[0,:,:,:,None]
    org_mask = (face_buffer >= 0).float()

    renderImg_eroded = renderImg_eroded * org_mask + (1. - org_mask) * 100000.0
    erosion_kernels = torch.ones([9, 9]).to(device)

    renderImg_eroded = rearrange(renderImg_eroded, 'b h w c -> b c h w')
    renderImg_eroded = kornia.morphology.erosion(
        renderImg_eroded, erosion_kernels
    )

    renderImg_eroded = rearrange(renderImg_eroded, 'b c h w -> b h w c')
    mask = (renderImg_eroded < 10000.0).float()
    renderImg_eroded = renderImg_eroded * mask

    renderImg_dilated = render_buffer[0,:,:,:,0:1] * org_mask
    dilation_kernels = torch.ones([9, 9]).to(device)
    renderImg_dilated = rearrange(renderImg_dilated, 'b h w c -> b c h w')
    renderImg_dilated = kornia.morphology.dilation(
        renderImg_dilated, dilation_kernels
    )
    renderImg_dilated = rearrange(renderImg_dilated, 'b c h w -> b h w c')

    near = renderImg_eroded - mask * jellyOffset
    far = renderImg_dilated + mask * jellyOffset

    depth_image = torch.cat(
        [near, far], dim = -1
    )
    return depth_image, mask[:,:,:,0]

