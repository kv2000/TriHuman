#pragma once

#define THREADS_PER_BLOCK_CUDABASEDRASTERIZER 512

#include <torch/extension.h>
#include <iostream>
#include "cuda_render_utils.h"

struct cuda_renderer_fw_data{
    //==============================================================================================//
    // These are the inputs
    //==============================================================================================//
    // the attributes
    torch::Tensor faces;    
    int number_of_cameras;
    
    int w;
    int h;
    // number of mesh vertices
    int N;
    // number of faces
    int F;
    int number_of_batches;
   
    torch::Tensor projected_face_bbox_min;
    torch::Tensor projected_face_bbox_max;
    torch::Tensor projected_vertices;

    //==============================================================================================//
    //These are the outputs
    //==============================================================================================//
    
    torch::Tensor face_buffer;    
    torch::Tensor barycentric_buffer;
    // b * c * h * w, or => the intermidate result
    torch::Tensor d_depth_buffer;
};