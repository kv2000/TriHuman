/*
@File: cuda_renderer.cpp
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The entry for cuda rasterizer,  move most of the computation to native pytorch compared with TF ddc.
*/

#include <torch/extension.h>
#include <iostream>
#include <math.h> 
// cuda render utils
#include "cuda_render_utils.h"

#include "cuda_renderer_data.h"
#include "cuda_renderer.h"

using namespace std;

//==============================================================================================//
// These are for the renderer
//==============================================================================================//

// set input data value
void set_renderer_input_data_value(
    cuda_renderer_fw_data& input,
    torch::Tensor& faces,
    int number_of_vertices,                 
    int number_of_cameras,                
    int render_resolution_u,
    int render_resolution_v,
    torch::Tensor& projected_face_bbox_min,
    torch::Tensor& projected_face_bbox_max,
    torch::Tensor& projected_vertices
){  
    //---MISC---
    input.number_of_batches = projected_face_bbox_min.size(0);
    input.number_of_cameras = number_of_cameras;
    
    // same topology but different vertex input
    input.N = number_of_vertices;
    input.F = faces.size(0);
        
    // output image size
	input.w = render_resolution_u;
	input.h = render_resolution_v;

    input.faces = faces;

    input.projected_face_bbox_min = projected_face_bbox_min;
    input.projected_face_bbox_max = projected_face_bbox_max;
    input.projected_vertices= projected_vertices;
    
    //---OUTPUT---
    // the intialization for the outpute);
    auto option_int_img_no_grad = torch::TensorOptions().dtype(torch::kInt32).layout(torch::kStrided)
        .device(torch::kCUDA).requires_grad(false);
    auto option_float_img_with_grad = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided)
        .device(torch::kCUDA).requires_grad(false);

    // set the output value 
    // barycentric weights, c * h * w * 3 -> storing the weights 
    input.barycentric_buffer = torch::full({input.number_of_batches, input.number_of_cameras, input.h, input.w, 3}, 0., option_float_img_with_grad);
    // face buffer, c * h * w -> storing the faceid
    input.face_buffer = torch::full({input.number_of_batches, input.number_of_cameras, input.h, input.w}, -1, option_int_img_no_grad);
    // face buffer, c * h * w * 3 -> storing the faceid
    input.d_depth_buffer = torch::full({input.number_of_batches, input.number_of_cameras, input.h, input.w}, NFLT_MAX, option_int_img_no_grad);
}

std::vector<torch::Tensor> render_fw(
    torch::Tensor faces,
    int number_of_vertices,
    int number_of_cameras,
    int render_resolution_u,
    int render_resolution_v,
    torch::Tensor projected_face_bbox_min,
    torch::Tensor projected_face_bbox_max,
    torch::Tensor projected_vertices
){
    
    torch::NoGradGuard no_grad;
    // check data
    CHECK_LONG_CUDA(faces);
    CHECK_FLOAT_CUDA(projected_face_bbox_min);
    CHECK_FLOAT_CUDA(projected_face_bbox_max);
    CHECK_FLOAT_CUDA(projected_vertices);
    
    cuda_renderer_fw_data input_data;

    torch::Tensor _faces = faces.to(torch::kInt32);

    set_renderer_input_data_value(
        input_data, _faces,
        number_of_vertices, number_of_cameras, 
        render_resolution_u, render_resolution_v,
        projected_face_bbox_min, projected_face_bbox_max,
        projected_vertices
    );

    for (int bid = 0; bid < input_data.number_of_batches; bid++){
        cuda_renderer_fw_cu(
            input_data.faces,
            input_data.projected_face_bbox_min[bid],
            input_data.projected_face_bbox_max[bid],
            input_data.projected_vertices[bid],
            input_data.number_of_cameras,
            input_data.N, 
            input_data.F,
            input_data.face_buffer[bid],
            input_data.barycentric_buffer[bid],
            input_data.d_depth_buffer[bid]
        );
    }

    return {
        input_data.face_buffer,
        input_data.barycentric_buffer
    };
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_fw", &render_fw);
}