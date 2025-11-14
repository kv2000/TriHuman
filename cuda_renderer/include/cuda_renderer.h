#pragma once

#include <iostream>
#include <torch/extension.h>

void cuda_renderer_fw_cu(
    const torch::Tensor& faces,
    const torch::Tensor& projected_face_bbox_min, 
    const torch::Tensor& projected_face_bbox_max,
    const torch::Tensor& projected_vertices,
    int number_of_cameras,
    int N,
    int F,
    const torch::Tensor& face_buffer,
    const torch::Tensor& barycentric_buffer,
    const torch::Tensor& d_depth_buffer
);