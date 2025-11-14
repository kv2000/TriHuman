/*
@File: cuda_renderer_fw_kernel.cu
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: The kernel for rasterization, move most of the computation to native pytorch compared with TF ddc.
*/

#include <torch/extension.h>
#include "cuda_renderer_data.h"
#include "cuda_render_utils.h"
#include <iostream>
#include <stdio.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

template <typename scalar_t>
__global__ void renderDepthBufferDevice(
    // f * 3, int
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> faces,
    // cam * face * 2 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_face_bbox_min,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_face_bbox_max,
    // cam * vertices * 3 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_vertices,
    // cam * h * w 
    int* __restrict__ d_depth_buffer,
    const int number_of_cameras,
    const int N, 
    const int F,
    const int h, const int w
){   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for each camera, each triangle 
    if (idx < (number_of_cameras * F) ){

        // => get the camera_id, fid(vid)
        // => get the projected bbox_min, bbox_max
        // => get the (camera conditioned) projected vertices
        // for the pixels in the range 
        // (1) check whether in the triangle
        // (2) compute the depth
        // (3) gpuAtomMin

        int2 index = index1DTo2D(number_of_cameras, F, idx);
        int idc = index.x;
		int idf = index.y;

		int indexv0 = faces[idf][0];
		int indexv1 = faces[idf][1];
		int indexv2 = faces[idf][2];

        float3 i_v0 = make_float3(
            projected_vertices[idc][indexv0][0], projected_vertices[idc][indexv0][1], projected_vertices[idc][indexv0][2]
        );
        float3 i_v1 = make_float3(
            projected_vertices[idc][indexv1][0], projected_vertices[idc][indexv1][1], projected_vertices[idc][indexv1][2]
        );
        float3 i_v2 = make_float3(
            projected_vertices[idc][indexv2][0], projected_vertices[idc][indexv2][1], projected_vertices[idc][indexv2][2]
        );

        // u-x-w, v-y-h
        int bb_u_min, bb_u_max, bb_v_min, bb_v_max; 
        
        bb_u_min = (int) fmaxf(fminf(i_v0.x, fminf(i_v1.x, i_v2.x)) - 0.5f, 0);  //minx - w - u
        bb_v_min = (int) fmaxf(fminf(i_v0.y, fminf(i_v1.y, i_v2.y)) - 0.5f, 0);  //miny - h - v

        bb_u_max = (int) fminf(fmaxf(i_v0.x, fmaxf(i_v1.x, i_v2.x)) + 0.5f, w - 1);  //maxx - w - u
        bb_v_max = (int) fminf(fmaxf(i_v0.y, fmaxf(i_v1.y, i_v2.y)) + 0.5f, h - 1);  //maxy - h - v

        for (int u = bb_u_min; u <= bb_u_max; u++){
            for (int v = bb_v_min; v <= bb_v_max; v++){
                float z = FLT_MAX;
                // - > from my own barycentric py code
                float px, py;
                float p0x, p0y, p1x, p1y, p2x, p2y;
                
                // the pixel to be quried
                px = (float) (u + 0.5);
                py = (float) (v + 0.5);
                // the current triangle, projected, vertices
                p0x = i_v0.x;
                p0y = i_v0.y;
                p1x = i_v1.x;
                p1y = i_v1.y;
                p2x = i_v2.x;
                p2y = i_v2.y;
                
                float signed_area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y);
                bool valid_triangle = true;

                // a valid area, which is the wrong one 
                if (fabs(signed_area) < 1e-9){
                    valid_triangle = false;
                    signed_area = 1e-4;
                }

                float w_1 = 1.f / (2.f * signed_area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py);
                float w_2 = 1.f / (2.f * signed_area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py);
                float w_0 = 1.f - w_1 - w_2;
                
                if (
                    (valid_triangle) && (w_0 >= -0.001f) && (w_1 >= -0.001f) && (w_2 >= -0.001f)
                                     && (w_0 <= 1.001f) && (w_1 <= 1.001f) && (w_2 <= 1.001f)
                ){
                    
                    z = 1.f / ( w_0 / i_v0.z + w_1 / i_v1.z + w_2 / i_v2.z);
                    z *= 10000.f;
                    z = fmaxf(fminf(z, (float)NFLT_MAX),0.f);
                    int pixelId = idc * w * h + w * v + u;
                    
                    atomicMin(d_depth_buffer + pixelId, (int)z);   
                }
            }
        }
    }
}

// the final rendering 
template <typename scalar_t>
__global__ void renderBufferDevice(
    // f * 3, int
    const torch::PackedTensorAccessor<int32_t, 2, torch::RestrictPtrTraits, size_t> faces,
    // cam * face * 2 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_face_bbox_min,
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_face_bbox_max,
    // cam * vertices * 3 
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> projected_vertices,
    // cam * h * w ,
    const torch::PackedTensorAccessor<int32_t, 3, torch::RestrictPtrTraits, size_t> d_depth_buffer,
    int* __restrict__ face_buffer,
    scalar_t* __restrict__ barycentric_buffer,
    const int number_of_cameras,
    const int N, 
    const int F,
    const int h, const int w

){   
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // for each camera, each triangle 
    if (idx < (number_of_cameras * F) ){
        int2 index = index1DTo2D(number_of_cameras, F, idx);
        int idc = index.x;
		int idf = index.y;

		int indexv0 = faces[idf][0];
		int indexv1 = faces[idf][1];
		int indexv2 = faces[idf][2];

        float3 i_v0 = make_float3(
            projected_vertices[idc][indexv0][0], projected_vertices[idc][indexv0][1], projected_vertices[idc][indexv0][2]
        );
        float3 i_v1 = make_float3(
            projected_vertices[idc][indexv1][0], projected_vertices[idc][indexv1][1], projected_vertices[idc][indexv1][2]
        );
        float3 i_v2 = make_float3(
            projected_vertices[idc][indexv2][0], projected_vertices[idc][indexv2][1], projected_vertices[idc][indexv2][2]
        );

        // u-x-w, v-y-h
        int bb_u_min, bb_u_max, bb_v_min, bb_v_max; 
        
        bb_u_min = (int) fmaxf(fminf(i_v0.x, fminf(i_v1.x, i_v2.x)) - 0.5f, 0);  //minx - w - u
        bb_v_min = (int) fmaxf(fminf(i_v0.y, fminf(i_v1.y, i_v2.y)) - 0.5f, 0);  //miny - h - v

        bb_u_max = (int) fminf(fmaxf(i_v0.x, fmaxf(i_v1.x, i_v2.x)) + 0.5f, w - 1);  //maxx - w - u
        bb_v_max = (int) fminf(fmaxf(i_v0.y, fmaxf(i_v1.y, i_v2.y)) + 0.5f, h - 1);  //maxy - h - v

        for (int u = bb_u_min; u <= bb_u_max; u++){
            for (int v = bb_v_min; v <= bb_v_max; v++){
                float z = FLT_MAX;
                float px, py;
                float p0x, p0y, p1x, p1y, p2x, p2y;
                
                // the pixel to be quried
                px = (float) (u + 0.5);
                py = (float) (v + 0.5);
                // the current triangle, projected, vertices
                p0x = i_v0.x;
                p0y = i_v0.y;
                p1x = i_v1.x;
                p1y = i_v1.y;
                p2x = i_v2.x;
                p2y = i_v2.y;
                
                float signed_area = 0.5 * (-p1y * p2x + p0y * (-p1x + p2x) + p0x * (p1y - p2y) + p1x * p2y);
                bool valid_triangle = true;

                // a valid area, which is the wrong one 
                if (fabs(signed_area) < 1e-9){
                    valid_triangle = false;
                    signed_area = 1e-4;
                }

                float w_1 = 1.f / (2.f * signed_area) * (p0y * p2x - p0x * p2y + (p2y - p0y) * px + (p0x - p2x) * py);
                float w_2 = 1.f / (2.f * signed_area) * (p0x * p1y - p0y * p1x + (p0y - p1y) * px + (p1x - p0x) * py);
                float w_0 = 1.f - w_1 - w_2;
                
                if (
                    (valid_triangle) && (w_0 >= -0.001f) && (w_1 >= -0.001f) && (w_2 >= -0.001f)
                                     && (w_0 <= 1.001f) && (w_1 <= 1.001f) && (w_2 <= 1.001f)
                ){
                    // current depth again
                    z = 1.f / ( w_0 / i_v0.z + w_1 / i_v1.z + w_2 / i_v2.z);
                    z *= 10000.f;
                    z = fmaxf(fminf(z, (float)NFLT_MAX),0.f);
                    
                    //int pixelId = idc * w * h + w * v + u;
                    int cur_int_z = (int)z;
                    
                    if(d_depth_buffer[idc][v][u] == cur_int_z){
                        // depth buffer c * h * w
                        // face buffer c * h * w
                        int pixelId = idc * w * h + w * v + u;
                        // barycentic w0, w1, w2
                        int pixelId_bary = 3 * idc * w * h + 3 * w * v + 3 * u;
                        // barycentric buffer, not the set the value
                        face_buffer[pixelId] = idf;
                        barycentric_buffer[pixelId_bary] = w_0;
                        barycentric_buffer[pixelId_bary + 1] = w_1;
                        barycentric_buffer[pixelId_bary + 2] = w_2;
                    }
                }
            }
        }
    }
}

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
){
 
    const int n_blocks =(int)(F * number_of_cameras + THREADS_PER_BLOCK_CUDABASEDRASTERIZER - 1) / THREADS_PER_BLOCK_CUDABASEDRASTERIZER;
    const int n_threads = (int)THREADS_PER_BLOCK_CUDABASEDRASTERIZER;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int h, w;
    h = face_buffer.size(1);
    w = face_buffer.size(2);

    //auto temp_depth_buffer = at::full({number_of_cameras, h, w}, NFLT_MAX, d_depth_buffer.options());

    AT_DISPATCH_FLOATING_TYPES(barycentric_buffer.type(), "renderDepthBufferDevice", ([&] {
        renderDepthBufferDevice <scalar_t> <<< n_blocks, n_threads, 0, stream>>>(
            faces.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),                      // f * 3
            projected_face_bbox_min.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),   // c * f * 2
            projected_face_bbox_max.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),   // c * f * 2
            projected_vertices.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),        // c * n * 3
            d_depth_buffer.data_ptr<int32_t>(),
            number_of_cameras,
            N,
            F,
            h, w
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(barycentric_buffer.type(), "renderBufferDevice", ([&] {
        renderBufferDevice <scalar_t> <<< n_blocks, n_threads, 0, stream>>>(
            faces.packed_accessor<int32_t, 2, torch::RestrictPtrTraits, size_t>(),                      // f * 3
            projected_face_bbox_min.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),   // c * f * 2
            projected_face_bbox_max.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),   // c * f * 2
            projected_vertices.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),        // c * n * 3
            d_depth_buffer.packed_accessor<int32_t, 3, torch::RestrictPtrTraits, size_t>(),             // c * h * w  int    
            face_buffer.data_ptr<int32_t>(),                                                            // c * h * w  int
            barycentric_buffer.data_ptr<scalar_t>(),                                                    // c * h * w * 3  float            
            number_of_cameras,
            N,
            F,
            h, w
        );
    }));

}
