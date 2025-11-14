#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <stdio.h>
#include <iostream>


#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a Contiguous tensor")
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.is_floating_point(), #x " must be a FLOAT tensor")

#define CHECK_FLOAT_CUDA(x) CHECK_CUDA(x); CHECK_FLOAT(x); CHECK_CONTIGUOUS(x)
#define CHECK_LONG_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#ifndef NFLT_MAX
#define NFLT_MAX 200000000
#endif

__inline__ __device__ int2 index1DTo2D(int size_0, int size_1, int index_1d) {
    int2 index2D;
    
    index2D.y = index_1d % size_1;
    index2D.x = (index_1d - index2D.y) / size_1;
    
    return index2D;
}



