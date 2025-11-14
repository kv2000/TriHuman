"""
@File: setup.py
@Author: Heming Zhu
@Email: hezhu@mpi-inf.mpg.de
@Date: 2023-09-25
@Desc: For building the pytorch cuda rasterizer, move most of the computation to native pytorch compared with TF ddc.
"""

import glob
import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6"

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# if there is more then appen more
include_dirs = [
    os.path.join(ROOT_DIR, "include"),
    os.path.join(ROOT_DIR, "third_party")
]

sources = glob.glob('*.cpp') + glob.glob('*.cu')

setup(
    name='woot_cuda_renderer',
    version='0.1',
    author='hemingzhu',
    description='cuda_renderer',
    long_description='mocked cuda_renderer by heming :D',
    ext_modules=[
        CUDAExtension(
            name='woot_cuda_renderer',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-std=c++17','-O3', '-pthread', '-mavx2', '-mfma','-Wdeprecated-declarations'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
