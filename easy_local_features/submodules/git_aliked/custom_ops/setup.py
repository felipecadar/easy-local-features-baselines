from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

EXT_MODULES = []
# if cuda
if cuda.is_available():
    EXT_MODULES=[
        CUDAExtension(
            'get_patches', 
            ['get_patches_cuda.cpp', 'get_patches_cuda.cu']
        )
    ]
else:
    # build cpu version
    EXT_MODULES=[
        CppExtension(
            name='get_patches', 
            sources=['get_patches_cpu.cpp']
        )
    ]

setup(name='custom_ops', 
      ext_modules=EXT_MODULES, 
      cmdclass={'build_ext': BuildExtension})
