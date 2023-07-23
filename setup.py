import sys
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch.cuda as cuda

if sys.version_info[0] < 3:
    with open('README.md') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

# TODO: Check if this is the best way to do this
CUSTOM_OPS = "easy_local_features/submodules/git_aliked/custom_ops/"
EXT_MODULES = []

# if not MACOS
if sys.platform != 'darwin':
    # if cuda
    if cuda.is_available():
        EXT_MODULES=[
            CUDAExtension(
                'easy_local_features.submodules.git_aliked.custom_ops.get_patches', 
                [CUSTOM_OPS+'get_patches_cuda.cpp', CUSTOM_OPS+'get_patches_cuda.cu']
            )
        ]
    else:
        # build cpu version
        EXT_MODULES=[
            CppExtension(
                name='easy_local_features.submodules.git_aliked.custom_ops.get_patches', 
                sources=[CUSTOM_OPS+'get_patches_cpu.cpp']
            )
        ]

setup(
    name='easy_local_features',
    version='0.3.5',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets', 'custom_ops')),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'torch>=1.9.1',
        'torchvision>=0.9.1',
        'thop',
        'opencv-python',
        'wget',
        'tqdm',
    ],
    long_description_content_type='text/markdown',
    long_description=long_description,

    ext_modules=EXT_MODULES,
    cmdclass={"build_ext": BuildExtension} if EXT_MODULES else {},
)
