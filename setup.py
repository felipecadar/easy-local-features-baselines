import sys
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

requires = [
    "torch>=1.9.1",
    "torchvision>=0.9.1",
    "opencv-python",
    "numpy",
    "scipy",
    "wget",
    "tqdm",
    "kornia-rs",
    "kornia_moons",
    "omegaconf",
    "tensorflow",
    "tensorflow_hub",
    "yacs",
]

setup(
    name='easy_local_features',
    version='0.4.8',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    # include the requirements file
    package_data={'': ['requirements.txt']},
    install_requires=requires,
    long_description_content_type='text/markdown',
    long_description=long_description,
)
