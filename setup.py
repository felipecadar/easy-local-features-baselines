import sys
from setuptools import setup, find_packages

# try:
#     import pypandoc
#     long_description = pypandoc.convert_file('README.md', 'rst')
# except(IOError, ImportError):
#     long_description = open('README.md').read()

if sys.version_info[0] < 3:
    with open('README.md') as f:
        long_description = f.read()
else:
    with open('README.md', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='easy_local_features',
    version='0.3.3',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'torch>=1.9.1',
        'torchvision>=0.9.1',
        'opencv-python',
        'wget',
        'tqdm',
    ],
    long_description_content_type='text/markdown',
    long_description=long_description,
)
