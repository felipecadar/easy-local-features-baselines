import sys
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

requires = []
with open('requirements.txt') as f:
    requires = f.read().splitlines()

setup(
    name='easy_local_features',
    version='0.4.0',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    install_requires=requires,
    long_description_content_type='text/markdown',
    long_description=long_description,
)
