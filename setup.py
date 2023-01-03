# setup.py

from setuptools import setup, find_packages

setup(
    name='easy_local_features',
    version='0.1',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    install_requires=[],
)
