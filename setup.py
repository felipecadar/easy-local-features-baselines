from setuptools import setup, find_packages

setup(
    name='easy_local_features',
    version='0.1',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    install_requires=[],
)
