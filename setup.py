from setuptools import setup, find_packages

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

setup(
    name='easy_local_features',
    version='0.2',
    author='eucadar',
    author_email='python@eucadar.com',
    packages=find_packages(exclude=('tests', 'docs', 'assets')),
    include_package_data=True,
    install_requires=[],
    long_description=long_description,
)
