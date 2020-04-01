from setuptools import setup, find_packages
long_description = open('./README.md').read()

setup(
    name='transparent_imshow',
    version='0.1.0',
    description="Pixel-dependent transparency with matplotlib's imshow",
    long_description=long_description,
    maintainer='Remi Lehe',
    maintainer_email='remi.lehe@normalesup.org',
    packages=find_packages('./'),
    install_requires=['matplotlib', 'numpy'] )
