import os, sys
# from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

cmd = 'gfortran ./glmnet_python/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnet_python/GLMnet.so'
os.system(cmd)

setup(name='glmnet_python',
        version = '0.1',
        description = 'Python version of glmnet, originally from Stanford University, modified by Han Fang',
        url="https://github.com/hanfang/glmnet_python",
        author = 'Han Fang',
        author_email = 'hanfang.cshl@gmail.com',
        license = 'GPL-2',
        packages=['glmnet_python'],
        package_data={'glmnet_python': ['*.so', 'glmnet_python/*.so']})


