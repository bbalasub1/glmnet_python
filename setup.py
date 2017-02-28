import os
from setuptools import setup, find_packages

cmd = 'gfortran ./glmnet_python/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnet_python/GLMnet.so'
os.system(cmd)

setup(name='glmnet_python_hf',
	version = '0.1',
	description = 'Python version of glmnet, originally from Stanford University, modified by Han Fang',
        url="https://github.com/hanfang/glmnet_python",
	author = 'Han Fang',
	author_email = 'hanfang.cshl@gmail.com',
	license = 'GPL-2',
	packages=['glmnet_python'],
	zip_safe = False)
