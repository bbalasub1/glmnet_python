import os
from setuptools import setup, find_packages
from codecs import open

#from os import path
#here = path.abspath(path.dirname(__file__))

cmd = 'gfortran ./glmnet_python_hanfang/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnet_python_hanfang/GLMnet.so'
os.system(cmd)

setup(name='glmnet_python_hanfang',
	version = '0.1',
	description = 'Python version of glmnet, originally from Stanford University, modified by Han Fang',
        url="https://github.com/hanfang/glmnet_python",
	author = 'Han Fang',
	author_email = 'hanfang.cshl@gmail.com',
	license = 'GPL-2',
	packages=['glmnet_python_hanfang'],
	zip_safe = False)

