import os, sys
from setuptools import setup, find_packages
# from numpy.distutils.core import setup, Extension

cmd = 'gfortran ./glmnet_python/GLMnet.f -fPIC -fdefault-real-8 -shared -o ./glmnet_python/GLMnet.so'
os.system(cmd)

setup(name='glmnet_python',
      version = '0.2.0',
      description = 'Python version of glmnet, from Stanford University',
      long_description=open('README.md').read(),
      url="https://github.com/bbalasub1/glmnet_python",
      author = 'Trevor Hastie, Balakumar B.J.',
      author_email = 'bbalasub@gmail.com',
      license = 'GPL-2',
      packages=['glmnet_python'],
      install_requires=['joblib>=0.10.3'],
      package_data={'glmnet_python': ['*.so', 'glmnet_python/*.so']},
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: Unix',
        ],
      keywords='glm glmnet ridge lasso elasticnet',
      )
