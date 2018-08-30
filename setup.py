import os, sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess

class CustomInstallCommand(install):
    """Compile shared fortran library"""
    def run(self):
      dir_path = os.path.dirname(os.path.realpath(__file__))
      command = ['gfortran ' + dir_path + '/glmnet_python/GLMnet.f -fPIC -fdefault-real-8 '
                 '-shared -o ' + dir_path + '/glmnet_python/GLMnet.so']
      subprocess.check_call(command, shell=True)

setup(name='glmnet_python_pyseer',
      version = '0.2.2',
      description = 'Python version of glmnet, from Stanford University',
      long_description=open('README.md').read(),
      url="https://github.com/johnlees/glmnet_python",
      author = 'Han Fang (modified by John Lees)',
      author_email = 'hanfang.cshl@gmail.com (john@johnlees.me)',
      license = 'GPL-2',
      packages=['glmnet_python'],
      install_requires=['joblib>=0.10.3'],
      cmdclass={
        'install': CustomInstallCommand,
      },
      classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Operating System :: Unix',
        ],
      keywords='glm glmnet ridge lasso elasticnet'
      )
