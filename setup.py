from setuptools import setup
from Cython.Build import cythonize
import numpy as np 
setup(name='Pythogen',
      version='0.4c',
      description='Library used for simulating cellular network diffusion',
      url='https://github.com/SirSharpest/pythogen',
      author='Nathan Hughes',
      author_email='nathan.hughes@jic.ac.uk',
      license='MIT',
      packages=['Pythogen'],
      install_requires=[
          'numpy', 'matplotlib', 'scipy', 'networkx', 'pandas', 'tqdm'
      ],
      ext_modules = cythonize("Pythogen/*.pyx"),
      include_dirs=np.get_include(),
      zip_safe=True)
