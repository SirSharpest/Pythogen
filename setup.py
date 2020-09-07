from setuptools import setup

setup(name='Pythogen',
      version='0.1',
      description='Library used for simulating cellular network diffusion',
      url='https://github.com/SirSharpest/pythogen',
      author='Nathan Hughes',
      author_email='nathan.hughes@jic.ac.uk',
      license='MIT',
      packages=['Pythogen'],
      install_requires=['numpy',
                        'matplotlib',
                        'scipy',
                        'networkx',
                        'pandas',
                        'tqdm'],
      zip_safe=True)
