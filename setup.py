from setuptools import setup, find_packages, Extension
 

setup(name='spkmeansmodule',
      version='0.1.0',
      author='Einav Brosh & Keren Peer',
      description='Spectral Clustering Wrap written in C',

      ext_modules=[
          Extension(
              'spkmeansmodule',
              ['spkmeansmodule.c', 'spkmeans.c']
          ),
      ]
    )
