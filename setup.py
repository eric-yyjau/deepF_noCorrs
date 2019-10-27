#nsml: nsml/tensorflow:1.8.0-oepncv3.4.1-gpu-py3.5

from distutils.core import setup
setup(
   name='nsml-kitti',
   version='1.0',
   description='',
   install_requires = ['pykitti','progressbar2','vapory','opencv-contrib-python']
)
