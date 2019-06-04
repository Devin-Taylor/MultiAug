#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='MultiAug',
    version='0.1.17',
    author='Devin Taylor',
    author_email='dev.t03@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    url='http://pypi.python.org/pypi/MultiAug/',
    license='LICENSE.txt',
    description='Multi-modal data augmentation library for machine learning',
    long_description=open('README.txt').read(),
)
