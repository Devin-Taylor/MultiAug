from distutils.core import setup

setup(
    name='MultiAug',
    version='0.1.0',
    author='Devin Taylor',
    author_email='dev.t03@gmail.com',
    packages=['multiaug', 'multiaug.test'],
    url='http://pypi.python.org/pypi/MultiAug/',
    license='LICENSE.txt',
    description='Multi-modal data augmentation library for machine learning',
    long_description=open('README.txt').read(),
#    install_requires=[
#        "Django >= 1.1.1",
#        "caldav == 0.1.4",
#    ],
)
