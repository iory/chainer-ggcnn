#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup


setup_requires = []
install_requires = [
    'chainer>=5.1.0',
    'chainercv',
]

setup(
    name='chainer_ggcnn',
    version='0.0.1',
    description='GGCNN implemenation by chainer',
    author='iory',
    author_email='ab.ioryz@gmail.com',
    url='https://github.com/iory/chainer-ggcnn',
    license='MIT License',
    packages=find_packages(),
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
)
