#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='steerable_pdo_experiments',
    version='0.0.1',
    description='Experiments on equivariant differential operators',
    author='Erik Jenner',
    author_email='erik.jenner99@gmail.com',
    url='https://github.com/ejnnr/steerable_pdo_experiments',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)