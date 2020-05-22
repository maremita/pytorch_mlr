from setuptools import setup, find_packages
from pytorch_mlr import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='pytorch_mlr',
    version=_version,
    description='Pytoch implementation of multinomial logistic regression',
    author='remita',
    author_email='amine.m.remita@gmail.com',
    packages=find_packages(),
    #scripts=['scripts/seq_mlr.py'],
    install_requires=INSTALL_REQUIRES
)
