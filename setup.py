# setup.py

from setuptools import setup, find_packages

# read requirements.txt as use as install_requires
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name='automl',
    version='0.1',
    packages=[
        'automl',
    ],
    install_requires=install_requires,
)
