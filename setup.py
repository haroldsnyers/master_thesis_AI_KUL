from pathlib import Path
from setuptools import setup, find_packages

with open('README.md') as f:
  readme = f.read()

with open('LICENSE') as f:
  license = f.read()

setup(
    name='torch_scbasset',
    version='',
    description='',
    long_description=readme,
    author='',
    author_email=', ',
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=[
        l.strip() for l in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ]
)
