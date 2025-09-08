# install using 'pip install -e .'

from setuptools import setup

setup(name='transformer',
      version='1.0',
      packages=['transformer'],
      package_dir={'transformer': 'transformer'},
      describtion="transformer model",
      install_requires=['torch==1.8.0',
                        'tqdm',
                        'torchtext==0.9.0',
                        'spacy==3.1.3',
                        'Pydantic==1.7.4',
                        'pandas'],
      )
