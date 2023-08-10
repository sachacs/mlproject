from setuptools import find_packages, setup
from typing import List

HYPHEN_EDOT = '-e .'
def get_requirements()-> List[str]:
    '''
    get requirements from requirements.txt
    :return requirements: list of requirements
    '''
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()

    return requirements.remove(HYPHEN_EDOT)

setup(
    name='mlproject',
    version='0.0.1',
    author='Sacha',
    author_email='sacha.santos@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)