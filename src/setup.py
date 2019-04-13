from setuptools import setup, find_packages

setup(
    name='dqnroute',
    version='0.1.0',
    description='Simple routing simulator in Python',
    author='Dmitrii Mukhutdinov',
    author_email='flyingleafe@gmail.com',
    packages=find_packages(),
    install_requires=(
        'more-itertools>=4.3.0',
        'networkx>=2.3',
        'numpy>=1.15.3',
        'pyyaml>=4.2b1',
        'scikit-learn>=0.20.0',
        'scipy>=1.1.0',
        'simpy>=3.0.11',
        'torch>=1.0.1',
        'typing>=3.6.6'
    )
)
