from setuptools import setup, find_packages

setup(
    name="deepcore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=1.0',
        'numpy>=1.0',
        'sdv>=1.0',
        'networkx>=1.0',
        'community>=1.0',
    ],
    author="Deep Found",
    description="Library for datascience and for data manipulation",
    keywords="burger_king_gavno",
    python_requires=">=3.6",
)