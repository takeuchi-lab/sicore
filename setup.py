from setuptools import find_packages, setup

setup(
    name="sicore",
    version="0.1.0",
    description="Core package for Selective Inference",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "numpy>=1.19.1",
        "mpmath>=1.1.0",
        "matplotlib>=3.3.1",
        "scipy>=1.5.2",
        "statsmodels>=0.11.1",
    ],
    python_requires=">=3.6",
)
