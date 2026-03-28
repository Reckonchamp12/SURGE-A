from setuptools import setup, find_packages

setup(
    name="surge-a",
    version="0.1.0",
    description="SURGE-A: Adaptive Conformal Prediction for Time Series under Subsampling",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "torch>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "requests>=2.28",
        "tqdm>=4.65",
    ],
)
