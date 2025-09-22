from setuptools import setup, find_packages

setup(
    name="weakly_supervised_learning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.2",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
    ],
    python_requires=">=3.7",
) 