import platform
import sys
from setuptools import setup, find_packages

# Print platform-specific warnings
if platform.system() == "Windows":
    print(
        "\n[GENBAIT INSTALL WARNING - Windows]\n"
        "Some dependencies (e.g., shap, xgboost) require Microsoft C++ Build Tools.\n"
        "If installation fails, please download and install them from:\n"
        "👉 https://visualstudio.microsoft.com/visual-cpp-build-tools/\n",
        file=sys.stderr
    )
elif platform.system() == "Darwin":
    print(
        "\n[GENBAIT INSTALL NOTE - macOS]\n"
        "If installation fails with compiler or header errors (e.g., Python.h not found), run:\n"
        "👉 xcode-select --install\n"
        "to install Xcode Command Line Tools.\n",
        file=sys.stderr
    )

# Read long description
with open("README.md", "r") as fh:
    long_description = fh.read()

# Setup configuration
setup(
    name='genbait',
    version='1.0.0',
    author='Vesal Kasmaeifar',
    author_email='vesal.kasmaeifar@mail.utoronto.ca',
    description='A python package for proximity labeling data feature (bait) selection',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/vesalkasmaeifar/genbait',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
        'seaborn',
        'matplotlib',
        'deap',
        'gprofiler-official',
        'igraph',
        'leidenalg',
        'XGBoost'
        'torch',
        'pytorch_lightning',
        'shap',
        
    ],
    license='MIT',
)
