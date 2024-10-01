from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='genbait',
    version='1.0.0',
    author='Vesal Kasmaeifar',
    author_email='vesal.kasmaeifar@mail.utoronto.ca',
    description='A python package for proximity labeling data feature selection',
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
    ],
    license='MIT',
)
