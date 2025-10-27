# GENBAIT: Genetic Algorithm for Bait Selection

[![DOI](https://zenodo.org/badge/865738565.svg)](https://doi.org/10.5281/zenodo.16579445)
![build](https://img.shields.io/badge/Build-passing-brightgreen)
![Python version](https://img.shields.io/badge/Python-3.10-blue)

GENBAIT is a Python package designed for bait (feature) selection in proximity labeling data using genetic algorithms. 

A **preprint** describing the method and introducing a novel benchmarking platform is available:  
[Kasmaeifar et al. (Nature Communications, 2025) _Computational design and evaluation of optimal bait sets for scalable proximity proteomics_](https://www.nature.com/articles/s41467-025-64383-1)

---

## Overview

![GENBAIT Overview](https://github.com/camlab-bioml/genbait/blob/main/overview_figure.png)

---

## Requirements

GENBAIT requires Python 3.10 or higher. We recommend creating a virtual environment to ensure smooth installation.

---

## Git Installation

To install GENBAIT, you first need Git. Follow the instructions below to install Git on your system.

### For Windows

1. **Download the Git installer**:
   - Go to the official Git website: [https://git-scm.com/download/win](https://git-scm.com/download/win).
   - Download the latest installer for Windows.

2. **Run the installer**:
   - Locate the downloaded file and double-click to open the installer.
   - Follow the prompts in the setup wizard. You can keep the default options or customize the installation.

### For macOS

1. **Install Git using Homebrew**:
   - If you have **Homebrew** installed, open the **Terminal** and run:
     ```bash
     brew install git
     ```
   - Homebrew will handle the download and installation of Git.

2. **Verify the installation**:
   - In **Terminal**, type the following command and press Enter:
     ```bash
     git --version
     ```
   - You should see a Git version number, confirming that Git is installed.

---

## Build Tools Installation

Some GENBAIT dependencies (such as `shap`, `xgboost`, or `leidenalg`) require system-level C++ build tools to compile extensions.

### For Windows

1. **Download Microsoft C++ Build Tools**:  
   [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

2. **In the installer, select the following**:
   - **C++ build tools** workload
   - **MSVC v14 or later** (e.g., v142 or v143)
   - **Windows 10 or 11 SDK**

3. **Install the components** and restart your terminal (or system).

> If these tools are missing, you may encounter errors like:  
> `error: Microsoft Visual C++ 14.0 or greater is required`

### For macOS

1. **Install Xcode Command Line Tools**:  
   Open Terminal and run:
   ```bash
   xcode-select --install

---

## Installation

To install the `genbait` package, follow these steps:

1. **Install GENBAIT from GitHub using pip**:
    ```bash
    pip install git+https://github.com/camlab-bioml/genbait.git
    ```

2. **Ensure all dependencies are installed**:
    GENBAIT requires the following Python packages:
    - pandas
    - numpy
    - scipy
    - scikit-learn
    - matplotlib
    - seaborn
    - gprofiler-official
    - igraph
    - leidenalg
    - deap (for Genetic Algorithm operations)
    - pytorch
    - pytorch-lightning 
    - shap
    - XGBoost

These packages will be installed automatically during the setup.

Installation takes less than 2 minutes.

---

## Getting started

A detailed tutorial of how to use different functions of the package can be found here: [GENBAIT Tutorial](https://github.com/camlab-bioml/genbait/blob/main/tutorials/GENBAIT_tutorial.ipynb)

### Expected run time
For 200 baits and 10 iterations for a panel size 50, running genbait takes approximately 30 minutes on a computer with 32 GB RAM.

## Authors

This software is authored by: Vesal Kasmaeifar, Kieran R Campbell  

Lunenfeld-Tanenbaum Research Institute & University of Toronto

## Citation

If you use GENBAIT, please cite the archived version:
https://doi.org/10.5281/zenodo.16579445