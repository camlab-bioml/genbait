# GENBAIT: Genetic Algorithm for Bait Selection

![build](https://img.shields.io/badge/Build-passing-brightgreen)
![Python version](https://img.shields.io/badge/Python-3.10-blue)

GENBAIT is a Python package designed for bait (feature) selection in proximity labeling data using genetic algorithms. 

---
## Overview

![GENBAIT Overview](https://github.com/vesalkasmaeifar/genbait/blob/main/overview_figure.png)

## Requirements

GENBAIT requires Python 3.10 or higher. We recommend creating a virtual environment to ensure smooth installation.

### Optional Dependencies

For Gene Ontology (GO) analysis, you'll need to install the `gprofiler-official` package, which is included with the GENBAIT dependencies.

## Installation

To install the `genbait` package, follow these steps:

1. **Install GENBAIT from GitHub using pip**:
    ```bash
    pip install git+https://github.com/vesalkasmaeifar/genbait.git
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

These packages will be installed automatically during the setup.

---

## Getting started

A detailed tutorial of how to use different functions of the package can be found here: [GENBAIT Tutorial](https://github.com/camlab-bioml/genbait/blob/main/GENBAIT_tutorial.ipynb)

## Authors

This software is authored by: Vesal Kasmaeifar, Kieran R Campbell  

Lunenfeld-Tanenbaum Research Institute & University of Toronto