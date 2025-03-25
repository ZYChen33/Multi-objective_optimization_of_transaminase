# Machine learning-guided protein engineering for the co-optimization of transaminase’s catalytic activity and stereoselectivity
This repository contains the code and data for the study **"Machine learning-assisted protein engineering for the co-optimization of transaminase’s catalytic activity and stereose-lectivity"**. It includes tools for first-round single-objective model training and multi-objective virtual screening of S-type variants.

## Requirements
To run the scripts provided in this repository, you'll need the following Python libraries:
- Python  3.6+
- scikit-learn >= 0.24.2
- scikit-optimize >= 0.9.0
- pandas >= 1.1.5
- numpy >= 1.19.5
- matplotlib >= 3.3.4
- pareto >= 1.1.1.post3

## Installation Guide

You can install the required Python packages using the following commands: 
pip install <package_name>==<version_number>

For example, to install scikit-learn version 0.24.2, you would run:
pip install scikit-learn==0.24.2

Ensure you replace <package_name> with the name of the required library and <version_number> with the specific version you need. It's important to use the correct versions to ensure compatibility with the script.

## Usage
### Downloading and Setting Up
   To get started, download the entire repository by:
   - Clicking the `Code` button at the top right of the page, then selecting `Download ZIP`.
   - After downloading, unzip the file to a directory where you want to store the code and data.

### Training ML Models and Virtual Screening
   This repository contains the code and data for the **first-round model training and virtual screening of S-type transaminases** as described in the article.
   - Primary workflow: train_screening.ipynb (Jupyter Notebook).
   - Data: Training data: train.csv.
           Exploration space: explore.csv.
   - Outputs: Results are saved in the output/ directory.

   To use your own data, please replace train.csv and explore.csv with your custom datasets and run train_screening.ipynb (ensure dependencies are installed).

## Citation
If you use this work, please cite: [DOI]

