# Description

This repository contains the Python codes, ML data, and discovered B-N-O compounds in our study for the search of new B-N-O ternary superhard materials. For more information, please find our paper "Discovering Superhard B-N-O Compounds by Iterative Machine Learning and Evolutionary Structure Predictions" on **ACS Omega 2022, 7, 24, 21035–21042**:https://doi.org/10.1021/acsomega.2c01818

or on ArXiv:https://arxiv.org/abs/2111.12923.

# Data

The **data.json** file contains the final data set in our iterative ML procedure. The data information includes crystal structure, cohesive energy, volumetric density, and hardness.

# Codes

As long as data.json is prepared, we can enter the code folders to build ML models for cohesive energy, volumetric density, and hardness. Please go to each folder for details.

Note: The hardness model depends on the volumetric density model (rf_density.joblib), so we need to first build the density model before the hardness model.

# Environment
  - python 3.7
  - pymatgen 2021.2.16
  - sklearn 0.23.2
  - ternary 1.0.8
