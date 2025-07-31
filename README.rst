#########################################################################
SAR Dirichlet : a Spatial AutoRegressive model for Dirichlet distribution
#########################################################################


Description
===========

This repository provides the implementation of a novel Spatial Autoregressive (SAR) Dirichlet regression model designed for analyzing compositional data with spatial dependencies. The model extends traditional Dirichlet regression by incorporating spatial lag terms to account for spatial autocorrelation, improving accuracy in spatially correlated datasets. The repository includes code for model fitting, parameter estimation, and performance evaluation, along with synthetic and real-world case studies to demonstrate its application.

The accompanying paper, *"Spatial Autoregressive Model on a Dirichlet Distribution"* (Nguyen et al., 2024), details the methodology and applications.  

Key features
============

- **Spatial Dirichlet Regression**:  
    - Incorporates spatial lag via the matrix :math:`M = I_n - \rho W`, where :math:`W` is a spatial weights matrix.  
    - Uses maximum likelihood estimation (MLE) with BFGS/BFGS-B optimization.  
- **Performance Metrics**:  
    - Evaluates models using :math:`R^2`, RMSE, cross-entropy, AIC, and cosine similarity.  
- **Case Studies**:  
    - Arctic Lake sediments (depth vs. composition).  
    - Maupiti Island coral reef classifications.  
    - French 2015 departmental election voting patterns.  
- **Benchmarks**:  
    - Compares against non-spatial Dirichlet, multinomial, and spatial logistic normal models. 

Repository Structure  
==================== 
::

  /src/          # Core R/Python code for model fitting and utilities  
  /data/         # Synthetic and real datasets (Arctic Lake, Maupiti, elections)  
  /examples/     # Tutorials and reproducibility scripts  
  /results/      # Outputs from case studies (figures, tables)  

Dependencies  
============  
- **R** (≥ 4.0): `DirichletReg`, `spdep`, `Matrix`  
- **Python** (optional): `numpy`, `scipy` for additional analysis  


Acknowledgments
===============

The authors would like to thank the GLADYS research team (https://www.gladys-littoral.org/) for the Maupiti data. The Maupiti satellite image comes from Pléiades satellite. Pléiades © CNES _ 2021, Distribution AIRBUS DS

The authors would also like to thank the researchers from Toulouse School of Economics who furnished the Occitanie Election dataset. More information on this dataset can be found in their paper: *Goulard, M., Laurent, T., & Thomas-Agnan, C. (2017). About predictions in spatial autoregressive models: Optimal and almost optimal strategies. Spatial Economic Analysis, 12(2-3), 304-325*.
