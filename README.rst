#########################################################################
SAR Dirichlet : a Spatial AutoRegressive model for Dirichlet distribution
#########################################################################

Description
===========

This repository provides the implementation of a novel Spatial Autoregressive (SAR) Dirichlet regression model designed for analyzing compositional data with spatial dependencies. The model extends traditional Dirichlet regression by incorporating spatial lag terms to account for spatial autocorrelation, improving accuracy in spatially correlated datasets. The repository includes code for model fitting, parameter estimation, and performance evaluation, along with synthetic and real-world case studies to demonstrate its application.

Key features
============

* Spatial Dirichlet Regression: Implements a Dirichlet regression model augmented with spatial lag terms to handle spatially dependent compositional data.
* Maximum Likelihood Estimation: Uses BFGS and BFGS-B algorithms for parameter optimization, including spatial correlation strength $ρ$ and regression coefficients.
* Performance Metrics: Includes tools for evaluating model performance using metrics such as $R^2$, RMSE, cross-entropy, and cosine similarity.
* Case Studies: Features applications to three real-world datasets : arctic lake sediments, coral reef classifications, and election voting patterns.
* Comparison with Alternatives: Benchmarks against non-spatial Dirichlet models, multinomial regression, and spatial logistic normal models.


Acknowledgments
===============

The authors would like to thank the GLADYS research team (https://www.gladys-littoral.org/) for the Maupiti data. The Maupiti satellite image comes from Pléiades satellite. Pléiades © CNES _ 2021, Distribution AIRBUS DS

The authors would also like to thank the researchers from Toulouse School of Economics who furnished the Occitanie Election dataset. More information on this dataset can be found in their paper: *Goulard, M., Laurent, T., & Thomas-Agnan, C. (2017). About predictions in spatial autoregressive models: Optimal and almost optimal strategies. Spatial Economic Analysis, 12(2-3), 304-325*.
