# Nonlinear ML project
Noninear ML is a framework to discover nonlinear relationships between independent variables such as financial factors and the corresponding dependent variable. The framework is developed to improve the existing investment model used in financial sector.

Given a set of features, a common investment model uses linear or logistic regression regression to predict return on investment, and any non-linear and interaction terms are built into features through feature engineering by financial intuitions and fundamental analysis. This is very common in social science or medical field where the relationship between dependent and independent variables is difficult to model.

With modern machine learning techniques, this tool is developed to explore the nonlinear relationship between dependent and independent variables.

# Method
First, given a set of features and the target label, classification or regression models are trained using various machine learning models (linear and logistic regression, tree-based models, svm, knn, neural nets, etc.). In the context of financial sector, features will be financial factors and the target is return on investment.

To avoid over-fitting due to correlation between adjacent dates in time-series data, purged cross-validation method[1] is used in training procedure. Once models are trained, the performance of models are compared between linear models (base-line) and machine learning models. Machine learning models are expected to perform better than linear models. Then interpretable machine learning methods[2] are used to extract nonlinear relationship. 

- Decision boundaries
- Partial dependence plots

# Dependencies
The following packages are required.

- [pandas](https://pandas.pydata.org) >= 0.25.0
- [sikit-learn](https://scikit-learn.org) >= 0.21.3
- [seaborn](https://seaborn.pydata.org) >= 0.9
- [statsmodels](https://www.statsmodels.org) >= 0.10.1
- [tensorflow](https://www.tensorflow.org) >= 2.0.0b1
- [xgboost](https://github.com/dmlc/xgboost) >= 0.90
- [cycler](https://pypi.org/project/Cycler/) >= 0.10
- [scipy](https://www.scipy.org) >= 1.3.1


# Install
Conda environment is recommended to run this package.

- Creating conda environment:

 ``` bash
 conda env create -f environment.yml
 source activate nonlinear
 ```
- Install NonlinearML using pip:

 After cloning or downloading the package, run

 ``` bash
 pip install NonlinearML
 ```

# Version history
## 1.03
- Fixed a bug on backtesting. Now replaced dictionary with list to specify top-bottom class so that it avoids potential mis-calculation of return
- Updated single factor model

## 1.02
- Implemented additional metrics: Risk-adjusted return, combined z-score.
- Added options for selecting CV metric
- Added correlation plot in CV study

## 1.0
- Initial version

# Contact
Please contact siinn.che[AT]gmail.com for more information.



---
[1]: M. L. de Prado, Advances in financial machine learning, John Wiley & Sons, 2018.

[2]: C Molnar - A Guide for Making Black Box Models Explainable, 2018
