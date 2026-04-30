# Bayesian-Optimized Gradient Boosting for Seismic Soil Liquefaction Prediction on the LIQ/7/2833 Global Database

**XGBoost and LightGBM with Geographic Stratified Cross-Validation and SHAP Interpretation**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-green)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)


## Abstract

This repository implements a focused machine learning pipeline for binary prediction of seismic soil liquefaction using Bayesian-optimized gradient boosting methods (XGBoost and LightGBM) on the LIQ/7/2833 global database (2,833 case histories from 15 countries spanning 50+ earthquakes). Both algorithms natively handle the structured missing data pattern inherent to multi-test-type geotechnical databases, eliminating the need for test-type subsetting or imputation. All models are evaluated under **StratifiedGroupKFold** cross-validation grouped by geographic region to prevent spatial information leakage. SHAP (SHapley Additive exPlanations) analysis validates that learned feature importance rankings align with classical geotechnical engineering principles from the Seed-Idriss simplified procedure.

**Key Result:** LightGBM achieves AUC-ROC of 0.854 (+/- 0.041) under geographic stratified 5-fold CV, with Recall(L=1) = 0.891 at a safety-first threshold of 0.35. SHAP analysis confirms CSR_corrected (seismic demand) as the dominant feature, followed by the three resistance parameters ((N1)60, qt1N, VS1), validating physical interpretability.

## Graphical Abstract
