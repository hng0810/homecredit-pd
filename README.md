# LIGHTGBM CLASSIFIER WITH HOME CREDIT DATASET

## Introduction

<p align="justify">
  Classification model used in the project was LightGBM. The goal was to predict clients’ repayment abilities by accurately identifying creditworthy individuals.
</p>
<p align="justify">
  The approach focusing on aggregating behavioral patterns from bureau and prev_application tables.
</p>
<p align="justify">
  Model validation was performed using K-Fold CV while hyperparameter optimization was carried out using Optuna with random trial strategy. The most influential features identified by ensembled LightGBM (mean predictions from each fold) were EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3.
</p>
<p align="justify">
  The final model achieved ROC AUC: 0.7779 and Gini: 0.5557 on the test set. Score on the OOT set is 0.7612 (Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target, the final results based on 80% of the OOT set).
</p>
<p align="justify" style="text-indent: 2em;">
  <em>Notes: Several aspects in the modeling process I can improve including EDA and advanced feature engineering techniques to find hidden patterns and reduce noise. Additional methods such as undersampling, oversampling, and bootstrapping were also considered to address class imbalance and further enhance model performance. </em>
</p>

Find my notebook here: [NOTEBOOK](/notebook)


## About the competition
<p align="justify">
  Data sources: (https://www.kaggle.com/competitions/home-credit-default-risk/data)
</p>

![schema](docs/home_credit.png)
