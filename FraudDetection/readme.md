# FRAUD DETECTION SYSTEM
## CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
### WORK IN PROGRESS...

**Problem Statement**: To improve performance of model for credit card fraud detection. 
**Value Proposition**: A better model will prevent fraudulent transactions thereby improving company profits.

# CONTENTS OF THIS REPOSITORY

## Readme

## Systems Planning & Requirements

## Jupyter Notebooks
- ** Exploratory Data Analysis**
- ** Model Performance Analysis**


## Python Files
- **data_pipeline.py** : For data engineering
- **dataset.py**: For partitioning data 
- **metrics.py**: For calculating various metrics
- **model.py** : For model training
- **fraud_detection_service.py**: a wrapper for testing new data

## Pickle Files
- Three pickle files for each of three models - random forest, logistic regression and an ensemble model.

## Text Files
-Files have model metrics that were generated while training

## Miscellaneous Files
- Various .csv files for testing
-original data file (transactions.csv)

## [Docker Image](https://hub.docker.com/layers/msherekar/705.603spring24/Fraud_Detection_System/images/sha256-02b1a1cc6cebacb6ed5e4e868fd9e2fa6fc24516351b106dab90430a0770d8c5?context=repo)

# NOTE
The code has some problems. This project is still incomplete. Models have been trained but there are issues when new data is transformed.