# SENTIMENT ANALYSIS SYSTEM
## FOURTH CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
 

**Problem Statement**: To create a sentiment analysis system for analysing movie reviews. 
**Value Proposition**: An sentiment analysis system will enable better prediction of movie ratings.

# CONTENTS OF THIS REPOSITORY

## Readme

## Systems Planning & Requirements

## Jupyter Notebooks
- ** Exploratory Data Analysis**
- ** Data Processing**
- ** Random Forest Model Training **


## Python Files
- **data_pipeline.py** : For data engineering
- **dataset.py**: For partitioning data 
- **metrics.py**: For calculating various metrics
- **model.py** : For model training
- **deployment.py**: The systems level file that executes the whole code

## Model Files
- **rf_basic.pkl**
- **word2vecmodel**

## Miscellaneous Files
- Docker File


## [Docker Image](https://hub.docker.com/repository/docker/msherekar/705.603spring24/general)

# INSTUCTIONS TO RUN
- Download the container using this command on command line: docker pull msherekar/705.603spring24:SentimentAnalysis
- Run this command to run the container: docker run -it -p 8786:8786 msherekar/705.603spring24:SentimentAnalysis
- Use POST to send .csv files to the container. Make sure to have this address http://localhost:8786/predict and name of key should be *text*



