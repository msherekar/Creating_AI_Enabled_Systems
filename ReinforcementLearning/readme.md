# EMAIL MARKETING SYSTEM
## FIFTH CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
 

**Problem Statement**: To improve response rates to campaign marketing emails. 
**Value Proposition**: Changing subject lines dynamically to obtain better response rates.

# CONTENTS OF THIS REPOSITORY

## Readme

## Systems Planning & Requirements

## Jupyter Notebooks
- ** Model Training**

## Python Files
- **preprocess.py** : For data engineering
- **qagent.py**: For Q learning agent class 
- **metrics.py**: For calculating various metrics
- **reinforcement.py** : For model training
- **deployment.py**: The systems level file that executes the whole code

## Model Files
- **q_table.pkl**

## Miscellaneous Files
- Docker File

## [Docker Image](https://hub.docker.com/repository/docker/msherekar/705.603spring24/general)

# INSTUCTIONS TO RUN
- Download the container using this command on command line: docker pull msherekar/705.603spring24:ReinforcementLearning
- Run this command to run the container: docker run -it -p 5000:5000 msherekar/705.603spring24:ReinforcementLearning
- Use POST to send .csv files to the container. Make sure to have this address http://localhost:5000/suggest_subject_lines and name of key should be **new_state**



